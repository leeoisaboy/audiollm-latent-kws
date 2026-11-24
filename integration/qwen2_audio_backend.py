"""Qwen2-Audio frozen backend implementing the 3-method AudioLLMBackend interface.

Uses the exact same reader-free **Shared-Audio-KV Binary Verification**
protocol as ``StepAudio2Backend``:
  - ``prefill_prefix_kv(wav_path, "")`` encodes the audio + verification stem
    ONCE into a reusable ``DynamicCache``.
  - ``score_suffixes(prefix_state, suffixes)`` scores calibrated binary
    log-odds ``log P("Yes") - log P("No")`` for all Top-K candidates in
    batched incremental forwards from the shared Audio KV-Cache.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from integration.backend import AudioLLMBackend, PrefixKVState

QWEN2_SAMPLE_RATE = 16_000
QWEN2_VERIFY_STEM = (
    "<|im_start|>user\n"
    "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
    'Listen carefully. Does this audio contain the spoken phrase "'
)
QWEN2_VERIFY_TAIL = '"? Answer only Yes or No.<|im_end|>\n<|im_start|>assistant\n'


def install_hopper_linear_guard(min_rows: int = 40):
    """Pad tiny fp16/bf16 CUDA GEMMs to avoid Hopper/cuBLAS SIGFPE on torch 2.3.1."""
    import torch
    import torch.nn.functional as F

    current = F.linear
    if getattr(current, "_latent_kws_hopper_guard", False):
        return current
    minimum = max(40, int(min_rows))

    def guarded_linear(input, weight, bias=None):
        if not (
            input.is_cuda
            and input.dtype in (torch.float16, torch.bfloat16)
            and input.dim() >= 2
        ):
            return current(input, weight, bias)
        width = int(input.shape[-1])
        rows = input.numel() // width if width else 0
        if rows == 0:
            return current(input, weight, bias)
        if rows >= minimum and input.dtype == torch.float16:
            return current(input, weight, bias)
        flat = input.reshape(rows, width)
        if rows < minimum:
            flat = torch.cat([flat, flat[-1:].expand(minimum - rows, -1)], dim=0)
        output = current(flat, weight, None)
        if bias is not None:
            output = output + bias
        if rows < minimum:
            output = output[:rows]
        return output.reshape(*input.shape[:-1], output.shape[-1])

    guarded_linear._latent_kws_hopper_guard = True
    F.linear = guarded_linear
    return guarded_linear


def _load_mono_16k(wav_path: str, max_duration: float = 30.0) -> np.ndarray:
    import librosa
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != QWEN2_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=QWEN2_SAMPLE_RATE)
    if max_duration > 0:
        limit = int(round(max_duration * QWEN2_SAMPLE_RATE))
        if len(audio) > limit:
            audio = audio[:limit]
    return np.asarray(audio, dtype=np.float32)


class Qwen2AudioBackend(AudioLLMBackend):
    """Frozen Qwen2-Audio-7B-Instruct backend (40 ms/frame, default audio_tower layer=-2)."""

    name = "qwen2_audio"
    frame_s = 0.04
    default_layer = -2
    hidden_dim = 1280

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        dtype: str = "float16",
        max_duration: float = 30.0,
        max_candidate_batch: int = 8,
    ):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        install_hopper_linear_guard(min_rows=40)
        self.model_dir = model_dir
        self.device = device
        self.max_duration = max_duration
        self.max_candidate_batch = max(1, int(max_candidate_batch))
        self.compute_dtype = (
            torch.float16
            if dtype in ("auto", "float16")
            else (torch.bfloat16 if dtype == "bfloat16" else torch.float32)
        )

        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=self.compute_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.hidden_dim = int(self.model.config.text_config.hidden_size)
        self.audio_token_id = int(self.model.config.audio_token_index)
        self.prompt_text = QWEN2_VERIFY_STEM
        self._verify_tail_ids = self.tokenizer.encode(
            QWEN2_VERIFY_TAIL, add_special_tokens=False
        )
        self._yes_id = int(self.tokenizer.encode("Yes", add_special_tokens=False)[0])
        self._no_id = int(self.tokenizer.encode("No", add_special_tokens=False)[0])
        self._cached_wav: str | None = None
        self._cached_inputs: dict | None = None
        self._cached_audio_kv = None
        self._cached_audio_last_hidden = None

    def _prepare_inputs(self, wav_path: str):
        if self._cached_wav == wav_path and self._cached_inputs is not None:
            return self._cached_inputs
        audio = _load_mono_16k(wav_path, self.max_duration)
        if len(audio) < 4000:
            audio = np.pad(audio, (0, 4000 - len(audio)))
        inputs = self.processor(
            text=self.prompt_text,
            audios=[audio],
            sampling_rate=QWEN2_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            k: (
                v.to(self.device, dtype=self.compute_dtype)
                if k == "input_features"
                else v.to(self.device)
            )
            for k, v in inputs.items()
        }
        self._cached_wav = wav_path
        self._cached_inputs = inputs
        self._cached_audio_kv = None
        self._cached_audio_last_hidden = None
        return inputs

    def extract_shallow(
        self, wav_path: str, layer: int | None = None
    ) -> np.ndarray:
        import torch

        target_layer = self.default_layer if layer is None else int(layer)
        inputs = self._prepare_inputs(wav_path)
        with torch.inference_mode():
            if target_layer == -2:
                # Bidirectional Whisper-large-v3 audio_tower output (no causal text mixing)
                input_features = inputs["input_features"]
                feature_attention_mask = inputs["feature_attention_mask"]
                audio_feat_lengths, audio_output_lengths = (
                    self.model.audio_tower._get_feat_extract_output_lengths(
                        feature_attention_mask.sum(-1)
                    )
                )
                _, _, max_mel_seq_len = input_features.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                seq_range = torch.arange(
                    0, max_seq_len, dtype=audio_feat_lengths.dtype, device=self.device
                ).unsqueeze(0)
                padding_mask = seq_range >= audio_feat_lengths.unsqueeze(1)
                attn_mask_bool = padding_mask.view(1, 1, 1, max_seq_len).expand(
                    1, 1, max_seq_len, max_seq_len
                )
                attn_mask = attn_mask_bool.to(
                    dtype=self.model.audio_tower.conv1.weight.dtype, device=self.device
                )
                attn_mask[attn_mask_bool] = float("-inf")
                out = self.model.audio_tower(
                    input_features, attention_mask=attn_mask
                ).last_hidden_state
                n_frames = int(audio_output_lengths[0].item())
                return out[0, :n_frames].detach().to(torch.float32).cpu().numpy()

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            state_idx = target_layer + 1
            if state_idx >= len(outputs.hidden_states):
                raise IndexError(
                    f"requested layer {target_layer}, model has {len(outputs.hidden_states) - 1} layers"
                )
            audio_mask = inputs["input_ids"][0] == self.audio_token_id
            span = outputs.hidden_states[state_idx][0, audio_mask]
            return span.detach().to(torch.float32).cpu().numpy()

    def _ensure_audio_kv(self, wav_path: str):
        import torch

        inputs = self._prepare_inputs(wav_path)
        if self._cached_audio_kv is None:
            with torch.inference_mode():
                out = self.model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=True,
                )
                self._cached_audio_kv = out.past_key_values
                self._cached_audio_last_hidden = out.hidden_states[-1][:, -1:, :]
        return self._cached_audio_kv, self._cached_audio_last_hidden

    def prefill_prefix_kv(
        self, wav_path: str, prefix_text: str = ""
    ) -> PrefixKVState:
        import torch

        with torch.inference_mode():
            base_kv, base_last = self._ensure_audio_kv(wav_path)
            cache = copy.deepcopy(base_kv)
            last_hidden = base_last
            if prefix_text:
                prefix_ids = self.tokenizer.encode(
                    prefix_text, add_special_tokens=False
                )
                if prefix_ids:
                    past_len = int(cache.get_seq_length())
                    ids_t = torch.tensor(
                        [prefix_ids], dtype=torch.long, device=self.device
                    )
                    embeds = self.model.get_input_embeddings()(ids_t)
                    mask = torch.ones(
                        1,
                        past_len + len(prefix_ids),
                        dtype=torch.long,
                        device=self.device,
                    )
                    out = self.model.language_model.model(
                        inputs_embeds=embeds,
                        attention_mask=mask,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = out.past_key_values
                    last_hidden = out.last_hidden_state[:, -1:, :]
        return PrefixKVState(
            wav_path=wav_path,
            prefix_text=prefix_text,
            cache=cache,
            last_hidden=last_hidden,
        )

    def _last_hidden_to_yes_no_logodds(self, last_h) -> np.ndarray:
        import torch

        logits = self.model.language_model.lm_head(last_h).float()
        logp = torch.log_softmax(logits, dim=-1)
        return (logp[:, self._yes_id] - logp[:, self._no_id]).cpu().numpy()

    def score_suffixes(
        self, prefix_state: PrefixKVState, suffixes: Sequence[str]
    ) -> np.ndarray:
        import torch

        if not suffixes:
            return np.empty(0, dtype=np.float64)
        seqs = [
            self.tokenizer.encode(s, add_special_tokens=False) + self._verify_tail_ids
            for s in suffixes
        ]
        past_len = int(prefix_state.cache.get_seq_length())
        values = np.empty(len(suffixes), dtype=np.float64)
        by_length: dict[int, list[int]] = {}
        for idx, ids in enumerate(seqs):
            by_length.setdefault(len(ids), []).append(idx)

        with torch.inference_mode():
            for length, indices in by_length.items():
                for start in range(0, len(indices), self.max_candidate_batch):
                    chunk = indices[start : start + self.max_candidate_batch]
                    batch = len(chunk)
                    expanded = copy.deepcopy(prefix_state.cache)
                    if batch > 1:
                        expanded.batch_repeat_interleave(batch)
                    q_ids = torch.tensor(
                        [seqs[i] for i in chunk],
                        dtype=torch.long,
                        device=self.device,
                    )
                    q_embeds = self.model.get_input_embeddings()(q_ids)
                    mask = torch.ones(
                        batch,
                        past_len + length,
                        dtype=torch.long,
                        device=self.device,
                    )
                    out = self.model.language_model.model(
                        inputs_embeds=q_embeds,
                        attention_mask=mask,
                        past_key_values=expanded,
                        use_cache=False,
                    )
                    values[chunk] = self._last_hidden_to_yes_no_logodds(
                        out.last_hidden_state[:, -1, :]
                    )
        return values

    def score_full(
        self,
        wav_path: str,
        prefixes: Sequence[str],
        suffixes: Sequence[str],
    ) -> np.ndarray:
        """Full-sequence non-cached Yes/No log-odds (for exact KV parity & speedup audit)."""
        import torch

        inputs = self._prepare_inputs(wav_path)
        base_ids = inputs["input_ids"]
        values = np.empty(len(suffixes), dtype=np.float64)
        with torch.inference_mode():
            for idx, (prefix, suffix) in enumerate(zip(prefixes, suffixes)):
                pref_ids = (
                    self.tokenizer.encode(prefix, add_special_tokens=False)
                    if prefix
                    else []
                )
                tail_ids = (
                    pref_ids
                    + self.tokenizer.encode(suffix, add_special_tokens=False)
                    + self._verify_tail_ids
                )
                tail_t = torch.tensor(
                    [tail_ids], dtype=torch.long, device=self.device
                )
                full_ids = torch.cat([base_ids, tail_t], dim=1)
                full_mask = torch.ones_like(full_ids)
                out = self.model(
                    input_ids=full_ids,
                    input_features=inputs["input_features"],
                    feature_attention_mask=inputs["feature_attention_mask"],
                    attention_mask=full_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                values[idx] = float(
                    self._last_hidden_to_yes_no_logodds(
                        out.hidden_states[-1][:, -1, :]
                    )[0]
                )
        return values
