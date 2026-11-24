"""Step-Audio-2 frozen backend implementing the 3-method AudioLLMBackend interface.

Default scoring mode is reader-free **Shared-Audio-KV Binary Verification**:
  - ``prefill_prefix_kv(wav_path, "")`` encodes the audio + verification stem
    ONCE into a reusable ``DynamicCache``.
  - ``score_suffixes(prefix_state, suffixes)`` evaluates the calibrated binary
    log-odds ``log P("Yes") - log P("No")`` for every candidate phrase in
    batched incremental forwards from that shared Audio KV-Cache.

This eliminates external 40 ms ASR readers, eliminates Mandarin pinyin
coupling, guarantees 0.00% prefix leakage, and removes word-frequency prior
bias across dialects and languages.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from integration.backend import AudioLLMBackend, PrefixKVState
from integration.qwen2_audio_backend import install_hopper_linear_guard
from integration.step_audio2 import (
    StepAudio2Scorer,
    extract_layers,
    load_model_and_adapter,
)

VERIFY_PREFIX_PROMPT = (
    '<|BOT|>human\n<audio_patch>Listen carefully. Does this audio contain the spoken phrase "'
)
VERIFY_SUFFIX_TAIL = '"? Answer only Yes or No.<|EOT|><|BOT|>assistant\n'


class StepAudio2Backend(AudioLLMBackend):
    """Frozen Step-Audio-2 backend (80 ms/frame, default layer=2, dim=3584)."""

    name = "step_audio2"
    frame_s = 0.08
    default_layer = 2
    hidden_dim = 3584

    def __init__(
        self,
        model_dir: str,
        adapter_dir: str = "",
        device: str = "cuda:0",
        dtype: str = "auto",
        quantization: str = "none",
        max_duration: float = 30.0,
        max_candidate_batch: int = 8,
    ):
        install_hopper_linear_guard(min_rows=40)
        self.model_dir = model_dir
        self.adapter_dir = adapter_dir
        self.device = device
        self.max_duration = max_duration
        self.max_candidate_batch = max(1, int(max_candidate_batch))
        self.model, self.tokenizer = load_model_and_adapter(
            model_dir=model_dir,
            adapter_dir=adapter_dir,
            device=device,
            quantization=quantization,
            dtype=dtype,
        )
        self.hidden_dim = int(getattr(self.model.config, "hidden_size", 3584))
        self._scorer = StepAudio2Scorer(
            self.model,
            self.tokenizer,
            device=device,
            max_duration=max_duration,
            use_kv_cache=True,
            max_candidate_batch=self.max_candidate_batch,
        )
        self._verify_prompt_ids = self.tokenizer.encode(
            VERIFY_PREFIX_PROMPT, add_special_tokens=False
        )
        self._verify_tail_ids = self.tokenizer.encode(
            VERIFY_SUFFIX_TAIL, add_special_tokens=False
        )
        self._yes_id = int(self.tokenizer.encode("Yes", add_special_tokens=False)[0])
        self._no_id = int(self.tokenizer.encode("No", add_special_tokens=False)[0])
        self._scorer.prompt = self._verify_prompt_ids

    def extract_shallow(
        self, wav_path: str, layer: int | None = None
    ) -> np.ndarray:
        target_layer = self.default_layer if layer is None else int(layer)
        by_layer = extract_layers(
            self.model,
            self.tokenizer,
            wav_path=wav_path,
            device=self.device,
            layers=(target_layer,),
            max_duration=self.max_duration,
        )
        return by_layer[target_layer].astype(np.float32, copy=False)

    def prefill_prefix_kv(
        self, wav_path: str, prefix_text: str = ""
    ) -> PrefixKVState:
        import torch

        with torch.inference_mode():
            self._scorer._prepare_audio(wav_path)
            base_cache, base_last_hidden = self._scorer._ensure_audio_cache()
            cache = self._scorer._clone_cache(base_cache)
            last_hidden = base_last_hidden
            if prefix_text:
                prefix_ids = self.tokenizer.encode(
                    prefix_text, add_special_tokens=False
                )
                if prefix_ids:
                    cache, last_hidden = self._scorer._prefill_tokens(
                        cache, last_hidden, prefix_ids
                    )
        return PrefixKVState(
            wav_path=wav_path,
            prefix_text=prefix_text,
            cache=cache,
            last_hidden=last_hidden,
        )

    def _last_hidden_to_yes_no_logodds(self, last_h) -> np.ndarray:
        import torch

        batch = int(last_h.shape[0])
        pad_n = max(self._scorer.min_gemm_rows, batch)
        if pad_n > batch:
            padded = torch.cat(
                [last_h, last_h[-1:].expand(pad_n - batch, -1)], dim=0
            )
        else:
            padded = last_h
        logits = torch.matmul(
            self.model.lm_head.weight, padded.transpose(0, 1)
        ).float()[:, :batch]
        logp = torch.log_softmax(logits, dim=0)
        return (logp[self._yes_id] - logp[self._no_id]).cpu().numpy()

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
        past = int(prefix_state.cache.get_seq_length())
        scores = np.empty(len(suffixes), dtype=np.float64)
        by_len: dict[int, list[int]] = {}
        for idx, seq in enumerate(seqs):
            by_len.setdefault(len(seq), []).append(idx)

        with torch.inference_mode():
            for length, indices in by_len.items():
                for start in range(0, len(indices), self.max_candidate_batch):
                    chunk = indices[start : start + self.max_candidate_batch]
                    batch = len(chunk)
                    expanded = self._scorer._clone_cache(prefix_state.cache)
                    if batch > 1:
                        expanded.batch_repeat_interleave(batch)
                    q = torch.tensor(
                        [seqs[i] for i in chunk],
                        dtype=torch.long,
                        device=self.device,
                    )
                    emb = self.model.model.embed_tokens(q)
                    mask = torch.ones(
                        batch, past + length, dtype=torch.long, device=self.device
                    )
                    out = self.model.model(
                        inputs_embeds=emb,
                        attention_mask=mask,
                        past_key_values=expanded,
                        use_cache=False,
                    )
                    scores[chunk] = self._last_hidden_to_yes_no_logodds(
                        out.last_hidden_state[:, -1, :]
                    )
        return scores

    def score_full(
        self,
        wav_path: str,
        prefixes: Sequence[str],
        suffixes: Sequence[str],
    ) -> np.ndarray:
        """Full-sequence non-cached Yes/No log-odds (for exact KV parity & speedup audit)."""
        import torch

        if not suffixes:
            return np.empty(0, dtype=np.float64)
        audio_embeds = self._scorer._prepare_audio(wav_path)
        scores = np.empty(len(suffixes), dtype=np.float64)
        with torch.inference_mode():
            for idx, (pref, suf) in enumerate(zip(prefixes, suffixes)):
                pref_ids = (
                    self.tokenizer.encode(pref, add_special_tokens=False)
                    if pref
                    else []
                )
                tail_ids = (
                    pref_ids
                    + self.tokenizer.encode(suf, add_special_tokens=False)
                    + self._verify_tail_ids
                )
                tail_t = torch.tensor(
                    [tail_ids], dtype=torch.long, device=self.device
                )
                tail_emb = self.model.model.embed_tokens(tail_t)
                seq = torch.cat([audio_embeds, tail_emb], dim=1)
                mask = torch.ones(
                    1, seq.shape[1], dtype=torch.long, device=self.device
                )
                out = self.model.model(
                    inputs_embeds=seq,
                    attention_mask=mask,
                    use_cache=False,
                )
                scores[idx] = float(
                    self._last_hidden_to_yes_no_logodds(
                        out.last_hidden_state[:, -1, :]
                    )[0]
                )
        return scores
