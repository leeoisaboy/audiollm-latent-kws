from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reranker.scoring import AudioLMScorer
AUDIO_START = 151688
AUDIO_PATCH = 151690
AUDIO_END = 151689
SAMPLE_RATE = 16_000
INSTRUCTION = "<audio>\u8bf7\u8bb0\u5f55\u4e0b\u4f60\u6240\u542c\u5230\u7684\u8bed\u97f3\u5185\u5bb9\u3002"
TEXT_INSTRUCTION = "\u8bf7\u6839\u636e\u5df2\u7ed9\u8bed\u5883\u8865\u5168\u4e0b\u4e00\u4e2a\u8bcd\u3002"


def install_bf16_linear_guard(min_rows: int = 40):
    """Work around SIGFPEs in bf16 GEMMs on this Hopper/cuBLAS build.

    Two independent crashes were observed on torch 2.3.1+cu121 / H20:

    * bias-epilogue bf16 GEMMs abort for some row counts (e.g. 884 rows);
    * plain bf16 GEMMs abort when the flattened row count is below 17.

    The guard removes the fused bias epilogue and pads tiny GEMMs. It is
    idempotent and only touches CUDA bf16 inputs, so fp32/fp16 numerics are
    unchanged.
    """
    import torch
    import torch.nn.functional as functional

    current = functional.linear
    if getattr(current, "_latent_kws_bf16_guard", False):
        return current
    minimum = max(17, int(min_rows))

    def guarded_linear(input, weight, bias=None):
        if not (input.is_cuda and input.dtype == torch.bfloat16 and input.dim() >= 2):
            return current(input, weight, bias)
        width = int(input.shape[-1])
        rows = input.numel() // width if width else 0
        if rows == 0:
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

    guarded_linear._latent_kws_bf16_guard = True
    functional.linear = guarded_linear
    return guarded_linear


def load_audio(path: str, max_duration: float = 30.0):
    import torchaudio

    waveform, source_rate = torchaudio.load(path)
    if source_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(source_rate, SAMPLE_RATE)(waveform)
    audio = waveform[0]
    limit = int(max_duration * SAMPLE_RATE)
    return audio[:limit] if max_duration > 0 and audio.numel() > limit else audio


def log_mel_spectrogram(audio, padding: int = 479):
    import librosa
    import torch

    audio = torch.nn.functional.pad(audio, (0, padding))
    spectrum = torch.stft(
        audio, 400, 160, window=torch.hann_window(400).to(audio.device),
        return_complex=True,
    )
    magnitude = spectrum[..., :-1].abs() ** 2
    mel_filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=400, n_mels=128)
    mel = torch.from_numpy(mel_filters).to(audio.device) @ magnitude
    log_mel = torch.clamp(mel, min=1e-10).log10()
    return (torch.maximum(log_mel, log_mel.max() - 8.0) + 4.0) / 4.0


def audio_token_count(mel_frames: int) -> int:
    reduced = (mel_frames - 2 + 1) // 2 // 2
    return (reduced + 2 - 3) // 2 + 1


def prompt_ids(tokenizer, text: str = INSTRUCTION) -> list[int]:
    return tokenizer(
        "<|BOT|>human\n" + text.replace("<audio>", "<audio_patch>")
        + "<|EOT|><|BOT|>assistant\n",
        add_special_tokens=False,
    )["input_ids"]


def text_prompt_ids(tokenizer, text: str = TEXT_INSTRUCTION) -> list[int]:
    return tokenizer(
        "<|BOT|>human\n" + text + "<|EOT|><|BOT|>assistant\n",
        add_special_tokens=False,
    )["input_ids"]


def expand_audio_token(input_ids: Sequence[int], token_count: int) -> list[int]:
    audio_block = [AUDIO_START] + [AUDIO_PATCH] * token_count + [AUDIO_END]
    output: list[int] = []
    replaced = False
    for token_id in input_ids:
        if not replaced and token_id == AUDIO_PATCH:
            output.extend(audio_block)
            replaced = True
        else:
            output.append(int(token_id))
    if not replaced:
        raise ValueError("prompt does not contain <audio_patch>")
    return output


def load_model_and_adapter(
    model_dir: str,
    adapter_dir: str = "",
    device: str = "cuda:0",
    quantization: str = "none",
    dtype: str = "auto",
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from integration.low_memory_safetensors import low_memory_safetensors_loading

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config._attn_implementation = "eager"
    if dtype == "auto":
        # bf16 GEMMs abort with SIGFPE on this Hopper/cuBLAS combination for
        # several row counts, so CUDA defaults to fp16 (which is also strictly
        # more precise than bf16). Pass --dtype bfloat16 to opt back in.
        compute_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    elif dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    elif dtype == "float16":
        compute_dtype = torch.float16
    elif dtype == "float32":
        compute_dtype = torch.float32
    else:
        raise ValueError("dtype must be auto, bfloat16, float16, or float32")
    if quantization not in ("none", "8bit", "4bit"):
        raise ValueError("quantization must be none, 8bit, or 4bit")
    quantization_config = None
    if quantization == "8bit":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    with low_memory_safetensors_loading(enabled=quantization != "none"):
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            device_map=device,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
    if adapter_dir:
        if quantization != "none":
            raise ValueError("adapter loading requires quantization none because this runner merges the adapter")
        import peft

        model = peft.PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        model = model.merge_and_unload()
        model = model.to(compute_dtype)
    model.eval()
    install_bf16_linear_guard()
    return model, tokenizer


def _audio_span(hidden, input_ids):
    import torch

    starts = torch.nonzero(input_ids[0] == AUDIO_START).flatten()
    ends = torch.nonzero(input_ids[0] == AUDIO_END).flatten()
    start = int(starts[-1]) + 1
    return hidden[0, start : max(start + 1, int(ends[-1]))]


def extract_layer2(model, tokenizer, wav_path: str, device: str, max_duration: float = 30.0):
    """Return the frozen audio-slot states after LLM block 2 as float32."""
    import torch

    audio = load_audio(wav_path, max_duration).to(device)
    if audio.numel() < 8_000:
        audio = torch.nn.functional.pad(audio, (0, 8_000 - audio.numel()))
    mel = log_mel_spectrogram(audio).to(device)
    ids = torch.tensor(
        [expand_audio_token(prompt_ids(tokenizer), audio_token_count(mel.shape[1]))],
        dtype=torch.long,
        device=device,
    )
    hidden = model.model.embed_tokens(ids)
    encoder_dtype = next(model.encoder.parameters()).dtype
    wavs = mel.unsqueeze(0).to(dtype=encoder_dtype)
    with torch.inference_mode():
        encoded, encoder_lengths = model.encoder(
            wavs, torch.tensor([mel.shape[1] - 2], dtype=torch.int32, device=device)
        )
        adapter_dtype = next(model.adapter.parameters()).dtype
        encoded = model.adapter(encoded.to(adapter_dtype))
        slots = audio_token_count(mel.shape[1])
        upper = (int(encoder_lengths[0].item()) - 1) // 2 + 1
        length = min(encoded.shape[1], max(upper, 1), max(slots, 1))
        start = int(torch.nonzero(ids[0] == AUDIO_START).flatten()[-1]) + 1
        hidden[0, start : start + length] = encoded[0, :length].to(hidden.dtype)
        states = model.model(
            inputs_embeds=hidden,
            attention_mask=torch.ones_like(ids),
            output_hidden_states=True,
        ).hidden_states
        return extract_layers(model, tokenizer, wav_path, device, (2,), max_duration)[2]


def extract_layers(
    model,
    tokenizer,
    wav_path: str,
    device: str,
    layers: Sequence[int],
    max_duration: float = 30.0,
) -> dict[int, np.ndarray]:
    """Export selected decoder block outputs in one forward pass.

    ``layer=0`` means the output of decoder block 0. Transformers prefixes the
    embedding tensor to ``hidden_states``, so its tensor index is layer plus one.
    """
    import torch

    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("layers must contain non-negative layer numbers")
    audio = load_audio(wav_path, max_duration).to(device)
    if audio.numel() < 8_000:
        audio = torch.nn.functional.pad(audio, (0, 8_000 - audio.numel()))
    mel = log_mel_spectrogram(audio).to(device)
    ids = torch.tensor(
        [expand_audio_token(prompt_ids(tokenizer), audio_token_count(mel.shape[1]))],
        dtype=torch.long,
        device=device,
    )
    hidden = model.model.embed_tokens(ids)
    encoder_dtype = next(model.encoder.parameters()).dtype
    wavs = mel.unsqueeze(0).to(dtype=encoder_dtype)
    with torch.inference_mode():
        encoded, encoder_lengths = model.encoder(
            wavs, torch.tensor([mel.shape[1] - 2], dtype=torch.int32, device=device)
        )
        adapter_dtype = next(model.adapter.parameters()).dtype
        encoded = model.adapter(encoded.to(adapter_dtype))
        slots = audio_token_count(mel.shape[1])
        upper = (int(encoder_lengths[0].item()) - 1) // 2 + 1
        length = min(encoded.shape[1], max(upper, 1), max(slots, 1))
        start = int(torch.nonzero(ids[0] == AUDIO_START).flatten()[-1]) + 1
        hidden[0, start : start + length] = encoded[0, :length].to(hidden.dtype)
        states = model.model(
            inputs_embeds=hidden,
            attention_mask=torch.ones_like(ids),
            output_hidden_states=True,
        ).hidden_states
        output: dict[int, np.ndarray] = {}
        for layer in sorted(set(layers)):
            state_index = layer + 1
            if state_index >= len(states):
                raise IndexError(f"requested layer {layer}, but model exposes {len(states) - 1} decoder layers")
            span = _audio_span(states[state_index], ids)
            output[layer] = span.detach().to(torch.float32).cpu().numpy()
        return output


class StepAudio2Scorer(AudioLMScorer):
    """Teacher-forcing scorer with reusable audio/prefix KV caches."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        max_duration: float = 30.0,
        use_kv_cache: bool = True,
        min_gemm_rows: int = 40,
        max_candidate_batch: int = 8,
    ):
        install_bf16_linear_guard()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_duration = max_duration
        self.use_kv_cache = use_kv_cache
        # bf16 GEMMs on some Hopper/cuBLAS builds abort with SIGFPE when the
        # flattened row count is below 32. Incremental prefills are padded to
        # this width and the cache is cropped back to the real token count.
        self.min_gemm_rows = max(1, int(min_gemm_rows))
        self.max_candidate_batch = max(1, int(max_candidate_batch))
        self.prompt = prompt_ids(tokenizer)
        self.text_prompt = text_prompt_ids(tokenizer)
        self.audio = None
        self._audio_key: str | None = None
        self._audio_cache = None
        self._audio_last_hidden = None
        self._text_cache = None
        self._text_last_hidden = None
        self.last_stats = {
            "cache_enabled": bool(use_kv_cache),
            "audio_prefills": 0,
            "prefix_prefills": 0,
            "candidate_prefills": 0,
            "full_sequence_forwards": 0,
        }

    def _prepare_audio(self, wav_path: str) -> None:
        import torch

        if self._audio_key == wav_path and self.audio is not None:
            return self.audio
        audio = load_audio(wav_path, self.max_duration).to(self.device)
        if audio.numel() < 8_000:
            audio = torch.nn.functional.pad(audio, (0, 8_000 - audio.numel()))
        mel = log_mel_spectrogram(audio).to(self.device)
        slots = audio_token_count(mel.shape[1])
        ids = torch.tensor(
            [expand_audio_token(self.prompt, slots)], dtype=torch.long, device=self.device
        )
        embeddings = self.model.model.embed_tokens(ids).clone()
        encoder_dtype = next(self.model.encoder.parameters()).dtype
        wavs = mel.unsqueeze(0).to(dtype=encoder_dtype)
        encoded, encoder_lengths = self.model.encoder(
            wavs, torch.tensor([mel.shape[1] - 2], dtype=torch.int32, device=self.device)
        )
        adapter_dtype = next(self.model.adapter.parameters()).dtype
        encoded = self.model.adapter(encoded.to(adapter_dtype))
        upper = (int(encoder_lengths[0].item()) - 1) // 2 + 1
        length = min(encoded.shape[1], max(upper, 1), max(slots, 1))
        start = int(torch.nonzero(ids[0] == AUDIO_START).flatten()[-1]) + 1
        embeddings[0, start : start + length] = encoded[0, :length].to(embeddings.dtype)
        self.audio = embeddings
        self._audio_key = wav_path
        self._audio_cache = None
        self._audio_last_hidden = None
        return embeddings

    def _ensure_audio_cache(self):
        import torch

        if not self.use_kv_cache:
            raise RuntimeError("KV cache is disabled")
        if self._audio_cache is None:
            hidden, cache = self._forward_hidden(
                inputs_embeds=self.audio,
                attention_mask=torch.ones(1, self.audio.shape[1], dtype=torch.long, device=self.device),
            )
            self._audio_cache = cache
            self._audio_last_hidden = hidden[:, -1:, :]
            self.last_stats["audio_prefills"] += 1
        return self._audio_cache, self._audio_last_hidden

    def _forward_hidden(
        self, inputs_embeds, attention_mask=None, past_key_values=None, use_cache=True
    ):
        import torch

        if not inputs_embeds.is_contiguous():
            inputs_embeds = inputs_embeds.contiguous()
        real_rows = int(inputs_embeds.shape[1])
        past_rows = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        total_rows = max(real_rows, self.min_gemm_rows)
        if total_rows > real_rows:
            pad = inputs_embeds.new_zeros(
                (inputs_embeds.shape[0], total_rows - real_rows, inputs_embeds.shape[-1])
            )
            inputs_embeds = torch.cat([inputs_embeds, pad], dim=1)
            if attention_mask is None:
                attention_mask = torch.ones(
                    1, past_rows + total_rows, dtype=torch.long, device=self.device
                )
            elif attention_mask.shape[1] == past_rows + real_rows:
                pad_mask = torch.zeros(
                    (attention_mask.shape[0], total_rows - real_rows),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
            else:
                attention_mask = torch.ones(
                    1, past_rows + total_rows, dtype=torch.long, device=self.device
                )
        output = self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=(
                None
                if attention_mask is None
                else attention_mask.long()
                .cumsum(dim=-1)
                .sub(1)
                .clamp_min(0)[:, -inputs_embeds.shape[1] :]
            ),
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        cache = output.past_key_values
        if (
            use_cache
            and total_rows > real_rows
            and cache is not None
            and hasattr(cache, "crop")
        ):
            cache.crop(past_rows + real_rows)
        return output.last_hidden_state[:, :real_rows, :], cache

    @staticmethod
    def _clone_cache(cache):
        import copy

        return copy.deepcopy(cache)

    def _prefill_tokens(self, cache, last_hidden, token_ids: Sequence[int]):
        import torch

        if not token_ids:
            return cache, last_hidden
        past_length = int(cache.get_seq_length())
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        embeddings = self.model.model.embed_tokens(ids)
        hidden, new_cache = self._forward_hidden(
            inputs_embeds=embeddings,
            attention_mask=torch.ones(1, past_length + len(token_ids), dtype=torch.long, device=self.device),
            past_key_values=cache,
        )
        self.last_stats["prefix_prefills"] += 1
        return new_cache, hidden[:, -1:, :]

    def _candidate_log_probability(self, cache, last_hidden, candidate_ids: Sequence[int]) -> float:
        import torch

        ids = list(int(value) for value in candidate_ids)
        if not ids:
            return float("-inf")
        past_length = int(cache.get_seq_length())
        hidden = last_hidden
        if len(ids) > 1:
            cache = self._clone_cache(cache)
            prompt = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
            embeddings = self.model.model.embed_tokens(prompt)
            hidden_out, _ = self._forward_hidden(
                inputs_embeds=embeddings,
                attention_mask=torch.ones(
                    1, past_length + len(ids) - 1, dtype=torch.long, device=self.device
                ),
                past_key_values=cache,
            )
            hidden = torch.cat([last_hidden, hidden_out], dim=1)
            self.last_stats["candidate_prefills"] += 1
        picked = hidden.reshape(-1, hidden.shape[-1])
        candidate_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        selected = self._vocab_logprobs(picked, candidate_tensor)
        return float(selected.mean().item())

    def _vocab_logprobs(self, picked, candidate_tensor):
        """Return per-position log-probabilities for the requested tokens.

        The projection is padded to ``min_gemm_rows`` columns because the same
        cuBLAS build also aborts on a single-column bf16 GEMM.
        """
        import torch

        real = int(picked.shape[0])
        if real < self.min_gemm_rows:
            picked = torch.cat(
                [picked, picked[-1:].expand(self.min_gemm_rows - real, -1)], dim=0
            )
        logits = torch.matmul(self.model.lm_head.weight, picked.transpose(0, 1)).float()
        log_probabilities = torch.log_softmax(logits[:, :real], dim=0)
        return log_probabilities[
            candidate_tensor, torch.arange(real, device=self.device)
        ]

    def _score_prefix_group(self, base_cache, base_last_hidden, prefix: str, candidates):
        cache = self._clone_cache(base_cache)
        last_hidden = base_last_hidden
        if prefix:
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            cache, last_hidden = self._prefill_tokens(cache, last_hidden, prefix_ids)
        return self._score_candidates_batched(cache, last_hidden, candidates)

    def _hidden_logprobs(self, hidden, candidates):
        picked = hidden.reshape(-1, hidden.shape[-1])
        import torch

        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=self.device).reshape(-1)
        selected = self._vocab_logprobs(picked, candidate_tensor)
        return selected.reshape(len(candidates), -1).mean(dim=1).cpu().numpy()

    def _score_candidates_batched(self, cache, last_hidden, candidates):
        """Score candidates from one shared prefix cache in short batches.

        All candidates of the same token length are evaluated in one forward
        pass. Rows are padded up to ``min_gemm_rows`` by duplicating the last
        sequence; duplicated rows never influence the retained rows.
        """
        import torch

        values = np.empty(len(candidates), dtype=np.float64)
        by_length: dict[int, list[int]] = {}
        for index, ids in enumerate(candidates):
            by_length.setdefault(len(ids), []).append(index)
        for length, indexes in by_length.items():
            for start in range(0, len(indexes), self.max_candidate_batch):
                chunk = indexes[start : start + self.max_candidate_batch]
                chunk_ids = [candidates[index] for index in chunk]
                values[chunk] = self._score_candidate_chunk(cache, last_hidden, chunk_ids)
        return values

    def _score_candidate_chunk(self, cache, last_hidden, candidates):
        import math

        import torch

        batch = len(candidates)
        length = len(candidates[0])
        if length == 1:
            hidden = last_hidden.expand(batch, -1, -1)
            return self._hidden_logprobs(hidden, candidates)
        query_length = length - 1
        past = int(cache.get_seq_length())
        padded_length = max(query_length, math.ceil(self.min_gemm_rows / batch))
        expanded = self._clone_cache(cache)
        if batch > 1:
            # transformers' DynamicCache mutates in place and returns None.
            expanded.batch_repeat_interleave(batch)
        prompt = torch.tensor(
            [ids[:query_length] for ids in candidates], dtype=torch.long, device=self.device
        )
        if padded_length > query_length:
            padding = torch.zeros(
                (batch, padded_length - query_length), dtype=torch.long, device=self.device
            )
            prompt = torch.cat([prompt, padding], dim=1)
        embeddings = self.model.model.embed_tokens(prompt)
        attention_mask = torch.ones(batch, past + padded_length, dtype=torch.long, device=self.device)
        if padded_length > query_length:
            attention_mask[:, past + query_length :] = 0
        output = self.model.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            past_key_values=expanded,
            use_cache=True,
        )
        hidden = torch.cat(
            [last_hidden.expand(batch, -1, -1), output.last_hidden_state[:, :query_length]], dim=1
        )
        self.last_stats["candidate_prefills"] += 1
        return self._hidden_logprobs(hidden, candidates)

    def _score_group(self, audio, prefixes: Sequence[str], candidate_ids: Sequence[Sequence[int]]):
        import torch

        candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)
        candidate_embeddings = self.model.model.embed_tokens(candidate_tensor)
        _, token_length, dimension = candidate_embeddings.shape
        prefix_embeddings = [
            self.model.model.embed_tokens(
                torch.tensor([self.tokenizer.encode(prefix, add_special_tokens=False)], device=self.device)
            )[0]
            if prefix else None
            for prefix in prefixes
        ]
        widths = [0 if value is None else value.shape[0] for value in prefix_embeddings]
        longest = max(widths)
        batch_size = len(prefixes)
        tail = torch.zeros(batch_size, longest + token_length, dimension, dtype=candidate_embeddings.dtype, device=self.device)
        mask = torch.zeros(batch_size, longest + token_length, dtype=torch.long, device=self.device)
        for index, value in enumerate(prefix_embeddings):
            pad = longest - widths[index]
            if value is not None:
                tail[index, pad:longest] = value
                mask[index, pad:longest] = 1
            tail[index, longest:] = candidate_embeddings[index]
            mask[index, longest:] = 1
        sequence = torch.cat([audio.expand(batch_size, -1, -1), tail], dim=1)
        attention_mask = torch.cat(
            [torch.ones(batch_size, audio.shape[1], dtype=torch.long, device=self.device), mask], dim=1
        )
        hidden, _ = self._forward_hidden(
            sequence, attention_mask, use_cache=False
        )
        offset = audio.shape[1] + longest - 1
        picked = hidden[:, offset : offset + token_length, :].reshape(-1, dimension)
        flat_candidates = candidate_tensor.reshape(-1)
        selected = self._vocab_logprobs(picked, flat_candidates)
        return selected.reshape(batch_size, token_length).mean(dim=1).cpu().numpy()

    def score(self, utterance: str, wav_path: str, prefixes: Sequence[str], candidates: Sequence[str]):
        if len(prefixes) != len(candidates):
            raise ValueError("one prefix is required for every candidate")
        audio = self._prepare_audio(wav_path)
        grouped: dict[str, list[int]] = {}
        tokenized: dict[str, list[list[int]]] = {}
        for index, candidate in enumerate(candidates):
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            key = prefixes[index]
            grouped.setdefault(key, []).append(index)
            tokenized.setdefault(key, []).append(ids)
        values = np.empty(len(candidates), dtype=np.float64)
        if self.use_kv_cache:
            base_cache, base_last = self._ensure_audio_cache()
            for prefix, indexes in grouped.items():
                values[indexes] = self._score_prefix_group(
                    base_cache, base_last, prefix, tokenized[prefix]
                )
        else:
            by_length: dict[tuple[int, int], list[int]] = {}
            for index, candidate in enumerate(candidates):
                ids = self.tokenizer.encode(candidate, add_special_tokens=False)
                prefix_length = len(
                    self.tokenizer.encode(prefixes[index], add_special_tokens=False)
                )
                by_length.setdefault((prefix_length, len(ids)), []).append(index)
            for _, indexes in by_length.items():
                group_prefixes = [prefixes[index] for index in indexes]
                group_ids = [
                    self.tokenizer.encode(candidates[index], add_special_tokens=False)
                    for index in indexes
                ]
                values[indexes] = self._score_group(audio, group_prefixes, group_ids)
                self.last_stats["full_sequence_forwards"] += 1
        return values

    def score_text_only(self, prefixes: Sequence[str], candidates: Sequence[str]):
        import torch

        if len(prefixes) != len(candidates):
            raise ValueError("one prefix is required for every candidate")
        if not self.use_kv_cache:
            raise ValueError("PMI text scoring requires KV-cache mode")
        if self._text_cache is None:
            ids = torch.tensor([self.text_prompt], dtype=torch.long, device=self.device)
            embeddings = self.model.model.embed_tokens(ids)
            hidden, cache = self._forward_hidden(
                inputs_embeds=embeddings,
                attention_mask=torch.ones_like(ids),
            )
            self._text_cache = cache
            self._text_last_hidden = hidden[:, -1:, :]
        grouped: dict[str, list[int]] = {}
        tokenized: dict[str, list[list[int]]] = {}
        for index, candidate in enumerate(candidates):
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            key = prefixes[index]
            grouped.setdefault(key, []).append(index)
            tokenized.setdefault(key, []).append(ids)
        values = np.empty(len(candidates), dtype=np.float64)
        for key, indexes in grouped.items():
            values[indexes] = self._score_prefix_group(
                self._text_cache, self._text_last_hidden, key, tokenized[key]
            )
        return values


def create_scorer(
    model: str,
    device: str = "cuda:0",
    adapter: str = "",
    kv_cache: bool = True,
    **_: object,
) -> AudioLMScorer:
    if not model:
        raise ValueError("--model is required for integration.step_audio2:create_scorer")
    loaded_model, tokenizer = load_model_and_adapter(model, adapter, device)
    return StepAudio2Scorer(loaded_model, tokenizer, device, use_kv_cache=kv_cache)


def extract_command() -> int:
    parser = argparse.ArgumentParser(description="Export frozen Step-Audio2 audio states for selected layers")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="0,1,2,3,4,8")
    parser.add_argument("--quantization", choices=("none", "8bit", "4bit"), default="none")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--output-root", required=True, help="root directory; each layer uses layer_<n>/enroll and layer_<n>/search")
    parser.add_argument("--enroll-manifest", default="", help="JSONL with clip_id and wav; defaults to --manifest")
    parser.add_argument("--search-manifest", default="", help="JSONL with clip_id and wav; defaults to --manifest")
    parser.add_argument("--manifest", required=True, help="JSONL with clip_id and wav")
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--min-duration", type=float, default=0.32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    layers = tuple(int(value.strip()) for value in args.layers.split(",") if value.strip())
    model, tokenizer = load_model_and_adapter(
        args.model,
        args.adapter,
        args.device,
        quantization=args.quantization,
        dtype=args.dtype,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split_name, manifest_name in (
        ("enroll", args.enroll_manifest or args.manifest),
        ("search", args.search_manifest or args.manifest),
    ):
        with open(manifest_name, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if args.limit > 0:
            rows = rows[: args.limit]
        for row_index, row in enumerate(rows, 1):
            audio = load_audio(row["wav"], args.max_duration)
            if audio.numel() < int(args.min_duration * SAMPLE_RATE):
                continue
            features_by_layer = extract_layers(
                model,
                tokenizer,
                row["wav"],
                args.device,
                layers,
                args.max_duration,
            )
            for layer, features in features_by_layer.items():
                destination = output_root / f"layer_{layer}" / split_name
                destination.mkdir(parents=True, exist_ok=True)
                target = destination / f"{row['clip_id']}.npy"
                if args.overwrite or not target.exists():
                    np.save(target, features.astype(np.float16))
            print(json.dumps({"split": split_name, "done": row_index, "total": len(rows), "clip_id": row["clip_id"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(extract_command())
