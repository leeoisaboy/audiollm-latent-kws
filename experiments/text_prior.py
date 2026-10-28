"""Research-only text-only scorer used by the rejected PMI ablation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from integration.step_audio2 import StepAudio2Scorer, load_model_and_adapter

TEXT_INSTRUCTION = "\u8bf7\u6839\u636e\u5df2\u7ed9\u8bed\u5883\u8865\u5168\u4e0b\u4e00\u4e2a\u8bcd\u3002"


def text_prompt_ids(tokenizer, text: str = TEXT_INSTRUCTION) -> list[int]:
    return tokenizer(
        "<|BOT|>human\n" + text + "<|EOT|><|BOT|>assistant\n",
        add_special_tokens=False,
    )["input_ids"]


class TextPriorScorer(StepAudio2Scorer):
    """StepAudio2Scorer with an extra text-only prefix cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_prompt = text_prompt_ids(self.tokenizer)
        self._text_cache = None
        self._text_last_hidden = None

    def score_text_only(
        self, prefixes: Sequence[str], candidates: Sequence[str]
    ) -> np.ndarray:
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
) -> TextPriorScorer:
    loaded_model, tokenizer = load_model_and_adapter(model, adapter, device)
    return TextPriorScorer(
        loaded_model,
        tokenizer,
        device,
        use_kv_cache=kv_cache,
    )
