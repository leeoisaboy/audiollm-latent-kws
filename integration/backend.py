"""Minimal 3-method Audio-LLM backend interface.

Every frozen Audio-LLM backend implements just three core methods:
  1. ``extract_shallow(wav_path, layer=None)`` -> ``[T, D]`` float32 shallow latents
  2. ``prefill_prefix_kv(wav_path, prefix_text)`` -> reusable ``PrefixKVState``
  3. ``score_suffixes(prefix_state, suffixes)`` -> ``[N]`` float64 mean log-probs

Optional helper methods (``transcribe_prefix`` for reader-free acoustic backoff
and ``score_full`` for KV-cache parity verification) are also specified with
clear defaults.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class PrefixKVState:
    """Opaque cached state returned by ``prefill_prefix_kv``."""

    wav_path: str
    prefix_text: str
    cache: Any
    last_hidden: Any
    extra: dict[str, Any] = field(default_factory=dict)


class AudioLLMBackend(ABC):
    """Minimal frozen Audio-LLM backend contract."""

    name: str = "unknown"
    frame_s: float = 0.08
    default_layer: int = 2
    hidden_dim: int = 3584

    @abstractmethod
    def extract_shallow(
        self, wav_path: str, layer: int | None = None
    ) -> np.ndarray:
        """Extract shallow audio hidden states ``[T, D]`` (float32)."""

    @abstractmethod
    def prefill_prefix_kv(
        self, wav_path: str, prefix_text: str
    ) -> PrefixKVState:
        """Encode ``wav_path`` and prefill ``prefix_text`` into a reusable KV state."""

    @abstractmethod
    def score_suffixes(
        self, prefix_state: PrefixKVState, suffixes: Sequence[str]
    ) -> np.ndarray:
        """Score candidate suffix strings from ``prefix_state`` (mean token log-prob)."""

    def transcribe_prefix(self, wav_path: str, cut_time_s: float) -> str:
        """Decode left-context prefix text from ``wav[:cut_time_s]`` using the same model.

        Returns empty string if ``cut_time_s <= 0`` or if the backend does not
        implement self-prefix decoding.
        """
        return ""

    def score_full(
        self,
        wav_path: str,
        prefixes: Sequence[str],
        suffixes: Sequence[str],
    ) -> np.ndarray:
        """Full-sequence teacher forcing without KV cache (for parity/speedup audits)."""
        raise NotImplementedError(f"{self.name} does not implement score_full")


class BackendScorerAdapter:
    """Adapt any ``AudioLLMBackend`` to the legacy ``AudioLMScorer.score`` contract."""

    def __init__(self, backend: AudioLLMBackend):
        self.backend = backend

    def score(
        self,
        utterance: str,
        wav_path: str,
        prefixes: Sequence[str],
        candidates: Sequence[str],
    ) -> np.ndarray:
        if len(prefixes) != len(candidates):
            raise ValueError("prefixes and candidates must have the same length")
        if not candidates:
            return np.empty(0, dtype=np.float64)
        grouped: dict[str, list[int]] = {}
        for idx, prefix in enumerate(prefixes):
            grouped.setdefault(prefix, []).append(idx)
        scores = np.empty(len(candidates), dtype=np.float64)
        for prefix, indices in grouped.items():
            state = self.backend.prefill_prefix_kv(wav_path, prefix)
            group_suffixes = [candidates[i] for i in indices]
            scores[indices] = self.backend.score_suffixes(state, group_suffixes)
        return scores
