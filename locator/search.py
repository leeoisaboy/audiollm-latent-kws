"""Fixed-window mean-template search for the production Stage-1 locator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from common.config import DEFAULT_POOL_TOP, FRAME_S, PRIMARY_K
from locator.templates import LatentTemplateBank, l2_normalize


@dataclass
class LocatedPool:
    utterance: str
    frames: int
    words: list[str]
    scores: np.ndarray
    starts: np.ndarray
    ends: list[int]
    gold: list[tuple[str, int, int]] | None = None

    def to_row(self, include_gold: bool = False) -> dict:
        row = {
            "utt": self.utterance,
            "t": self.frames,
            "words": self.words,
            "score": [float(value) for value in self.scores],
            "ps": [int(value) for value in self.starts],
            "pe": [int(value) for value in self.ends],
            "ntop": len(self.words),
        }
        if include_gold and self.gold is not None:
            row["gold"] = [
                {"word": word, "ts": start, "te": end}
                for word, start, end in self.gold
            ]
        return row


def window_means(features: np.ndarray, window: int = PRIMARY_K) -> np.ndarray | None:
    """Compute all length-``window`` means in one prefix-sum pass."""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("latent features must be [frames, dimensions]")
    if array.shape[0] < window:
        return None
    prefix = np.vstack([np.zeros((1, array.shape[1])), np.cumsum(array, axis=0)])
    return (prefix[window:] - prefix[:-window]) / float(window)


def score_utterance(
    features: np.ndarray,
    bank: LatentTemplateBank,
    window: int = PRIMARY_K,
) -> tuple[np.ndarray, np.ndarray]:
    means = window_means(features, window)
    if means is None:
        return (
            np.full(len(bank.words), -np.inf, dtype=np.float64),
            np.zeros(len(bank.words), dtype=np.int64),
        )
    response = l2_normalize(means) @ bank.vectors.T
    return response.max(axis=0), response.argmax(axis=0)


def fixed_span(start: int, window: int, frame_count: int) -> tuple[int, int]:
    begin = int(np.clip(start, 0, max(0, frame_count - 1)))
    return begin, min(begin + window, frame_count)


def build_pool(
    utterance: str,
    features: np.ndarray,
    bank: LatentTemplateBank,
    pool_top: int = DEFAULT_POOL_TOP,
    window: int = PRIMARY_K,
    gold: Iterable[Mapping] | None = None,
) -> LocatedPool:
    features = np.asarray(features, dtype=np.float64)
    scores, starts = score_utterance(features, bank, window=window)
    order = np.argsort(-scores, kind="stable")[:pool_top]
    ends = [
        fixed_span(int(starts[index]), window, len(features))[1] for index in order
    ]
    gold_rows = None
    if gold is not None:
        gold_rows = []
        for item in gold:
            start_frame = int(np.floor(float(item["kw_t0"]) / FRAME_S))
            end_frame = int(np.ceil(float(item["kw_t1"]) / FRAME_S))
            gold_rows.append((str(item["word"]), start_frame, end_frame))
    return LocatedPool(
        utterance=utterance,
        frames=len(features),
        words=[bank.words[index] for index in order],
        scores=scores[order],
        starts=starts[order],
        ends=ends,
        gold=gold_rows,
    )
