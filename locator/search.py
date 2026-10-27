from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from common.config import (
    DEFAULT_NMS,
    DEFAULT_NMS_RADIUS,
    DEFAULT_POOL_TOP,
    FRAME_S,
    GROW_DELTA,
    PRIMARY_K,
)
from locator.templates import LatentTemplateBank, l2_normalize


@dataclass
class LocatedPool:
    utterance: str
    frames: int
    words: list[str]
    scores: np.ndarray
    starts: np.ndarray
    gold: list[tuple[str, int, int]] | None = None
    alt_starts: list[list[int]] | None = None
    ends: list[int] | None = None

    def to_row(self, include_gold: bool = False) -> dict:
        row = {
            "utt": self.utterance,
            "t": self.frames,
            "words": self.words,
            "score": [float(value) for value in self.scores],
            "ps": [int(value) for value in self.starts],
            "ntop": len(self.words),
        }
        if self.alt_starts is not None:
            row["ps_alt"] = [[int(value) for value in values] for values in self.alt_starts]
        if self.ends is not None:
            row["pe"] = [int(value) for value in self.ends]
        if include_gold and self.gold is not None:
            row["gold"] = [{"word": word, "ts": start, "te": end} for word, start, end in self.gold]
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


def _response_matrix(
    features: np.ndarray,
    bank: LatentTemplateBank,
    window: int,
    adaptive_window: bool,
    multi_scale: int,
    use_prototypes: bool,
) -> np.ndarray:
    response = np.full((len(features), len(bank.words)), -np.inf, dtype=np.float64)
    words_by_width: dict[int, list[int]] = {}
    for word_index, word in enumerate(bank.words):
        base = bank.window_for(word) if adaptive_window else int(window)
        radius = max(0, int(multi_scale)) if adaptive_window else 0
        for candidate_width in range(max(2, base - radius), base + radius + 1):
            words_by_width.setdefault(candidate_width, []).append(word_index)
    cache: dict[int, np.ndarray | None] = {}
    for candidate_width, positions in words_by_width.items():
        if candidate_width not in cache:
            means = window_means(features, candidate_width)
            cache[candidate_width] = None if means is None else l2_normalize(means)
        normalized = cache[candidate_width]
        if normalized is None:
            continue
        if use_prototypes and bank.prototypes is not None and bank.prototype_offsets is not None:
            all_scores = np.maximum.reduceat(
                normalized @ bank.prototypes.T, bank.prototype_offsets[:-1], axis=1
            )
        else:
            all_scores = normalized @ bank.vectors.T
        index = np.asarray(positions, dtype=np.int64)
        observed = all_scores.shape[0]
        response[:observed, index] = np.maximum(
            response[:observed, index], all_scores[:, index]
        )
    return response


def score_utterance(
    features: np.ndarray,
    bank: LatentTemplateBank,
    window: int = PRIMARY_K,
    adaptive_window: bool = False,
    multi_scale: int = 0,
    use_prototypes: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(features, dtype=np.float64)
    response = _response_matrix(
        array, bank, window, adaptive_window, multi_scale, use_prototypes
    )
    starts = response.argmax(axis=0)
    scores = response.max(axis=0)
    primary = max(2, min(window, len(array)))
    means = window_means(array, primary) if len(array) >= primary else None
    normalized = np.zeros((0, array.shape[1])) if means is None else l2_normalize(means)
    return scores, starts, normalized, response


def frame_cosine(features: np.ndarray, bank: LatentTemplateBank) -> np.ndarray:
    return l2_normalize(np.asarray(features, dtype=np.float64)) @ bank.vectors.T


def fixed_span(start: int, window: int, frame_count: int) -> tuple[int, int]:
    begin = int(np.clip(start, 0, max(0, frame_count - 1)))
    return begin, min(begin + window, frame_count)


def grow_span(
    column: np.ndarray,
    start: int,
    window: int,
    frame_count: int,
    delta: float = GROW_DELTA,
    width: int | None = None,
) -> tuple[int, int]:
    """Extend the peak window while per-frame cosine stays near its maximum."""
    begin, end = fixed_span(start, width or window, frame_count)
    peak = float(column[begin:end].max())
    threshold = peak - delta
    while begin > 0 and float(column[begin - 1]) >= threshold:
        begin -= 1
    while end < frame_count and float(column[end]) >= threshold:
        end += 1
    return begin, end


def nms_starts(
    column: np.ndarray,
    count: int = DEFAULT_NMS,
    radius: int = DEFAULT_NMS_RADIUS,
) -> list[int]:
    """Return up to ``count`` separated local maxima, highest score first."""
    count = max(1, int(count))
    finite = np.isfinite(column)
    if not finite.any():
        return [0]
    candidates = np.flatnonzero(finite)
    ordered = candidates[np.argsort(-column[candidates], kind="stable")]
    chosen: list[int] = []
    for index in ordered:
        if any(abs(int(index) - previous) <= radius for previous in chosen):
            continue
        chosen.append(int(index))
        if len(chosen) == count:
            break
    return chosen or [int(ordered[0])]


def _grown_score(
    features: np.ndarray,
    bank: LatentTemplateBank,
    word: str,
    begin: int,
    end: int,
) -> float:
    span = np.asarray(features[begin:end], dtype=np.float64)
    if not len(span):
        return -np.inf
    mean = l2_normalize(span.mean(axis=0))
    return float(np.max(mean @ bank.prototypes_for(word).T))


def build_pool(
    utterance: str,
    features: np.ndarray,
    bank: LatentTemplateBank,
    pool_top: int = DEFAULT_POOL_TOP,
    window: int = PRIMARY_K,
    gold: Iterable[Mapping] | None = None,
    adaptive_window: bool = False,
    multi_scale: int = 0,
    use_prototypes: bool = True,
    use_grow_span: bool = False,
    grow_delta: float = GROW_DELTA,
    nms_candidates: int = DEFAULT_NMS,
    nms_radius: int = DEFAULT_NMS_RADIUS,
) -> LocatedPool:
    features = np.asarray(features, dtype=np.float64)
    scores, starts, _, response = score_utterance(
        features,
        bank,
        window=window,
        adaptive_window=adaptive_window,
        multi_scale=multi_scale,
        use_prototypes=use_prototypes,
    )
    frame_columns = None
    if use_grow_span:
        # Compute all per-frame word responses once.  The previous implementation
        # repeated this projection inside the candidate loop, which made
        # grow-span quadratic in the vocabulary size for no algorithmic reason.
        normalized_frames = l2_normalize(features)
        if bank.prototypes is not None and bank.prototype_offsets is not None:
            frame_scores = normalized_frames @ bank.prototypes.T
            frame_columns = np.maximum.reduceat(
                frame_scores, bank.prototype_offsets[:-1], axis=1
            )
        else:
            frame_columns = normalized_frames @ bank.vectors.T
    all_starts: list[list[int]] = []
    ends: list[int] = []
    for word_index, word in enumerate(bank.words):
        width = bank.window_for(word) if adaptive_window else int(window)
        all_starts.append(nms_starts(response[:, word_index], nms_candidates, nms_radius))
        begin, end = fixed_span(int(starts[word_index]), width, len(features))
        if use_grow_span:
            grow_begin, grow_end = grow_span(
                frame_columns[:, word_index],
                int(starts[word_index]),
                width,
                len(features),
                delta=grow_delta,
            )
            grown = _grown_score(features, bank, word, grow_begin, grow_end)
            if grown > scores[word_index]:
                scores[word_index] = grown
                begin, end = grow_begin, grow_end
        ends.append(end)
    order = np.argsort(-scores, kind="stable")[:pool_top]
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
        gold=gold_rows,
        alt_starts=[all_starts[index] for index in order],
        ends=[ends[index] for index in order],
    )
