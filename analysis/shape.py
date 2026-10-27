from __future__ import annotations

from typing import Sequence

import numpy as np

from locator.templates import l2_normalize


def resample(values: np.ndarray, output_length: int) -> np.ndarray:
    """Linearly resample the leading axis without changing the leading dimension semantics."""
    array = np.asarray(values, dtype=np.float64)
    if output_length <= 0:
        raise ValueError("output_length must be positive")
    if len(array) == output_length:
        return array
    if len(array) == 0:
        raise ValueError("cannot resample an empty sequence")
    positions = np.linspace(0, len(array) - 1, output_length)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(array) - 1)
    weight = (positions - lower).reshape((-1,) + (1,) * (array.ndim - 1))
    return array[lower] * (1.0 - weight) + array[upper] * weight


def first_difference(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if len(array) < 2:
        raise ValueError("first difference needs at least two frames")
    return np.diff(array, axis=0)


def shape_template(enrollments: Sequence[np.ndarray], descriptor_length: int) -> np.ndarray:
    vectors = []
    for enrollment in enrollments:
        if len(enrollment) < 2:
            continue
        difference = first_difference(enrollment)
        vectors.append(l2_normalize(resample(difference, descriptor_length).reshape(-1)))
    if not vectors:
        raise ValueError("no enrollment long enough for a shape template")
    return l2_normalize(np.mean(np.stack(vectors), axis=0))


def shape_similarity(features: np.ndarray, template: np.ndarray, descriptor_length: int) -> float:
    difference = first_difference(features)
    query = l2_normalize(resample(difference, descriptor_length).reshape(-1))
    return float(query @ template)


def pairwise_cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = l2_normalize(np.asarray(left, dtype=np.float64))
    right = l2_normalize(np.asarray(right, dtype=np.float64))
    return 1.0 - left @ right.T


def dtw_distance(query: np.ndarray, template: np.ndarray) -> float:
    """Length-normalized DTW cost with cosine local distance."""
    left = np.asarray(query, dtype=np.float64)
    right = np.asarray(template, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or not len(left) or not len(right):
        raise ValueError("DTW needs two non-empty 2-D sequences")
    cost = pairwise_cosine_distance(left, right)
    table = np.full((len(left) + 1, len(right) + 1), np.inf, dtype=np.float64)
    lengths = np.zeros_like(table)
    table[0, 0] = 0.0
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            previous = np.argmin(
                [table[i - 1, j - 1], table[i - 1, j], table[i, j - 1]]
            )
            if previous == 0:
                table[i, j] = table[i - 1, j - 1]
                lengths[i, j] = lengths[i - 1, j - 1]
            elif previous == 1:
                table[i, j] = table[i - 1, j]
                lengths[i, j] = lengths[i - 1, j]
            else:
                table[i, j] = table[i, j - 1]
                lengths[i, j] = lengths[i, j - 1]
            table[i, j] += cost[i - 1, j - 1]
            lengths[i, j] += 1
    return float(table[-1, -1] / max(lengths[-1, -1], 1.0))


def subsequence_dtw_distance(query: np.ndarray, sequence: np.ndarray) -> float:
    """Match ``query`` anywhere inside ``sequence`` with free endpoint boundaries."""
    left = np.asarray(query, dtype=np.float64)
    right = np.asarray(sequence, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or not len(left) or not len(right):
        raise ValueError("subsequence DTW needs two non-empty 2-D sequences")
    cost = pairwise_cosine_distance(left, right)
    table = np.full((len(left) + 1, len(right) + 1), np.inf, dtype=np.float64)
    lengths = np.zeros_like(table)
    table[0, :] = 0.0
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            previous = np.argmin(
                [table[i - 1, j - 1], table[i - 1, j], table[i, j - 1]]
            )
            if previous == 0:
                table[i, j] = table[i - 1, j - 1]
                lengths[i, j] = lengths[i - 1, j - 1]
            elif previous == 1:
                table[i, j] = table[i - 1, j]
                lengths[i, j] = lengths[i - 1, j]
            else:
                table[i, j] = table[i, j - 1]
                lengths[i, j] = lengths[i, j - 1]
            table[i, j] += cost[i - 1, j - 1]
            lengths[i, j] += 1
    return float(np.min(table[-1, 1:] / np.maximum(lengths[-1, 1:], 1.0)))


def shift_invariant_similarity(query: np.ndarray, template: np.ndarray, max_shift: int = 2) -> float:
    """Maximum cosine over small relative frame shifts, immune to boundary jitter."""
    left = l2_normalize(np.asarray(query, dtype=np.float64))
    right = l2_normalize(np.asarray(template, dtype=np.float64))
    best = -np.inf
    for shift in range(-int(max_shift), int(max_shift) + 1):
        if shift >= 0:
            left_part, right_part = left[shift:], right[: len(right) - shift]
        else:
            left_part, right_part = left[: len(left) + shift], right[-shift:]
        overlap = min(len(left_part), len(right_part))
        if overlap < 2:
            continue
        value = float(np.mean(np.sum(left_part[:overlap] * right_part[:overlap], axis=1)))
        best = max(best, value)
    return float(best) if np.isfinite(best) else -1.0


def sequence_entropy(values: np.ndarray) -> float:
    """Normalized entropy of a candidate-pool probability distribution."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) < 2:
        return 0.0
    probabilities = np.exp(finite - finite.max())
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return entropy / np.log(len(finite))


def shape_fusion_delta(
    acoustic: Sequence[Sequence[float]],
    language_model: Sequence[Sequence[float]],
    shape: Sequence[Sequence[float]],
    gold: Sequence[Sequence[str]],
    candidates: Sequence[Sequence[str]],
    beta: float = 1.0,
    gating: bool = False,
    entropy_threshold: float = 0.65,
    lambda_lm: float = 1.5,
) -> dict[str, float]:
    """Compare LM-only and LM plus shape top-1 over finite in-pool occurrences."""
    from common.scoring import fuse, zpool

    baseline_correct = []
    fused_correct = []
    gated_correct = []
    finite_shape = []
    entropies = []
    for row_acoustic, row_lm, row_shape, row_gold, row_words in zip(
        acoustic, language_model, shape, gold, candidates
    ):
        baseline = fuse(row_acoustic, row_lm, lambda_lm=lambda_lm)
        enhanced = baseline + beta * zpool(row_shape)
        entropy = sequence_entropy(row_lm)
        gate = gating and entropy >= entropy_threshold
        gated = baseline + beta * zpool(row_shape) if gate else baseline
        for target in row_gold:
            if target not in row_words:
                continue
            index = row_words.index(target)
            baseline_correct.append(float(np.argmax(baseline) == index))
            fused_correct.append(float(np.argmax(enhanced) == index))
            gated_correct.append(float(np.argmax(gated) == index))
            finite_shape.append(bool(np.isfinite(row_shape[index])))
            entropies.append(entropy)
    count = len(baseline_correct)
    baseline = float(np.mean(baseline_correct)) if count else 0.0
    enhanced = float(np.mean(fused_correct)) if count else 0.0
    gated = float(np.mean(gated_correct)) if count else 0.0
    return {
        "n": count,
        "lm_fused_top1": baseline,
        "shape_fused_top1": enhanced,
        "delta": enhanced - baseline,
        "gated_top1": gated,
        "gated_delta": gated - baseline,
        "mean_lm_entropy": float(np.mean(entropies)) if count else 0.0,
        "gate_rate": float(np.mean([value >= entropy_threshold for value in entropies])) if count else 0.0,
        "finite_shape_ratio": float(np.mean(finite_shape)) if count else 0.0,
    }
