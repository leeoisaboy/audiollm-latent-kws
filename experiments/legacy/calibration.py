from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

LEGACY_CALIBRATION_FRAMES = 8192
LEGACY_WHITENING_RANK = 32


def _sample_frames(arrays: Iterable[np.ndarray], max_frames: int) -> np.ndarray:
    """Collect at most ``max_frames`` rows with deterministic stride sampling."""
    frames = [np.asarray(array, dtype=np.float32) for array in arrays]
    frames = [array for array in frames if array.ndim == 2 and len(array)]
    if not frames:
        raise ValueError("calibration needs at least one non-empty feature array")
    total = sum(len(array) for array in frames)
    if total <= max_frames:
        return np.concatenate(frames, axis=0)
    output = np.empty((max_frames, frames[0].shape[1]), dtype=np.float32)
    filled = 0
    for array in frames:
        take = max(1, int(round(max_frames * len(array) / total)))
        take = min(take, len(array), max_frames - filled)
        if take <= 0:
            break
        positions = np.linspace(0, len(array) - 1, take).round().astype(int)
        output[filled : filled + take] = array[positions]
        filled += take
        if filled == max_frames:
            break
    return output[:filled]


@dataclass
class LatentCalibrator:
    """Training-free anisotropy calibration for frozen latent frames.

    ``center`` removes the global mean, ``abtt`` additionally removes the top
    principal directions (All-but-the-Top), and ``whiten`` performs low-rank PCA
    whitening.  The transform is fitted only on enrollment/background features,
    never on evaluation labels or per-query decisions.
    """

    mean: np.ndarray
    components: np.ndarray
    scales: np.ndarray
    mode: str = "center"

    def transform(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float64)
        centered = array - self.mean
        if not len(self.components) or self.mode == "center":
            return centered
        projection = centered @ self.components.T
        if self.mode == "abtt":
            return centered - projection @ self.components
        if self.mode == "whiten":
            return (projection * self.scales) @ self.components
        raise ValueError(f"unknown calibrator mode: {self.mode}")

    def to_metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "rank": int(len(self.components)),
            "retained_variance": float(np.sum(self.scales ** -2)) if self.mode == "whiten" else None,
        }


def fit_latent_calibrator(
    arrays: Iterable[np.ndarray],
    mode: str = "center",
    rank: int = LEGACY_WHITENING_RANK,
    max_frames: int = LEGACY_CALIBRATION_FRAMES,
    variance_floor: float = 1e-6,
) -> LatentCalibrator:
    if mode not in {"center", "abtt", "whiten"}:
        raise ValueError("mode must be center, abtt, or whiten")
    frames = _sample_frames(arrays, max_frames).astype(np.float64)
    mean = frames.mean(axis=0)
    centered = frames - mean
    if mode == "center":
        empty = np.empty((0, frames.shape[1]), dtype=np.float64)
        return LatentCalibrator(mean=mean, components=empty, scales=np.empty(0), mode=mode)
    rank = max(1, min(int(rank), min(centered.shape) - 1))
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:rank]
    variance = (singular[:rank] ** 2) / max(len(centered) - 1, 1)
    if mode == "abtt":
        scales = np.ones(rank, dtype=np.float64)
    else:
        scales = 1.0 / np.sqrt(np.maximum(variance, variance_floor))
        scales = np.minimum(scales, 1.0 / np.sqrt(variance_floor))
    return LatentCalibrator(mean=mean, components=components, scales=scales, mode=mode)


def calibration_metadata(calibrator: LatentCalibrator | None) -> dict[str, object] | None:
    return None if calibrator is None else calibrator.to_metadata()
