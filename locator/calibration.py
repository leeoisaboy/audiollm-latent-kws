"""Mean-centering calibration for frozen latent frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_CALIBRATION_FRAMES = 8192


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
    """Remove the global latent mean fitted only on enrollment features."""

    mean: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64) - self.mean

    def to_metadata(self) -> dict[str, object]:
        return {"mode": "center"}


def fit_latent_calibrator(
    arrays: Iterable[np.ndarray],
    mode: str = "center",
    max_frames: int = DEFAULT_CALIBRATION_FRAMES,
) -> LatentCalibrator:
    if mode != "center":
        raise ValueError("the production locator only supports mean centering")
    frames = _sample_frames(arrays, max_frames).astype(np.float64)
    return LatentCalibrator(mean=frames.mean(axis=0))


def calibration_metadata(
    calibrator: LatentCalibrator | None,
) -> dict[str, object] | None:
    return None if calibrator is None else calibrator.to_metadata()
