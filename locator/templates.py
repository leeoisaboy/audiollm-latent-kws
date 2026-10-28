"""Mean templates for the production Stage-1 locator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.config import FRAME_S, LATENT_WIDTH
from locator.calibration import LatentCalibrator


def l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


@dataclass
class LatentTemplateBank:
    words: list[str]
    vectors: np.ndarray
    durations: dict[str, float]
    takes: dict[str, int]

    def __len__(self) -> int:
        return len(self.words)

    @property
    def index(self) -> dict[str, int]:
        return {word: position for position, word in enumerate(self.words)}


class LatentStore:
    def get(self, clip_id: str) -> np.ndarray:
        raise NotImplementedError


class DirectoryLatentStore(LatentStore):
    def __init__(self, root: str | Path, expected_width: int = LATENT_WIDTH):
        self.root = Path(root)
        self.expected_width = expected_width

    def get(self, clip_id: str) -> np.ndarray:
        array = np.load(self.root / f"{clip_id}.npy")
        array = np.asarray(array, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.expected_width:
            raise ValueError(
                f"{clip_id}: expected [frames, {self.expected_width}], got {array.shape}"
            )
        return array


def span_frames(
    start_time: float, end_time: float, frame_s: float, length: int
) -> tuple[int, int]:
    start = int(np.floor(float(start_time) / frame_s))
    end = int(np.ceil(float(end_time) / frame_s))
    return max(0, start), min(int(length), max(end, start + 1))


def build_templates(
    rows: Sequence[Mapping[str, Any]],
    latent_store: LatentStore,
    frame_s: float = FRAME_S,
    only: set[str] | None = None,
    calibrator: LatentCalibrator | None = None,
) -> LatentTemplateBank:
    """Average enrollments twice: take means first, then a word centroid."""
    take_means: dict[str, list[np.ndarray]] = {}
    durations: dict[str, list[float]] = {}
    for row in rows:
        word = str(row["word"])
        if only is not None and word not in only:
            continue
        clip_id = str(row["clip_id"])
        try:
            features = latent_store.get(clip_id)
        except (FileNotFoundError, OSError):
            continue
        if calibrator is not None:
            features = calibrator.transform(features)
        start, end = span_frames(row["kw_t0"], row["kw_t1"], frame_s, len(features))
        if end <= start:
            continue
        take_means.setdefault(word, []).append(
            l2_normalize(features[start:end].mean(axis=0))
        )
        durations.setdefault(word, []).append(float(row["kw_t1"]) - float(row["kw_t0"]))

    words = sorted(take_means)
    if not words:
        raise ValueError("no usable enrollment features")
    vectors = l2_normalize(
        np.stack([np.mean(np.stack(take_means[word]), axis=0) for word in words])
    )
    return LatentTemplateBank(
        words=words,
        vectors=vectors,
        durations={word: float(np.mean(values)) for word, values in durations.items()},
        takes={word: len(values) for word, values in take_means.items()},
    )
