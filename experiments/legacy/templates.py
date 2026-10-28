from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.config import FRAME_S, LATENT_WIDTH
from experiments.legacy.calibration import LatentCalibrator


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
    prototypes: np.ndarray | None = None
    prototype_offsets: np.ndarray | None = None
    windows: dict[str, int] | None = None

    def __len__(self) -> int:
        return len(self.words)

    @property
    def index(self) -> dict[str, int]:
        return {word: position for position, word in enumerate(self.words)}

    def prototypes_for(self, word: str) -> np.ndarray:
        if self.prototypes is None or self.prototype_offsets is None:
            return self.vectors[[self.index[word]]]
        position = self.index[word]
        begin = int(self.prototype_offsets[position])
        end = int(self.prototype_offsets[position + 1])
        return self.prototypes[begin:end]

    def window_for(self, word: str, frame_s: float = FRAME_S) -> int:
        if self.windows and word in self.windows:
            return self.windows[word]
        duration = self.durations.get(word, 5.0 * frame_s)
        return max(2, int(round(duration / frame_s)))


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


def span_frames(start_time: float, end_time: float, frame_s: float, length: int) -> tuple[int, int]:
    start = int(np.floor(float(start_time) / frame_s))
    end = int(np.ceil(float(end_time) / frame_s))
    return max(0, start), min(int(length), max(end, start + 1))


def _kmeans_prototypes(points: np.ndarray, count: int, iterations: int = 32) -> np.ndarray:
    """Deterministic farthest-first k-means over one word's take means."""
    array = np.asarray(points, dtype=np.float64)
    count = max(1, min(int(count), len(array)))
    if count == 1:
        return l2_normalize(array.mean(axis=0, keepdims=True))
    centers = [array[0]]
    distances = np.linalg.norm(array - centers[0], axis=1)
    while len(centers) < count:
        index = int(np.argmax(distances))
        centers.append(array[index])
        distances = np.minimum(distances, np.linalg.norm(array - array[index], axis=1))
    centers_array = np.stack(centers)
    for _ in range(iterations):
        similarities = l2_normalize(array) @ l2_normalize(centers_array).T
        assignment = np.argmax(similarities, axis=1)
        updated = []
        for index in range(len(centers_array)):
            members = array[assignment == index]
            updated.append(members.mean(axis=0) if len(members) else centers_array[index])
        next_centers = np.stack(updated)
        if np.allclose(next_centers, centers_array, atol=1e-7):
            centers_array = next_centers
            break
        centers_array = next_centers
    return l2_normalize(centers_array)


def build_templates(
    rows: Sequence[Mapping[str, Any]],
    latent_store: LatentStore,
    frame_s: float = FRAME_S,
    only: set[str] | None = None,
    prototypes: int = 1,
    calibrator: LatentCalibrator | None = None,
) -> LatentTemplateBank:
    """Average enrollments twice: take means first, then a global centroid."""
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
        take_means.setdefault(word, []).append(l2_normalize(features[start:end].mean(axis=0)))
        durations.setdefault(word, []).append(float(row["kw_t1"]) - float(row["kw_t0"]))

    words = sorted(take_means)
    if not words:
        raise ValueError("no usable enrollment features")
    vectors = l2_normalize(np.stack([np.mean(np.stack(take_means[word]), axis=0) for word in words]))
    prototype_rows = []
    offsets = [0]
    for word in words:
        word_prototypes = _kmeans_prototypes(np.stack(take_means[word]), prototypes)
        prototype_rows.extend(word_prototypes)
        offsets.append(len(prototype_rows))
    return LatentTemplateBank(
        words=words,
        vectors=vectors,
        durations={word: float(np.mean(values)) for word, values in durations.items()},
        takes={word: len(values) for word, values in take_means.items()},
        prototypes=l2_normalize(np.stack(prototype_rows)),
        prototype_offsets=np.asarray(offsets, dtype=np.int64),
        windows={
            word: max(2, int(round(float(np.mean(durations[word])) / frame_s)))
            for word in words
        },
    )
