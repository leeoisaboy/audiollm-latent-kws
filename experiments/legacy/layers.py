from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from locator.templates import LatentStore, l2_normalize


@dataclass(frozen=True)
class LayerFusionSpec:
    layers: tuple[int, ...]
    weights: tuple[float, ...]
    concat: bool = False

    @property
    def name(self) -> str:
        if self.concat:
            return "concat:" + ",".join(str(layer) for layer in self.layers)
        return ",".join(f"{layer}:{weight:g}" for layer, weight in zip(self.layers, self.weights))


def parse_fusion_spec(specification: str) -> LayerFusionSpec:
    """Parse ``0:0.7,2:0.3`` (weighted) or ``concat:0,2`` (concatenation)."""
    text = specification.strip()
    if not text:
        raise ValueError("empty layer fusion specification")
    if text.startswith("concat:"):
        layers = tuple(int(value) for value in text.split(":", 1)[1].split(",") if value.strip())
        if len(layers) < 2:
            raise ValueError("concat fusion needs at least two layers")
        return LayerFusionSpec(layers=layers, weights=(1.0,) * len(layers), concat=True)
    pairs = []
    for item in text.split(","):
        if not item.strip():
            continue
        layer_text, separator, weight_text = item.partition(":")
        if not separator:
            raise ValueError(f"fusion entry must be layer:weight, got {item!r}")
        pairs.append((int(layer_text), float(weight_text)))
    if len(pairs) < 1:
        raise ValueError("fusion needs at least one layer")
    total = sum(abs(weight) for _, weight in pairs)
    if total <= 0:
        raise ValueError("fusion weights must not all be zero")
    return LayerFusionSpec(
        layers=tuple(layer for layer, _ in pairs),
        weights=tuple(weight / total for _, weight in pairs),
    )


def parse_fusion_specs(specifications: str | Sequence[str]) -> list[LayerFusionSpec]:
    values = specifications.split(";") if isinstance(specifications, str) else list(specifications)
    return [parse_fusion_spec(value) for value in values if value.strip()]


class LayerViewStore(LatentStore):
    """Read one clip from several layers and fuse them into a single view."""

    def __init__(
        self,
        features_root: str | Path,
        split: str,
        specification: LayerFusionSpec,
        expected_layer_width: int | None = None,
        layer_pattern: str = "layer_{layer}",
    ):
        self.features_root = Path(features_root)
        self.split = split
        self.specification = specification
        self.expected_layer_width = expected_layer_width
        self.layer_pattern = layer_pattern

    @property
    def _paths(self) -> list[Path]:
        return [
            self.features_root / self.layer_pattern.format(layer=layer) / self.split
            for layer in self.specification.layers
        ]

    def get(self, clip_id: str) -> np.ndarray:
        arrays = []
        for root in self._paths:
            array = np.asarray(np.load(root / f"{clip_id}.npy"), dtype=np.float64)
            if array.ndim != 2:
                raise ValueError(f"{clip_id}: expected a 2-D layer view, got {array.shape}")
            if self.expected_layer_width is not None and array.shape[1] != self.expected_layer_width:
                raise ValueError(
                    f"{clip_id}: expected width {self.expected_layer_width}, got {array.shape[1]}"
                )
            arrays.append(l2_normalize(array))
        if len({array.shape[0] for array in arrays}) != 1:
            raise ValueError(f"{clip_id}: fused layers disagree on frame count")
        if len({array.shape[1] for array in arrays}) != 1:
            raise ValueError(f"{clip_id}: fused layers disagree on width")
        if self.specification.concat:
            return np.concatenate(arrays, axis=1)
        return sum(
            weight * array
            for weight, array in zip(self.specification.weights, arrays)
        )

