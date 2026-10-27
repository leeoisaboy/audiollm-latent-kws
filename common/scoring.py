from __future__ import annotations

import numpy as np

from .config import LAMBDA_LM, Z_EPS


def zpool(values) -> np.ndarray:
    """Standardize finite values inside one candidate pool."""
    array = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(array)
    if not mask.any():
        return np.zeros_like(array)
    deviation = float(array[mask].std())
    if deviation < Z_EPS:
        return np.zeros_like(array)
    centered = (array - float(array[mask].mean())) / deviation
    return np.where(mask, centered, 0.0)


def fuse(acoustic, language_model, lambda_lm: float = LAMBDA_LM) -> np.ndarray:
    """Return ``z_pool(acoustic) + lambda_lm * z_pool(language_model)``."""
    return zpool(acoustic) + lambda_lm * zpool(language_model)


def rank_of(scores, index: int) -> float:
    """Competition-free descending rank: one plus items above plus half ties."""
    array = np.asarray(scores)
    above = int(np.count_nonzero(array > array[index]))
    equal = int(np.count_nonzero(array == array[index])) - 1
    return 1.0 + above + 0.5 * equal
