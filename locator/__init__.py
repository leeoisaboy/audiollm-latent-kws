"""Stage 1: acoustic latent templates, sliding-window cosine search, and pools."""

from .calibration import LatentCalibrator, fit_latent_calibrator
from .search import build_pool, score_utterance, window_means
from .templates import DirectoryLatentStore, LatentStore, build_templates

__all__ = [
    "DirectoryLatentStore",
    "LatentCalibrator",
    "LatentStore",
    "build_pool",
    "build_templates",
    "fit_latent_calibrator",
    "score_utterance",
    "window_means",
]
