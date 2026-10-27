"""Stage 1: acoustic latent templates, sliding-window cosine search, and pools."""

from .calibration import LatentCalibrator, fit_latent_calibrator
from .layers import LayerFusionSpec, LayerViewStore, parse_fusion_spec, parse_fusion_specs
from .search import build_pool, frame_cosine, grow_span, nms_starts, score_utterance, window_means
from .templates import DirectoryLatentStore, LatentStore, build_templates

__all__ = [
    "DirectoryLatentStore",
    "LayerFusionSpec",
    "LayerViewStore",
    "LatentCalibrator",
    "LatentStore",
    "build_pool",
    "build_templates",
    "frame_cosine",
    "fit_latent_calibrator",
    "grow_span",
    "nms_starts",
    "parse_fusion_spec",
    "parse_fusion_specs",
    "score_utterance",
    "window_means",
]
