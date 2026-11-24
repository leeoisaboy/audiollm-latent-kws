"""Language-agnostic 5-step pure-function core API."""

from core.pipeline import (
    AutoTemplateBank,
    CandidatePool,
    DetectionResult,
    PrefixCut,
    RerankedPool,
    compute_auto_window,
    enroll,
    kv_rerank,
    locate,
    safe_prefix_cut,
    threshold,
)

__all__ = [
    "AutoTemplateBank",
    "CandidatePool",
    "DetectionResult",
    "PrefixCut",
    "RerankedPool",
    "compute_auto_window",
    "enroll",
    "kv_rerank",
    "locate",
    "safe_prefix_cut",
    "threshold",
]
