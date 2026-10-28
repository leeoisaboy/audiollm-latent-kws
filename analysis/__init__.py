"""Production diagnostics: candidate evaluation and prefix-leakage audit."""

from .audit import audit_prefix_pollution
from .evaluation import occurrence_metrics, pool_recall

__all__ = [
    "audit_prefix_pollution",
    "occurrence_metrics",
    "pool_recall",
]
