"""Non-decision analysis tools: alignment audit, shape operators, evaluation."""

from .audit import audit_prefix_pollution
from .evaluation import occurrence_metrics, pool_recall
from .shape import (
    dtw_distance,
    first_difference,
    resample,
    sequence_entropy,
    shape_similarity,
    shape_template,
    shift_invariant_similarity,
    subsequence_dtw_distance,
)

__all__ = [
    "audit_prefix_pollution",
    "dtw_distance",
    "first_difference",
    "occurrence_metrics",
    "pool_recall",
    "resample",
    "sequence_entropy",
    "shape_similarity",
    "shape_template",
    "shift_invariant_similarity",
    "subsequence_dtw_distance",
]
