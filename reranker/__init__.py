"""Stage 2: reading-grid alignment and frozen LM teacher forcing."""

from .align import build_contexts, candidate_arms, snapped_cut_text
from .scoring import AudioLMScorer, rerank_pool

__all__ = [
    "AudioLMScorer",
    "build_contexts",
    "candidate_arms",
    "rerank_pool",
    "snapped_cut_text",
]
