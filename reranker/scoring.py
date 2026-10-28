"""Stage-2 scoring for the single production arm: snapped C."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from common.config import FRAME_S
from reranker.align import build_prefixes


class AudioLMScorer(Protocol):
    """A frozen audio-language model scorer."""

    def score(
        self,
        utterance: str,
        wav_path: str,
        prefixes: Sequence[str],
        candidates: Sequence[str],
    ) -> np.ndarray:
        ...


@dataclass
class RerankResult:
    utterance: str
    scores: dict[str, float]
    prefixes: list[str]

    def to_row(self) -> dict[str, Any]:
        return {
            "utt": self.utterance,
            "arm": "C",
            "scores": {
                word: {"C": float(value)} for word, value in self.scores.items()
            },
            "prefixes": self.prefixes,
        }


def rerank_pool(
    pool_row: Mapping[str, Any],
    chars: Sequence[tuple[int, str]],
    read_frame_s: float,
    scorer: AudioLMScorer,
    wav_path: str,
    frame_s: float = FRAME_S,
    phonetic: bool = False,
    max_phoneme_distance: int = 1,
) -> RerankResult:
    words = list(pool_row["words"])
    starts = [int(value) for value in pool_row["ps"]]
    prefixes = build_prefixes(
        words,
        starts,
        chars,
        read_frame_s,
        frame_s=frame_s,
        phonetic=phonetic,
        max_phoneme_distance=max_phoneme_distance,
    )
    values = scorer.score(
        wav_path=wav_path,
        utterance=pool_row["utt"],
        prefixes=prefixes,
        candidates=words,
    )
    resolved = np.asarray(values, dtype=np.float64)
    if len(resolved) != len(words):
        raise ValueError("scorer must return one value per candidate")
    return RerankResult(
        utterance=pool_row["utt"],
        scores={word: float(value) for word, value in zip(words, resolved)},
        prefixes=prefixes,
    )


def load_scorer(specification: str, **options: Any) -> AudioLMScorer:
    """Load ``module:factory`` and invoke it with the remaining options."""
    try:
        module_name, function_name = specification.split(":", 1)
    except ValueError as error:
        raise ValueError("scorer must use module:factory syntax") from error
    module = import_module(module_name)
    factory = getattr(module, function_name)
    parameters = signature(factory).parameters
    accepts_kwargs = any(
        parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if not accepts_kwargs:
        options = {key: value for key, value in options.items() if key in parameters}
    return factory(**options)
