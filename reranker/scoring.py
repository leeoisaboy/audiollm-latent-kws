from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from reranker.align import build_contexts, candidate_arms


class AudioLMScorer(Protocol):
    """A frozen audio-language model scorer.

    Implementations receive CJK prefixes and candidate words and return one mean
    candidate-token log probability. Audio encoding and tokenization belong to
    the implementation, which keeps this package model-agnostic.
    """

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
    scores: dict[str, dict[str, float]]
    prefixes: dict[str, list[str]]
    text_scores: dict[str, dict[str, float]] | None = None

    def to_row(self) -> dict[str, Any]:
        row = {"utt": self.utterance, "scores": self.scores, "prefixes": self.prefixes}
        if self.text_scores is not None:
            row["text_scores"] = self.text_scores
        return row


def rerank_pool(
    pool_row: Mapping[str, Any],
    chars: Sequence[tuple[int, str]],
    read_frame_s: float,
    scorer: AudioLMScorer,
    wav_path: str,
    variants: Sequence[str] = ("C",),
    frame_s: float = 0.08,
    null_chars: Sequence[tuple[int, str]] | None = None,
    gold: Sequence[tuple[str, float, float]] | None = None,
    transcript: str = "",
    phonetic: bool = False,
    max_phoneme_distance: int = 1,
    alternate_starts: bool = False,
    text_only: bool = False,
) -> RerankResult:
    words = list(pool_row["words"])
    starts = [int(value) for value in pool_row["ps"]]
    contexts = build_contexts(
        words,
        starts,
        chars,
        read_frame_s,
        frame_s=frame_s,
        variants=variants,
        null_chars=null_chars,
        gold=gold,
        transcript=transcript,
        phonetic=phonetic,
        max_phoneme_distance=max_phoneme_distance,
    )
    scores: dict[str, dict[str, float]] = {word: {} for word in words}
    for arm, prefixes in contexts.items():
        active_words: Sequence[str] = words
        active_prefixes: Sequence[str] = prefixes
        owners: list[int] | None = None
        if alternate_starts and arm in {"P", "C", "X"} and pool_row.get("ps_alt"):
            active_words = []
            active_prefixes = []
            owners = []
            expanded_starts = []
            for word_index, word in enumerate(words):
                starts_for_word = [int(value) for value in pool_row["ps_alt"][word_index]]
                if int(pool_row["ps"][word_index]) not in starts_for_word:
                    starts_for_word.insert(0, int(pool_row["ps"][word_index]))
                for start in starts_for_word:
                    active_words.append(word)
                    owners.append(word_index)
                    expanded_starts.append(start)
            expanded = build_contexts(
                active_words,
                expanded_starts,
                chars,
                read_frame_s,
                frame_s=frame_s,
                variants=(arm,),
                null_chars=null_chars,
                phonetic=phonetic,
                max_phoneme_distance=max_phoneme_distance,
            )[arm]
            active_prefixes = expanded
        values = scorer.score(
            wav_path=wav_path,
            utterance=pool_row["utt"],
            prefixes=active_prefixes,
            candidates=active_words,
        )
        if owners is None:
            resolved = np.asarray(values, dtype=np.float64)
        else:
            resolved = np.full(len(words), -np.inf, dtype=np.float64)
            for owner, value in zip(owners, values):
                resolved[owner] = max(resolved[owner], float(value))
        for index, word in enumerate(words):
            scores[word][arm] = float(resolved[index])
    text_scores = None
    if text_only:
        method = getattr(scorer, "score_text_only", None)
        if method is None:
            raise ValueError("scorer does not implement score_text_only for PMI calibration")
        text_scores = {word: {} for word in words}
        for arm, prefixes in contexts.items():
            values = method(prefixes=prefixes, candidates=words)
            for word, value in zip(words, values):
                text_scores[word][arm] = float(value)
    return RerankResult(pool_row["utt"], scores, contexts, text_scores)


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
