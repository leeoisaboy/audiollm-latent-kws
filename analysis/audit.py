from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from common.config import FRAME_S
from reranker.align import cut_text, snapped_cut_text


def audit_prefix_pollution(
    pool_rows: Sequence[Mapping[str, Any]],
    lm_rows: Mapping[str, Mapping[str, Any]],
    readings: Mapping[str, Sequence[tuple[int, str]]],
    read_frame_s: float,
    frame_s: float = FRAME_S,
    null_shift: int = 137,
    tolerance: int = 6,
    phonetic: bool = False,
    max_phoneme_distance: int = 1,
) -> dict[str, dict[str, float]]:
    """Count self-first-character leakage for P, C, S, and the X control."""
    order = [row["utt"] for row in pool_rows]
    stats: dict[str, dict[str, list[int | float]]] = {
        arm: {"n": [], "leak": [], "ends": [], "length": []} for arm in ("P", "C", "S", "X")
    }
    for position, row in enumerate(pool_rows):
        if row["utt"] not in lm_rows:
            continue
        words = list(row["words"])
        starts = [int(value) for value in row["ps"]]
        if not words:
            continue
        index = {word: position for position, word in enumerate(words)}
        chars = readings.get(row["utt"], [])
        null_id = order[(position + null_shift) % len(order)]
        null_chars = readings.get(null_id, [])
        shared = snapped_cut_text(
            chars,
            words[0],
            starts[0] * frame_s,
            read_frame_s,
            tolerance,
            phonetic,
            max_phoneme_distance,
        )
        gold_rows = row.get("gold", [])
        normalized_gold = []
        for gold in gold_rows:
            if isinstance(gold, Mapping):
                normalized_gold.append((str(gold["word"]), gold.get("ts"), gold.get("te")))
            else:
                normalized_gold.append(tuple(gold))
        for gold in normalized_gold:
            word, _start, _end = gold
            pool_position = index.get(word)
            if pool_position is None:
                continue
            seconds = starts[pool_position] * frame_s
            contexts = {
                "P": cut_text(chars, seconds, read_frame_s),
                "C": snapped_cut_text(
                    chars,
                    word,
                    seconds,
                    read_frame_s,
                    tolerance,
                    phonetic,
                    max_phoneme_distance,
                ),
                "S": shared,
                "X": cut_text(null_chars, seconds, read_frame_s),
            }
            for arm, prefix in contexts.items():
                bucket = stats[arm]
                bucket["n"].append(1)
                bucket["leak"].append(int(bool(word) and bool(prefix) and word[0] in prefix))
                bucket["ends"].append(int(bool(prefix) and word and prefix.endswith(word[0])))
                bucket["length"].append(len(prefix))

    report: dict[str, dict[str, float]] = {}
    for arm, bucket in stats.items():
        count = len(bucket["n"])
        lengths = np.asarray(bucket["length"], dtype=np.float64)
        report[arm] = {
            "n": count,
            "contains_own_first_char": float(np.mean(bucket["leak"])) if count else 0.0,
            "ends_on_own_first_char": float(np.mean(bucket["ends"])) if count else 0.0,
            "prefix_len_median": float(np.median(lengths)) if count else 0.0,
        }
    return report
