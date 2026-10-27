from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from common.scoring import rank_of


def pool_recall(pool_rows: Sequence[Mapping[str, Any]], pool_top: int = 20) -> float:
    values = []
    for row in pool_rows:
        words = row["words"][:pool_top]
        values.extend(1.0 if item["word"] in words else 0.0 for item in row.get("gold", []))
    return float(np.mean(values)) if values else 0.0


def occurrence_metrics(records: Sequence[Mapping[str, Any]], top_k=(1, 3, 5, 20)) -> dict[str, float]:
    if not records:
        return {"n": 0}
    output = {"n": len(records), "pool_recall": float(np.mean([record["in_pool"] for record in records]))}
    for top in top_k:
        output[f"top{top}"] = float(np.mean([record["rank"] <= top for record in records]))
    by_word: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_word[record["word"]].append(float(record["rank"] <= 1))
    output["top1_words"] = float(np.mean([np.mean(values) for values in by_word.values()]))
    output["rank_median"] = float(np.median([record["rank"] for record in records]))
    return output


def score_records(
    pool_rows,
    lm_rows,
    variant: str,
    lambda_lm: float = 4.0,
    pool_top: int = 20,
    calibration: str = "none",
    gamma: float = 1.0,
):
    from common.scoring import fuse, zpool

    records = []
    for pool_row in pool_rows:
        lm_row = lm_rows.get(pool_row["utt"])
        if lm_row is None:
            continue
        take = min(pool_top, int(pool_row.get("ntop", len(pool_row["words"]))), len(pool_row["words"]))
        words = pool_row["words"][:take]
        acoustic = np.asarray(pool_row["score"][:take], dtype=np.float64)
        language_model = np.asarray(
            [lm_row.get("scores", {}).get(word, {}).get(variant, np.nan) for word in words],
            dtype=np.float64,
        )
        if calibration == "pmi":
            text_scores = np.asarray(
                [
                    lm_row.get("text_scores", {}).get(word, {}).get(variant, np.nan)
                    for word in words
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(text_scores)):
                raise ValueError("PMI calibration requested but text-only scores are missing")
            language_model = language_model - gamma * text_scores
        elif calibration != "none":
            raise ValueError("calibration must be none or pmi")
        fused = fuse(acoustic, language_model, lambda_lm)
        winner = int(np.argmax(fused))
        for item in pool_row.get("gold", []):
            word = item["word"]
            if word not in words:
                records.append({"word": word, "in_pool": False, "rank": take + 1})
                continue
            records.append(
                {
                    "word": word,
                    "in_pool": True,
                    "rank": rank_of(fused, words.index(word)),
                    "winner": words[winner],
                }
            )
    return records
