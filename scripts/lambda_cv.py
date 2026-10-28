#!/usr/bin/env python3
"""Nested grouped cross-validation for the Stage-2 arm and LM weight.

The lambda sweep in ``scripts/lambda_sweep.py`` chooses a point on the same
occurrences used for reporting.  This script separates selection from
assessment.  Each held-out fold chooses an arm and weight from cached scores on
the remaining utterances, then evaluates the selected point once on the
held-out split.

Usage::

    python scripts/lambda_cv.py \
        --root /path/to/stage2-cache \
        --raw /path/to/stage2_real_lm.jsonl \
        --group speaker \
        --output reports/lambda_cv_speaker.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import read_jsonl
from common.scoring import rank_of, zpool


def fold_of(utterance: str, folds: int = 5, group: str = "utterance") -> int:
    key = utterance.rsplit("_", 1)[0] if group == "speaker" else utterance
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def gold_word(item) -> str:
    return str(item["word"] if isinstance(item, dict) else item[0])


def evaluate(
    pool_rows: list[dict],
    score_rows: dict[str, dict],
    arm: str,
    lambda_lm: float,
    pool_top: int,
) -> list[dict]:
    records = []
    for pool_row in pool_rows:
        row = score_rows.get(pool_row["utt"])
        if row is None:
            continue
        words = list(pool_row["words"][:pool_top])
        acoustic = np.asarray(pool_row["score"][:pool_top], dtype=np.float64)
        language_model = np.asarray(
            [row["scores"][arm].get(word, np.nan) for word in words],
            dtype=np.float64,
        )
        fused = zpool(acoustic) + lambda_lm * zpool(language_model)
        for item in pool_row.get("gold", []):
            word = gold_word(item)
            if word not in words:
                correct = 0.0
            else:
                correct = float(rank_of(fused, words.index(word)) <= 1.0)
            records.append({"utt": pool_row["utt"], "word": word, "top1": correct})
    return records


def summarize(records: list[dict]) -> dict[str, float]:
    per_word: dict[str, list[float]] = collections.defaultdict(list)
    for record in records:
        per_word[record["word"]].append(record["top1"])
    return {
        "n": len(records),
        "top1": float(np.mean([record["top1"] for record in records]))
        if records
        else 0.0,
        "top1_words": float(np.mean([np.mean(values) for values in per_word.values()]))
        if per_word
        else 0.0,
    }


def paired_bootstrap(
    baseline: list[dict], candidate: list[dict], samples: int = 4000
) -> dict[str, float]:
    by_utt_base: dict[str, list[float]] = collections.defaultdict(list)
    by_utt_candidate: dict[str, list[float]] = collections.defaultdict(list)
    for left, right in zip(baseline, candidate):
        if left["utt"] != right["utt"] or left["word"] != right["word"]:
            raise ValueError("bootstrap inputs are not aligned")
        by_utt_base[left["utt"]].append(left["top1"])
        by_utt_candidate[right["utt"]].append(right["top1"])
    utterances = sorted(by_utt_base)
    rng = np.random.default_rng(20261028)
    gains = []
    for _ in range(samples):
        chosen = rng.choice(len(utterances), len(utterances), replace=True)
        base_values = [
            value for index in chosen for value in by_utt_base[utterances[index]]
        ]
        candidate_values = [
            value for index in chosen for value in by_utt_candidate[utterances[index]]
        ]
        gains.append(float(np.mean(candidate_values) - np.mean(base_values)))
    observed = float(np.mean([r["top1"] for r in candidate])) - float(
        np.mean([r["top1"] for r in baseline])
    )
    return {
        "mean_gain_pts": 100.0 * observed,
        "ci95_low_pts": 100.0 * float(np.percentile(gains, 2.5)),
        "ci95_high_pts": 100.0 * float(np.percentile(gains, 97.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--pool-top", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group", choices=("utterance", "speaker"), default="speaker")
    parser.add_argument("--arms", default="C,C_phonetic,C_alt")
    parser.add_argument(
        "--lambdas", default="0,1,1.5,2,2.5,3,3.5,4,4.25,4.5,4.75,5,5.5,6,8"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pool_rows = read_jsonl(Path(args.root) / "data" / "pool.jsonl")
    score_rows = {row["utt"]: row for row in read_jsonl(args.raw)}
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]
    candidates = {(arm, value): evaluate(
        pool_rows, score_rows, arm, value, args.pool_top
    ) for arm in arms for value in lambdas}
    fold_ids = {
        row["utt"]: fold_of(row["utt"], args.folds, args.group) for row in pool_rows
    }

    selected_records: list[dict] = []
    fixed_default: list[dict] = []
    inherited_baseline: list[dict] = []
    fold_details = []
    for fold in range(args.folds):
        train_rows = [row for row in pool_rows if fold_ids[row["utt"]] != fold]
        test_utts = {row["utt"] for row in pool_rows if fold_ids[row["utt"]] == fold}
        best_key = None
        best_rank = None
        best_train = None
        for key, records in candidates.items():
            train_records = [record for record in records if record["utt"] not in test_utts]
            metrics = summarize(train_records)
            rank_key = (metrics["top1"], metrics["top1_words"])
            if best_rank is None or rank_key > best_rank:
                best_key = key
                best_rank = rank_key
                best_train = metrics
        assert best_key is not None and best_train is not None
        test_records = [record for record in candidates[best_key] if record["utt"] in test_utts]
        baseline_records = [
            record for record in candidates[("C", 1.5)] if record["utt"] in test_utts
        ]
        default_records = [
            record for record in candidates[("C", 4.0)] if record["utt"] in test_utts
        ]
        selected_records.extend(test_records)
        inherited_baseline.extend(baseline_records)
        fixed_default.extend(default_records)
        fold_details.append(
            {
                "fold": fold,
                "n_train": len(train_rows),
                "n_test": len(test_utts),
                "selected": {"arm": best_key[0], "lambda_lm": best_key[1]},
                "train": best_train,
                "nested": summarize(test_records),
                "baseline_C_1.5": summarize(baseline_records),
                "fixed_C_4.0": summarize(default_records),
            }
        )

    report = {
        "folds": args.folds,
        "group": args.group,
        "selection": "nested grouped CV; train occurrence Top-1, tie-break word-level Top-1",
        "candidates": {
            "arms": arms,
            "lambdas": lambdas,
        },
        "nested_selected": summarize(selected_records),
        "fixed_C_4.0": summarize(fixed_default),
        "baseline_C_1.5": summarize(inherited_baseline),
        "bootstrap": {
            "nested_vs_C_1.5": paired_bootstrap(inherited_baseline, selected_records),
            "fixed_C_4.0_vs_C_1.5": paired_bootstrap(inherited_baseline, fixed_default),
        },
        "fold_details": fold_details,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
