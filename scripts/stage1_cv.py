#!/usr/bin/env python3
"""Utterance-grouped nested CV for Stage-1 window and calibration choices.

The real-corpus ablation selected k8 plus mean centering on the same 3,244
occurrences used for reporting.  This harness separates selection from
assessment: each held-out fold chooses its window/calibration pair on the
remaining utterances, using occurrence Top-1 (then pool recall) as the
pre-registered objective, and is evaluated once on the held-out split.

Usage::

    python scripts/stage1_cv.py \
        --root /path/to/v94_latent_mean_locator \
        --output reports/stage1_cv.json
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

from stage1_real_ablation import (
    CONFIGS,
    build_bank,
    fit_calibrator,
    load_feature,
    read_jsonl,
    score_utterance,
    span_frames,
)

CANDIDATES = (
    ("k5_mean", "none"),
    ("k8_mean", "none"),
    ("k5_mean", "center"),
    ("k8_mean", "center"),
)
BASELINE = ("k5_mean", "none")


def fold_of(utterance: str, folds: int, group: str = "utterance") -> int:
    key = utterance.rsplit("_", 1)[0] if group == "speaker" else utterance
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def summarize(rows: list[dict]) -> dict[str, float]:
    per_word: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        per_word[row["word"]].append(row["top1"])
    return {
        "n": len(rows),
        "top1": float(np.mean([row["top1"] for row in rows])) if rows else 0.0,
        "top1_words": float(np.mean([np.mean(v) for v in per_word.values()]))
        if per_word
        else 0.0,
        "top20": float(np.mean([row["top20"] for row in rows])) if rows else 0.0,
    }


def paired_bootstrap(
    baseline: list[dict], candidate: list[dict], samples: int = 4000
) -> dict[str, float]:
    base: dict[str, list[float]] = collections.defaultdict(list)
    cand: dict[str, list[float]] = collections.defaultdict(list)
    for left, right in zip(baseline, candidate):
        if left["utt"] != right["utt"]:
            raise ValueError("bootstrap inputs are not aligned")
        base[left["utt"]].append(left["top1"])
        cand[left["utt"]].append(right["top1"])
    utterances = sorted(base)
    rng = np.random.default_rng(20261027)
    gains = []
    for _ in range(samples):
        chosen = rng.choice(len(utterances), len(utterances), replace=True)
        left = [value for index in chosen for value in base[utterances[index]]]
        right = [value for index in chosen for value in cand[utterances[index]]]
        gains.append(float(np.mean(right) - np.mean(left)))
    observed = float(np.mean([row["top1"] for row in candidate])) - float(
        np.mean([row["top1"] for row in baseline])
    )
    return {
        "mean_gain_pts": 100.0 * observed,
        "ci95_low_pts": 100.0 * float(np.percentile(gains, 2.5)),
        "ci95_high_pts": 100.0 * float(np.percentile(gains, 97.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--layer", default="llm2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group", choices=("utterance", "speaker"), default="utterance")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="reports/stage1_cv.json")
    args = parser.parse_args()

    root = Path(args.root)
    feature_dir = root / "feats" / "h2"
    templates = read_jsonl(root / "data" / "h2" / "templates.jsonl")
    search = read_jsonl(root / "data" / "h2" / "search.jsonl")
    if args.limit:
        search = search[: args.limit]

    spans: dict[str, np.ndarray] = {}
    for row in templates:
        path = feature_dir / "templates" / args.layer / f"{row['clip_id']}.npy"
        if not path.exists():
            continue
        feature = load_feature(path)
        start, end = span_frames(
            float(row["kw_t0"]), float(row["kw_t1"]), feature.shape[0]
        )
        spans[row["clip_id"]] = feature[start:end]
    raw_takes = np.stack([span.mean(axis=0) for span in spans.values()])
    config_by_name = {config["name"]: config for config in CONFIGS}
    calibrators = {
        name: fit_calibrator(raw_takes, name, 32)
        for name in sorted({calibration for _, calibration in CANDIDATES})
    }
    banks = {
        (name, calibration): build_bank(
            templates,
            spans,
            calibrators[calibration],
            config_by_name[name].get("fixed_window"),
            int(config_by_name[name].get("kmeans", 0)),
        )
        for name, calibration in CANDIDATES
    }

    records: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for utterance_index, row in enumerate(search, 1):
        path = feature_dir / "search" / args.layer / f"{row['clip_id']}.npy"
        if not path.exists():
            continue
        feature = load_feature(path)
        calibrated = {
            name: calibrator.apply(feature) for name, calibrator in calibrators.items()
        }
        scores_by_key = {}
        for key in CANDIDATES:
            bank = banks[key]
            if bank is None:
                continue
            config = config_by_name[key[0]]
            features = calibrated[key[1]]
            scores, starts, widths = score_utterance(
                features,
                bank,
                fixed_window=config.get("fixed_window"),
                prototypes=False,
                multi_scale=bool(config.get("multi_scale")),
            )
            scores_by_key[key] = (bank, scores, starts, widths, features.shape[0])
        for occurrence in row["occurrences"]:
            word = occurrence["word"]
            true_start, true_end = span_frames(
                float(occurrence["t0"]),
                float(occurrence["t1"]),
                scores_by_key[CANDIDATES[0]][4],
            )
            for key, (bank, scores, starts, widths, _) in scores_by_key.items():
                column = bank.index.get(word)
                if column is None:
                    continue
                rank = 1 + int(np.count_nonzero(scores > scores[column]))
                width = int(widths[column])
                start = int(starts[column])
                records[key].append(
                    {
                        "utt": row["clip_id"],
                        "word": word,
                        "top1": float(rank == 1),
                        "top20": float(rank <= 20),
                        "rank": rank,
                        "start": start,
                        "width": width,
                        "true_start": true_start,
                        "true_end": true_end,
                    }
                )
        if utterance_index % 100 == 0:
            print(
                json.dumps({"done": utterance_index, "total": len(search)}),
                flush=True,
            )

    fold_ids = {
        row["clip_id"]: fold_of(row["clip_id"], args.folds, args.group) for row in search
    }
    all_records = records[BASELINE]
    fold_details = []
    selected_records: list[dict] = []
    baseline_records: list[dict] = []
    for fold in range(args.folds):
        train_rows = [row for row in all_records if fold_ids[row["utt"]] != fold]
        test_rows = [row for row in all_records if fold_ids[row["utt"]] == fold]
        best_key = None
        best_metrics = None
        for key in CANDIDATES:
            rows = [row for row in records[key] if fold_ids[row["utt"]] != fold]
            metrics = summarize(rows)
            rank_key = (metrics["top1"], metrics["top20"], metrics["top1_words"])
            if best_key is None or rank_key > best_metrics[0]:
                best_key = key
                best_metrics = (rank_key, metrics)
        assert best_key is not None and best_metrics is not None
        chosen = [
            row
            for row in records[best_key]
            if fold_ids[row["utt"]] == fold
        ]
        baseline = [
            row
            for row in records[BASELINE]
            if fold_ids[row["utt"]] == fold
        ]
        selected_records.extend(chosen)
        baseline_records.extend(baseline)
        fold_details.append(
            {
                "fold": fold,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "selected": {"config": best_key[0], "calibration": best_key[1]},
                "train": best_metrics[1],
                "baseline": summarize(baseline),
                "nested": summarize(chosen),
                "gain_pts": 100.0
                * (summarize(chosen)["top1"] - summarize(baseline)["top1"]),
            }
        )
        print(json.dumps(fold_details[-1], ensure_ascii=False), flush=True)

    report = {
        "folds": args.folds,
        "group": args.group,
        "layer": args.layer,
        "selection": f"nested {args.group}-grouped CV; train Top-1, tie-break Top-20",
        "candidates": [
            {"config": config, "calibration": calibration}
            for config, calibration in CANDIDATES
        ],
        "baseline": summarize(baseline_records),
        "nested_selected": summarize(selected_records),
        "gain_pts": 100.0
        * (
            summarize(selected_records)["top1"]
            - summarize(baseline_records)["top1"]
        ),
        "bootstrap": paired_bootstrap(baseline_records, selected_records),
        "fold_details": fold_details,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
