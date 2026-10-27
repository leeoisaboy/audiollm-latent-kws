#!/usr/bin/env python3
"""Utterance-grouped nested cross-validation for entropy-gated shape fusion.

The initial grid search selected its hyperparameters on the same 3,244
occurrences used for evaluation.  This script separates selection from
assessment: for each held-out fold, the entropy temperature, gate percentile,
shape arm and beta are selected on the remaining utterances, then applied once
to the held-out split.

One utterance is assigned to exactly one fold, so multiple keyword occurrences
from the same audio never leak across train and test.

The default baseline is LM-only for comparability with the original shape
audit.  Pass ``--baseline fused`` to evaluate the deployable
``z_pool(acoustic) + lambda * z_pool(LM)`` score instead.

Usage::

    python scripts/shape_gate_cv.py --root /path/to/v96_shape_tiebreak \
        --output reports/shape_gate_cv.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from shape_gate_real import baseline_scores, entropy, read_jsonl, zscore

TEMPERATURES = (0.5, 1.0, 2.0, 4.0)
PERCENTILES = (0, 50, 60, 70, 80, 90, 100)
ARMS = ("d1", "res", "vel")
BETAS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def fold_of(utterance: str, folds: int, group: str = "utterance") -> int:
    key = utterance.rsplit("_", 1)[0] if group == "speaker" else utterance
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def fuse_rows(
    rows: list[dict],
    lm_scores: list[np.ndarray],
    shape_scores: list[np.ndarray],
    entropies: np.ndarray,
    cutoff: float,
    beta: float,
) -> list[np.ndarray]:
    fused = []
    for lm, shape, value in zip(lm_scores, shape_scores, entropies):
        combined = zscore(lm)
        if value >= cutoff:
            combined = combined + beta * zscore(shape)
        fused.append(combined)
    return fused


def correctness(rows: list[dict], scores: list[np.ndarray]) -> list[tuple[str, float]]:
    """Return in-pool occurrence correctness, matching ``shape_gate_real.py``.

    Pool misses are reported separately by Stage 1 and are identical for the
    baseline and gated branch, so including them here would only dilute the
    paired Top-1 comparison.
    """
    output = []
    for row, values in zip(rows, scores):
        index = {word: position for position, word in enumerate(row["words"])}
        for word, _, _ in row["gold"]:
            if word not in index:
                continue
            column = index[word]
            rank = 1 + int(np.count_nonzero(values > values[column]))
            output.append((row["utt"], float(rank == 1)))
    return output


def summarize(occurrences: list[tuple[str, float]]) -> dict[str, float]:
    per_word = collections.defaultdict(list)
    values = [value for _, value in occurrences]
    return {
        "n": len(values),
        "top1": float(np.mean(values)) if values else 0.0,
    }


def evaluate(rows: list[dict], scores: list[np.ndarray]) -> dict[str, float]:
    occurrences = correctness(rows, scores)
    per_word = collections.defaultdict(list)
    for row, values in zip(rows, scores):
        index = {word: position for position, word in enumerate(row["words"])}
        for word, _, _ in row["gold"]:
            if word not in index:
                continue
            per_word[word].append(float(np.argmax(values) == index[word]))
    result = summarize(occurrences)
    result["top1_words"] = (
        float(np.mean([np.mean(items) for items in per_word.values()]))
        if per_word
        else 0.0
    )
    return result


def choose(
    indices: np.ndarray,
    rows: list[dict],
    lm_scores: list[np.ndarray],
    shape_by_arm: dict[str, list[np.ndarray]],
    entropies: dict[float, np.ndarray],
    arms: tuple[str, ...],
) -> tuple[dict, list[np.ndarray], float]:
    best = None
    best_scores = None
    best_cutoff = 0.0
    train_rows = [rows[index] for index in indices]
    for temperature in TEMPERATURES:
        values = entropies[temperature][indices]
        for percentile in PERCENTILES:
            cutoff = float(np.percentile(values, percentile))
            for arm in arms:
                arm_scores = [shape_by_arm[arm][index] for index in indices]
                for beta in BETAS:
                    fused = fuse_rows(
                        train_rows,
                        [lm_scores[index] for index in indices],
                        arm_scores,
                        values,
                        cutoff,
                        beta,
                    )
                    metrics = evaluate(train_rows, fused)
                    candidate = {
                        **metrics,
                        "temperature": temperature,
                        "percentile": percentile,
                        "arm": arm,
                        "beta": beta,
                    }
                    key = (metrics["top1"], metrics["top1_words"], -beta)
                    if best is None or key > best[0]:
                        best = (key, candidate, cutoff)
                        best_scores = fused
    assert best is not None and best_scores is not None
    return best[1], best_scores, best[2]


def paired_bootstrap(
    baseline: list[tuple[str, float]],
    candidate: list[tuple[str, float]],
    samples: int = 4000,
    seed: int = 20261027,
) -> dict[str, float]:
    base: dict[str, list[float]] = collections.defaultdict(list)
    cand: dict[str, list[float]] = collections.defaultdict(list)
    for (utterance, left), (other, right) in zip(baseline, candidate):
        if utterance != other:
            raise ValueError("bootstrap inputs are not aligned")
        base[utterance].append(left)
        cand[utterance].append(right)
    utterances = sorted(base)
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(samples):
        chosen = rng.choice(len(utterances), len(utterances), replace=True)
        left = [value for index in chosen for value in base[utterances[index]]]
        right = [value for index in chosen for value in cand[utterances[index]]]
        gains.append(float(np.mean(right) - np.mean(left)))
    observed = float(np.mean([value for _, value in candidate])) - float(
        np.mean([value for _, value in baseline])
    )
    return {
        "mean_gain_pts": 100.0 * observed,
        "ci95_low_pts": 100.0 * float(np.percentile(gains, 2.5)),
        "ci95_high_pts": 100.0 * float(np.percentile(gains, 97.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group", choices=("utterance", "speaker"), default="utterance")
    parser.add_argument("--arms", default="d1,res,vel")
    parser.add_argument("--shape-file", default="")
    parser.add_argument("--pool-file", default="")
    parser.add_argument("--baseline", choices=("lm", "fused"), default="lm")
    parser.add_argument("--lambda-lm", type=float, default=4.0)
    parser.add_argument("--output", default="reports/shape_gate_cv.json")
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_jsonl(Path(args.shape_file) if args.shape_file else root / "data" / "shape.jsonl")
    gold_by_utt = {
        row["utt"]: row["gold"]
        for row in read_jsonl(Path(args.pool_file) if args.pool_file else root / "data" / "pool.jsonl")
    }
    for row in rows:
        row["gold"] = gold_by_utt.get(row["utt"], [])
    lm_scores = [np.asarray(row["lm_C"], dtype=np.float64) for row in rows]
    base_scores = baseline_scores(rows, args.baseline, args.lambda_lm)
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    shape_by_arm = {
        arm: [np.asarray(row[arm], dtype=np.float64) for row in rows] for arm in arms
    }
    entropies = {
        temperature: np.asarray(
            [entropy(value, temperature) for value in lm_scores], dtype=np.float64
        )
        for temperature in TEMPERATURES
    }
    folds = [
        np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if fold_of(row["utt"], args.folds, args.group) == fold
            ],
            dtype=np.int64,
        )
        for fold in range(args.folds)
    ]
    fold_report = []
    baseline_occurrences: list[tuple[str, float]] = []
    gated_occurrences: list[tuple[str, float]] = []
    for fold, test_indices in enumerate(folds):
        train_mask = np.ones(len(rows), dtype=bool)
        train_mask[test_indices] = False
        train_indices = np.flatnonzero(train_mask)
        selected, _, cutoff = choose(
            train_indices, rows, base_scores, shape_by_arm, entropies, arms
        )
        test_rows = [rows[index] for index in test_indices]
        test_base = [base_scores[index] for index in test_indices]
        test_shape = [shape_by_arm[selected["arm"]][index] for index in test_indices]
        baseline = test_base
        gated = fuse_rows(
            test_rows,
            test_base,
            test_shape,
            entropies[selected["temperature"]][test_indices],
            cutoff,
            selected["beta"],
        )
        baseline_metrics = evaluate(test_rows, baseline)
        gated_metrics = evaluate(test_rows, gated)
        baseline_occurrences.extend(correctness(test_rows, baseline))
        gated_occurrences.extend(correctness(test_rows, gated))
        fold_report.append(
            {
                "fold": fold,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                "selected": selected,
                "cutoff": cutoff,
                "baseline": baseline_metrics,
                "gated": gated_metrics,
                "gain_pts": 100.0 * (gated_metrics["top1"] - baseline_metrics["top1"]),
            }
        )
        print(json.dumps(fold_report[-1], ensure_ascii=False), flush=True)

    baseline_all = summarize(baseline_occurrences)
    gated_all = summarize(gated_occurrences)
    report = {
        "folds": args.folds,
        "group": args.group,
        "selection": f"nested {args.group}-grouped CV",
        "baseline_mode": args.baseline,
        "lambda_lm": args.lambda_lm,
        "baseline": baseline_all,
        "nested_gated": gated_all,
        "gain_pts": 100.0 * (gated_all["top1"] - baseline_all["top1"]),
        "bootstrap": paired_bootstrap(baseline_occurrences, gated_occurrences),
        "fold_details": fold_report,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
