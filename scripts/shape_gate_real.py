#!/usr/bin/env python3
"""Entropy-gated acoustic-shape fusion on the real corpus.

The earlier shape experiment combined ``lm_C`` and a first-difference shape
score with a single global weight beta.  Every beta hurt top-1, because the
shape score is much weaker than the language model (top-1 0.213) and its
errors are not limited to cases where the LM is unsure.

This script tests the uncertainty-aware variant: normalise both scores inside
one utterance, apply the shape term only when the LM distribution over the
pool is high-entropy, and sweep the entropy threshold and beta.  Pass
``--baseline fused`` to test the deployable acoustic+LM score rather than the
LM-only diagnostic.

Input is ``data/shape.jsonl`` from the v96 experiment::

    utt, t, words, pool_score, ac, ps, lm_C, d1, res, vel

Usage::

    python scripts/shape_gate_real.py --root /path/to/v96_shape_tiebreak \
        --output reports/shape_gate.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(float(values.std()), 1e-6)


def entropy(scores: np.ndarray, temperature: float, normalize: bool = True) -> float:
    scaled = zscore(scores) / max(temperature, 1e-6)
    scaled = scaled - scaled.max()
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    value = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    return value / np.log(len(scores)) if normalize else value


def baseline_scores(rows: list[dict], mode: str, lambda_lm: float) -> list[np.ndarray]:
    """Build the score vector to which the gated shape term is added.

    ``lm`` reproduces the original diagnostic. ``fused`` evaluates the
    deployable ``z_pool(acoustic) + lambda_lm * z_pool(LM)`` baseline.
    """
    lm = [np.asarray(row["lm_C"], dtype=np.float64) for row in rows]
    if mode == "lm":
        return lm
    acoustic = [np.asarray(row["ac"], dtype=np.float64) for row in rows]
    return [zscore(ac) + lambda_lm * zscore(value) for ac, value in zip(acoustic, lm)]


def evaluate(rows: list[dict], scores_by_row: list[np.ndarray]) -> dict:
    top1, per_word, n = [], collections.defaultdict(list), 0
    for row, scores in zip(rows, scores_by_row):
        index = {word: position for position, word in enumerate(row["words"])}
        for word, _, _ in row["gold"]:
            if word not in index:
                continue
            column = index[word]
            rank = 1 + int(np.count_nonzero(scores > scores[column]))
            value = 1.0 if rank == 1 else 0.0
            top1.append(value)
            per_word[word].append(value)
            n += 1
    return {
        "n": n,
        "top1": float(np.mean(top1)) if top1 else float("nan"),
        "top1_words": float(np.mean([np.mean(v) for v in per_word.values()])) if per_word else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="v96 experiment directory")
    parser.add_argument("--output", default="reports/shape_gate.json")
    parser.add_argument("--shape-arm", default="d1")
    parser.add_argument("--arms", default="d1,res,vel")
    parser.add_argument("--shape-file", default="")
    parser.add_argument("--pool-file", default="")
    parser.add_argument("--baseline", choices=("lm", "fused"), default="lm")
    parser.add_argument("--lambda-lm", type=float, default=4.0)
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_jsonl(Path(args.shape_file) if args.shape_file else root / "data" / "shape.jsonl")
    gold_by_utt = {
        row["utt"]: row["gold"]
        for row in read_jsonl(Path(args.pool_file) if args.pool_file else root / "data" / "pool.jsonl")
    }
    for row in rows:
        row["gold"] = gold_by_utt.get(row["utt"], [])
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    entropies = {
        temperature: np.asarray([entropy(np.asarray(row["lm_C"], dtype=np.float64), temperature) for row in rows])
        for temperature in (0.5, 1.0, 2.0, 4.0)
    }
    lm_scores = [np.asarray(row["lm_C"], dtype=np.float64) for row in rows]
    base_scores = baseline_scores(rows, args.baseline, args.lambda_lm)
    baseline = evaluate(rows, base_scores)
    thresholds = {
        str(percentile): float(np.percentile(entropies[1.0], percentile))
        for percentile in (0, 50, 60, 70, 80, 90, 100)
    }

    results: dict = {
        "baseline": baseline,
        "baseline_mode": args.baseline,
        "lambda_lm": args.lambda_lm,
        "thresholds": thresholds,
        "sweep": {},
    }
    for temperature in (0.5, 1.0, 2.0, 4.0):
        values = entropies[temperature]
        for percentile in (0, 50, 60, 70, 80, 90, 100):
            cut = float(np.percentile(values, percentile))
            for arm in arms:
                shape = [np.asarray(row[arm], dtype=np.float64) for row in rows]
                for beta in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
                    fused = []
                    for base, sh, value in zip(base_scores, shape, values):
                        combined = zscore(base)
                        if value >= cut:
                            combined = combined + beta * zscore(sh)
                        fused.append(combined)
                    metrics = evaluate(rows, fused)
                    gate_rate = float(np.mean(values >= cut))
                    key = f"{temperature}|p{percentile}|{arm}|{beta}"
                    results["sweep"][key] = {
                        **metrics,
                        "temperature": temperature,
                        "percentile": percentile,
                        "arm": arm,
                        "beta": beta,
                        "gate_rate": gate_rate,
                        "net_gain_pts": 100.0 * (metrics["top1"] - baseline["top1"]),
                    }
    # Diagnostics: do LM errors concentrate in high entropy?
    errors_entropy, correct_entropy = [], []
    for row_index, (row, lm) in enumerate(zip(rows, lm_scores)):
        index = {word: position for position, word in enumerate(row["words"])}
        value = entropies[1.0][row_index]
        for word, _, _ in row["gold"]:
            if word not in index:
                continue
            column = index[word]
            rank = 1 + int(np.count_nonzero(lm > lm[column]))
            (errors_entropy if rank > 1 else correct_entropy).append(value)
    results["lm_error_entropy"] = {
        "correct_mean": float(np.mean(correct_entropy)),
        "error_mean": float(np.mean(errors_entropy)),
        "correct_n": len(correct_entropy),
        "error_n": len(errors_entropy),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    best = sorted(results["sweep"].items(), key=lambda item: -item[1]["top1"])[:10]
    print("baseline", baseline)
    for key, value in best:
        print(f"{key:28s} top1={value['top1']:.4f} words={value['top1_words']:.4f} "
              f"gain={value['net_gain_pts']:+.2f} gate={value['gate_rate']:.3f}")
    print("lm error entropy", results["lm_error_entropy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
