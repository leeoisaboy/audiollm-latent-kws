#!/usr/bin/env python3
"""Sweep the Stage-2 language weight on already cached scores.

This script never loads the audio model.  It reuses the per-word scores from
``stage2_real_ablation.py`` and recomputes the ranking for several values of
``lambda_LM``.  It is the reproducible source of the public operating-point
table.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.scoring import rank_of, zpool


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold_word(item) -> str:
    return str(item["word"] if isinstance(item, dict) else item[0])


def evaluate(
    pool_rows: list[dict],
    score_rows: dict[str, dict],
    arm: str,
    lambda_lm: float,
    pool_top: int,
    pmi_gamma: float | None = None,
) -> tuple[dict[str, float], list[tuple[str, float]]]:
    per_word: dict[str, list[float]] = collections.defaultdict(list)
    occurrences: list[tuple[str, float]] = []
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
        if pmi_gamma is not None:
            text_scores = np.asarray(
                [row.get("text_scores", {}).get(word, np.nan) for word in words],
                dtype=np.float64,
            )
            language_model = language_model - pmi_gamma * text_scores
        fused = zpool(acoustic) + lambda_lm * zpool(language_model)
        for item in pool_row.get("gold", []):
            word = gold_word(item)
            if word not in words:
                correct = 0.0
            else:
                correct = float(rank_of(fused, words.index(word)) <= 1.0)
            per_word[word].append(correct)
            occurrences.append((pool_row["utt"], correct))
    return {
        "n_occurrences": len(occurrences),
        "top1": float(np.mean([value for _, value in occurrences])) if occurrences else 0.0,
        "top1_words": float(np.mean([np.mean(values) for values in per_word.values()]))
        if per_word
        else 0.0,
    }, occurrences


def paired_bootstrap(
    baseline: list[tuple[str, float]],
    candidate: list[tuple[str, float]],
    samples: int = 4000,
) -> dict[str, float]:
    by_utt_base: dict[str, list[float]] = collections.defaultdict(list)
    by_utt_cand: dict[str, list[float]] = collections.defaultdict(list)
    for (utt, value), (other_utt, other_value) in zip(baseline, candidate):
        if utt != other_utt:
            raise ValueError("bootstrap inputs are not aligned")
        by_utt_base[utt].append(value)
        by_utt_cand[utt].append(other_value)
    utterances = sorted(by_utt_base)
    rng = np.random.default_rng(20261028)
    gains = []
    for _ in range(samples):
        chosen = rng.choice(len(utterances), len(utterances), replace=True)
        base_values = [value for index in chosen for value in by_utt_base[utterances[index]]]
        cand_values = [value for index in chosen for value in by_utt_cand[utterances[index]]]
        gains.append(float(np.mean(cand_values) - np.mean(base_values)))
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
    parser.add_argument("--raw", required=True)
    parser.add_argument("--pool-top", type=int, default=20)
    parser.add_argument("--lambdas", default="0,1,1.5,2,2.5,3,3.5,4,4.25,4.5,4.75,5,5.5,6,8")
    parser.add_argument("--arms", default="C,C_phonetic")
    parser.add_argument("--pmi-gammas", default="0.25,0.5,0.75,1,1.5")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pool_rows = read_jsonl(Path(args.root) / "data" / "pool.jsonl")
    score_rows = {row["utt"]: row for row in read_jsonl(Path(args.raw))}
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]

    report: dict[str, object] = {
        "config": {
            "pool_top": args.pool_top,
            "lambdas": lambdas,
            "arms": arms,
            "n_utterances": len(pool_rows),
        },
        "metrics": {},
        "pmi_at_4.0": {},
        "bootstrap": {},
    }
    occurrences: dict[tuple[str, float], list[tuple[str, float]]] = {}
    for arm in arms:
        report["metrics"][arm] = {}
        for lambda_lm in lambdas:
            key = f"{lambda_lm:g}"
            metrics, values = evaluate(
                pool_rows, score_rows, arm, lambda_lm, args.pool_top
            )
            report["metrics"][arm][key] = metrics
            occurrences[(arm, lambda_lm)] = values

    pmi_gammas = [float(value) for value in args.pmi_gammas.split(",") if value.strip()]
    if 4.0 in lambdas and "C" in arms:
        for gamma in pmi_gammas:
            key = f"{gamma:g}"
            metrics, values = evaluate(
                pool_rows,
                score_rows,
                "C",
                4.0,
                args.pool_top,
                pmi_gamma=gamma,
            )
            report["pmi_at_4.0"][key] = metrics
            occurrences[("C_pmi", gamma)] = values

    def add_bootstrap(name: str, left: tuple[str, float], right: tuple[str, float]) -> None:
        if left not in occurrences or right not in occurrences:
            return
        report["bootstrap"][name] = paired_bootstrap(
            occurrences[left], occurrences[right]
        )

    if ("C", 1.5) in occurrences and ("C", 4.0) in occurrences:
        add_bootstrap("C_at_4.0_vs_C_at_1.5", ("C", 1.5), ("C", 4.0))
    for arm in arms:
        if arm == "C":
            continue
        if ("C", 4.0) in occurrences and (arm, 4.0) in occurrences:
            add_bootstrap(f"{arm}_at_4.0_vs_C_at_4.0", ("C", 4.0), (arm, 4.0))
        if ("C", 1.5) in occurrences and (arm, 4.0) in occurrences:
            add_bootstrap(f"{arm}_at_4.0_vs_C_at_1.5", ("C", 1.5), (arm, 4.0))
    for gamma in pmi_gammas:
        if ("C", 4.0) in occurrences and ("C_pmi", gamma) in occurrences:
            add_bootstrap(
                f"C_pmi_gamma{gamma:g}_at_4.0_vs_C_at_4.0",
                ("C", 4.0),
                ("C_pmi", gamma),
            )

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best = {}
    for arm in arms:
        best[arm] = max(
            report["metrics"][arm].items(),
            key=lambda item: (item[1]["top1"], item[1]["top1_words"]),
        )
    print(json.dumps({"best": best, "bootstrap": report["bootstrap"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
