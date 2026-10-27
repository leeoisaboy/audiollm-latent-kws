#!/usr/bin/env python3
"""Measure peak-neighbourhood span growing on the frozen Stage-1 corpus."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locator import DirectoryLatentStore, build_pool, build_templates
from stage1_real_ablation import load_feature, read_jsonl, span_frames


def evaluate(
    search: list[dict],
    pools: list,
) -> dict[str, float]:
    top1, top20, coverage, iou, start_error = [], [], [], [], []
    per_word: dict[str, list[float]] = collections.defaultdict(list)
    for row, pool in zip(search, pools):
        retained = {word: index for index, word in enumerate(pool.words)}
        frames = pool.frames
        for occurrence in row["occurrences"]:
            word = occurrence["word"]
            index = retained.get(word)
            in_pool = index is not None
            correct = bool(in_pool and index == 0)
            top1.append(float(correct))
            top20.append(float(in_pool))
            per_word[word].append(float(correct))
            if not in_pool:
                coverage.append(0.0)
                iou.append(0.0)
                start_error.append(999.0)
                continue
            true_start, true_end = span_frames(
                float(occurrence["t0"]), float(occurrence["t1"]), frames
            )
            predicted_start = int(pool.starts[index])
            predicted_end = int(pool.ends[index])
            intersection = max(
                0, min(predicted_end, true_end) - max(predicted_start, true_start)
            )
            true_length = max(1, true_end - true_start)
            predicted_length = max(1, predicted_end - predicted_start)
            coverage.append(intersection / true_length)
            iou.append(
                intersection
                / max(1, true_length + predicted_length - intersection)
            )
            start_error.append(abs(predicted_start - true_start))
    return {
        "n": len(top1),
        "top1": float(np.mean(top1)),
        "top1_words": float(np.mean([np.mean(values) for values in per_word.values()])),
        "top20": float(np.mean(top20)),
        "coverage_mean": float(np.mean(coverage)),
        "coverage_ge_0_5": float(np.mean(np.asarray(coverage) >= 0.5)),
        "iou_median": float(np.median(iou)),
        "start_abs_error_median": float(np.median(start_error)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--layer", default="llm2")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--grow-delta", type=float, default=0.05)
    parser.add_argument("--output", default="reports/stage1_grow.json")
    args = parser.parse_args()

    root = Path(args.root)
    feature_root = root / "feats" / "h2"
    templates = read_jsonl(root / "data" / "h2" / "templates.jsonl")
    search = read_jsonl(root / "data" / "h2" / "search.jsonl")
    bank = build_templates(
        templates,
        DirectoryLatentStore(feature_root / "templates" / args.layer),
    )
    report = {
        "config": {
            "layer": args.layer,
            "window": args.window,
            "grow_delta": args.grow_delta,
            "n_utterances": len(search),
        },
        "arms": {},
    }
    for name, adaptive, grow in (
        ("k5_fixed", False, False),
        ("k5_grow", False, True),
        ("adaptive_fixed", True, False),
        ("adaptive_grow", True, True),
    ):
        pools = []
        for index, row in enumerate(search, 1):
            feature = load_feature(feature_root / "search" / args.layer / f"{row['clip_id']}.npy")
            pools.append(
                build_pool(
                    row["clip_id"],
                    feature,
                    bank,
                    pool_top=20,
                    window=args.window,
                    adaptive_window=adaptive,
                    multi_scale=0,
                    use_prototypes=True,
                    use_grow_span=grow,
                    grow_delta=args.grow_delta,
                )
            )
            if index % 250 == 0:
                print(json.dumps({"arm": name, "done": index}), flush=True)
        report["arms"][name] = evaluate(search, pools)
        print(name, report["arms"][name], flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
