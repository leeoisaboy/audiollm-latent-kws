#!/usr/bin/env python3
"""Score-level multi-layer fusion on the frozen Stage-1 corpus.

The locator ships two layer-fusion paths: vector concatenation/weighted
pooling (``locator.layers``) and this late-fusion harness.  Late fusion keeps
every layer's cosine geometry intact and combines within-utterance z-scores,
which is the safer training-free option when layers have different anisotropy.

Usage::

    python scripts/stage1_layer_fusion_real.py \
        --root /path/to/v94_latent_mean_locator \
        --config k8_mean \
        --fusions "llm2;llm1:0.3,llm2:0.7;llm1:0.5,llm2:0.5;llm2:0.7,llm6:0.3" \
        --output reports/stage1_layer_fusion.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.scoring import zpool
from stage1_real_ablation import (
    CONFIGS,
    Calibrator,
    build_bank,
    fit_calibrator,
    load_feature,
    read_jsonl,
    score_utterance,
    span_frames,
)


def parse_fusion(specification: str) -> tuple[tuple[str, float], ...]:
    pairs = []
    for item in specification.split(","):
        if not item.strip():
            continue
        layer, separator, weight = item.partition(":")
        pairs.append((layer.strip(), float(weight) if separator else 1.0))
    if not pairs:
        raise ValueError("empty fusion specification")
    total = sum(abs(weight) for _, weight in pairs)
    if total <= 0:
        raise ValueError("fusion weights must not all be zero")
    return tuple((layer, weight / total) for layer, weight in pairs)


def layer_scores(
    root: Path,
    layer: str,
    templates: list[dict],
    search: list[dict],
    config: dict,
    calibration: str,
) -> tuple[list[str], np.ndarray]:
    feature_dir = root / "feats" / "h2"
    spans = {}
    for row in templates:
        path = feature_dir / "templates" / layer / f"{row['clip_id']}.npy"
        if not path.exists():
            continue
        feature = load_feature(path)
        start, end = span_frames(float(row["kw_t0"]), float(row["kw_t1"]), feature.shape[0])
        spans[row["clip_id"]] = feature[start:end]
    raw_takes = np.stack([span.mean(axis=0) for span in spans.values()])
    calibrator = fit_calibrator(raw_takes, calibration, 32)
    bank = build_bank(
        templates,
        spans,
        calibrator,
        config.get("fixed_window"),
    )
    if bank is None:
        raise ValueError(f"{layer}: no usable templates")
    scores = np.full((len(search), len(bank.words)), -np.inf, dtype=np.float32)
    for index, row in enumerate(search):
        path = feature_dir / "search" / layer / f"{row['clip_id']}.npy"
        if not path.exists():
            continue
        feature = calibrator.apply(load_feature(path))
        best, _, _ = score_utterance(
            feature,
            bank,
            fixed_window=config.get("fixed_window"),
            prototypes=bool(config.get("prototypes")),
            multi_scale=bool(config.get("multi_scale")),
        )
        scores[index] = best
    return bank.words, scores


def evaluate(
    search: list[dict],
    words: list[str],
    scores: np.ndarray,
    pool_top: int = 20,
) -> dict:
    index = {word: position for position, word in enumerate(words)}
    top1, top5, top20, ranks = [], [], [], []
    per_word: dict[str, list[float]] = collections.defaultdict(list)
    in_pool = []
    missing = 0
    for row, row_scores in zip(search, scores):
        order = np.argsort(-row_scores, kind="stable")[:pool_top]
        retained = {words[position]: rank for rank, position in enumerate(order, 1)}
        for occurrence in row["occurrences"]:
            word = occurrence["word"]
            if word not in index:
                missing += 1
                continue
            rank = retained.get(word)
            hit = rank is not None
            in_pool.append(float(hit))
            if hit:
                ranks.append(float(rank))
                top1.append(float(rank == 1))
                top5.append(float(rank <= 5))
                top20.append(1.0)
                per_word[word].append(float(rank == 1))
            else:
                top1.append(0.0)
                top5.append(0.0)
                top20.append(0.0)
                per_word[word].append(0.0)
    return {
        "n_occurrences": len(top1),
        "missing_templates": missing,
        "pool_recall": float(np.mean(in_pool)) if in_pool else 0.0,
        "top1": float(np.mean(top1)) if top1 else 0.0,
        "top1_words": float(np.mean([np.mean(values) for values in per_word.values()]))
        if per_word
        else 0.0,
        "top5": float(np.mean(top5)) if top5 else 0.0,
        "top20": float(np.mean(top20)) if top20 else 0.0,
        "rank_median": float(np.median(ranks)) if ranks else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="k8_mean")
    parser.add_argument("--calibration", default="none", choices=("none", "center", "abtt", "whiten"))
    parser.add_argument(
        "--fusions",
        default="llm2;llm1:0.3,llm2:0.7;llm1:0.5,llm2:0.5;llm2:0.7,llm6:0.3",
    )
    parser.add_argument("--pool-top", type=int, default=20)
    parser.add_argument("--output", default="reports/stage1_layer_fusion.json")
    args = parser.parse_args()

    root = Path(args.root)
    templates = read_jsonl(root / "data" / "h2" / "templates.jsonl")
    search = read_jsonl(root / "data" / "h2" / "search.jsonl")
    config = next((item for item in CONFIGS if item["name"] == args.config), None)
    if config is None:
        raise ValueError(f"unknown config {args.config!r}")

    specifications = args.fusions.replace(";", "|").split("|")
    fusions = [parse_fusion(value) for value in specifications if value.strip()]
    layers = sorted({layer for fusion in fusions for layer, _ in fusion})
    cache: dict[str, tuple[list[str], np.ndarray]] = {}
    for layer in layers:
        print(f"scoring {layer} with {args.config}", flush=True)
        cache[layer] = layer_scores(
            root, layer, templates, search, config, args.calibration
        )

    reference_words = cache[layers[0]][0]
    for layer in layers[1:]:
        if cache[layer][0] != reference_words:
            raise ValueError(f"{layer}: template vocabulary differs from {layers[0]}")

    report = {
        "config": args.config,
        "calibration": args.calibration,
        "pool_top": args.pool_top,
        "fusion": {},
    }
    for fusion in fusions:
        name = ",".join(f"{layer}:{weight:g}" for layer, weight in fusion)
        scores = np.zeros_like(cache[fusion[0][0]][1], dtype=np.float64)
        for layer, weight in fusion:
            layer_matrix = cache[layer][1].astype(np.float64)
            for index, row in enumerate(layer_matrix):
                scores[index] += weight * zpool(row)
        report["fusion"][name] = evaluate(search, reference_words, scores, args.pool_top)
        print(name, report["fusion"][name], flush=True)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
