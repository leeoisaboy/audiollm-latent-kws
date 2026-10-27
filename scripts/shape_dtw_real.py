#!/usr/bin/env python3
"""Real-corpus subsequence-DTW temporal-shape audit.

The original first-difference score linearly resampled every keyword window
and flattened it, which is sensitive to a one-frame onset shift.  This script
keeps the same frozen latents and enrollment-only fitting protocol but compares
three operators in one common 64-D PCA velocity space:

* linear resampling plus cosine (the original shape operator);
* shift-invariant frame cross-correlation;
* subsequence DTW with free endpoint boundaries.

PCA is fitted only on enrollment-keyword velocity vectors.  No evaluation
labels, transcripts or thresholds are used to build the representation.

Usage::

    python scripts/shape_dtw_real.py \
        --root /path/to/v96_shape_tiebreak \
        --features-root /path/to/v94_latent_mean_locator \
        --output reports/shape_dtw_real.jsonl \
        --report reports/shape_dtw_real.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.shape import (
    first_difference,
    resample,
    shift_invariant_similarity,
    subsequence_dtw_distance,
)
from locator import DirectoryLatentStore, build_templates
from locator.templates import l2_normalize
from stage1_real_ablation import load_feature, read_jsonl, span_frames


def pca_velocity_basis(spans: dict[str, np.ndarray], rank: int) -> tuple[np.ndarray, np.ndarray]:
    velocities = []
    for span in spans.values():
        if len(span) >= 2:
            velocities.append(first_difference(span).mean(axis=0))
    if not velocities:
        raise ValueError("no enrollment sequence long enough for a velocity PCA")
    matrix = np.stack(velocities).astype(np.float64)
    mean = matrix.mean(axis=0)
    _, _, components = np.linalg.svd(matrix - mean, full_matrices=False)
    keep = max(1, min(int(rank), components.shape[0] - 1, components.shape[1]))
    return mean, components[:keep].T


def project_velocity(difference: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(difference, dtype=np.float64) - mean) @ basis


def build_shape_templates(
    spans: dict[str, np.ndarray],
    templates: list[dict],
    mean: np.ndarray,
    basis: np.ndarray,
    descriptor_length: int,
) -> dict[str, dict]:
    sequences: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for row in templates:
        span = spans.get(row["clip_id"])
        if span is None or len(span) < 2:
            continue
        projected = project_velocity(first_difference(span), mean, basis)
        sequences[row["word"]].append(projected)
    output = {}
    for word, values in sequences.items():
        flat = l2_normalize(
            np.mean(
                [
                    l2_normalize(resample(value, descriptor_length).reshape(-1))
                    for value in values
                ],
                axis=0,
            )
        )
        resampled = [resample(value, descriptor_length) for value in values]
        output[word] = {
            "sequences": values,
            "flat": flat,
            "resampled": resampled,
        }
    return output


def score_candidate(
    projected_query: np.ndarray,
    word_templates: dict,
    descriptor_length: int,
) -> tuple[float, float, float]:
    linear = float(
        l2_normalize(resample(projected_query, descriptor_length).reshape(-1))
        @ word_templates["flat"]
    )
    shifted = max(
        shift_invariant_similarity(projected_query, template, max_shift=2)
        for template in word_templates["resampled"]
    )
    dtw = min(
        subsequence_dtw_distance(template, projected_query)
        for template in word_templates["sequences"]
    )
    return linear, float(shifted), -float(dtw)


def shape_metrics(rows: list[dict], key: str) -> dict[str, float]:
    top1 = []
    per_word: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        scores = np.asarray(row[key], dtype=np.float64)
        index = {word: position for position, word in enumerate(row["words"])}
        for word, _, _ in row["gold"]:
            if word not in index:
                continue
            correct = float(np.argmax(scores) == index[word])
            top1.append(correct)
            per_word[word].append(correct)
    return {
        "n": len(top1),
        "top1": float(np.mean(top1)) if top1 else 0.0,
        "top1_words": float(np.mean([np.mean(values) for values in per_word.values()]))
        if per_word
        else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--layer", default="llm2")
    parser.add_argument("--pca-rank", type=int, default=64)
    parser.add_argument("--descriptor-length", type=int, default=12)
    parser.add_argument("--output", default="reports/shape_dtw_real.jsonl")
    parser.add_argument("--report", default="reports/shape_dtw_real.json")
    args = parser.parse_args()

    root = Path(args.root)
    features_root = Path(args.features_root)
    rows = read_jsonl(root / "data" / "shape.jsonl")
    gold_by_utt = {
        row["utt"]: row["gold"] for row in read_jsonl(root / "data" / "pool.jsonl")
    }
    for row in rows:
        row["gold"] = gold_by_utt.get(row["utt"], [])
    templates = read_jsonl(features_root / "data" / "h2" / "templates.jsonl")
    feature_dir = features_root / "feats" / "h2"

    spans = {}
    for row in templates:
        path = feature_dir / "templates" / args.layer / f"{row['clip_id']}.npy"
        if not path.exists():
            continue
        feature = load_feature(path)
        start, end = span_frames(float(row["kw_t0"]), float(row["kw_t1"]), feature.shape[0])
        spans[row["clip_id"]] = feature[start:end]
    mean, basis = pca_velocity_basis(spans, args.pca_rank)
    shape_templates = build_shape_templates(
        spans, templates, mean, basis, args.descriptor_length
    )
    bank = build_templates(
        templates,
        DirectoryLatentStore(feature_dir / "templates" / args.layer),
    )

    output = []
    for index, row in enumerate(rows, 1):
        path = feature_dir / "search" / args.layer / f"{row['utt']}.npy"
        if not path.exists():
            continue
        feature = load_feature(path)
        pca_linear, pca_shift, dtw = [], [], []
        for word, start in zip(row["words"], row["ps"]):
            width = bank.window_for(word)
            begin = max(0, int(start) - 2)
            end = min(len(feature), int(start) + width + 2)
            if end - begin < 2 or word not in shape_templates:
                pca_linear.append(-1.0)
                pca_shift.append(-1.0)
                dtw.append(-2.0)
                continue
            query = project_velocity(first_difference(feature[begin:end]), mean, basis)
            linear, shifted, dtw_score = score_candidate(
                query, shape_templates[word], args.descriptor_length
            )
            pca_linear.append(linear)
            pca_shift.append(shifted)
            dtw.append(dtw_score)
        copied = {
            key: row[key]
            for key in ("utt", "t", "words", "ps", "pool_score", "ac", "lm_C", "d1", "res", "vel")
            if key in row
        }
        copied.update(
            {
                "pca_linear": pca_linear,
                "pca_shift": pca_shift,
                "dtw": dtw,
                "gold": row["gold"],
            }
        )
        output.append(copied)
        if index % 100 == 0:
            print(json.dumps({"done": index, "total": len(rows)}), flush=True)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "config": {
            "layer": args.layer,
            "pca_rank": args.pca_rank,
            "descriptor_length": args.descriptor_length,
            "n_utterances": len(output),
        },
        "shape_only": {
            key: shape_metrics(output, key)
            for key in ("d1", "res", "vel", "pca_linear", "pca_shift", "dtw")
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
