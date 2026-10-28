#!/usr/bin/env python3
"""Stage-1 ablations on the frozen Step-Audio2 latent store.

The published prototype numbers (pool recall 0.9266, acoustic top-1 0.4399)
come from a fixed five-frame (400 ms) cosine window with a single mean
template per word.  This harness re-derives those numbers from the stored
features and then adds the training-free extensions implemented in
``locator/``:

* duration-adaptive windows and a +/-1 frame multi-scale search;
* multi-prototype (all enrolment takes) scoring;
* mean centering, All-but-the-Top and PCA whitening fitted on enrolment only;
* score-level fusion of layers llm1/llm2/llm6.

The manifest and feature layout match the v94 experiment::

    <root>/data/h2/templates.jsonl   clip_id, word, kw_t0, kw_t1
    <root>/data/h2/search.jsonl      clip_id, occurrences[{word, t0, t1}]
    <root>/feats/h2/templates/<layer>/<clip_id>.npy
    <root>/feats/h2/search/<layer>/<clip_id>.npy

Usage::

    python scripts/stage1_real_ablation.py \
        --root /path/to/v94_latent_mean_locator \
        --output reports/stage1_ablation.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.legacy.templates import _kmeans_prototypes

FRAME_S = 0.08
EPS = 1e-8


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_feature(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.shape[0] == 128 and array.shape[1] != 3584:
        array = array.T
    return array.astype(np.float32)


def l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), EPS)


def span_frames(t0: float, t1: float, frames: int) -> tuple[int, int]:
    start = max(0, min(int(round(t0 / FRAME_S)), frames - 1))
    end = max(start + 1, min(int(round(t1 / FRAME_S)), frames))
    return start, end


def window_means(feature: np.ndarray, width: int) -> np.ndarray | None:
    frames = feature.shape[0]
    if frames < width:
        return None
    cumulative = np.cumsum(feature, axis=0, dtype=np.float32)
    cumulative = np.concatenate(
        [np.zeros((1, feature.shape[1]), dtype=np.float32), cumulative], axis=0
    )
    return (cumulative[width:] - cumulative[:-width]) / float(width)


@dataclass
class Calibrator:
    mean: np.ndarray | None = None
    directions: np.ndarray | None = None
    projection: np.ndarray | None = None
    scale: np.ndarray | None = None

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = values
        if self.mean is not None:
            array = array - self.mean
        if self.directions is not None and len(self.directions):
            array = array - (array @ self.directions.T) @ self.directions
        if self.projection is not None:
            array = (array @ self.projection) * self.scale
        return array


def fit_calibrator(takes: np.ndarray, mode: str, rank: int) -> Calibrator:
    if mode == "none":
        return Calibrator()
    mean = takes.mean(axis=0)
    if mode == "center":
        return Calibrator(mean=mean)
    centered = takes - mean
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    keep = max(1, min(rank, components.shape[0] - 1))
    if mode == "abtt":
        return Calibrator(mean=mean, directions=components[:keep])
    if mode == "whiten":
        variance = (singular[:keep] ** 2) / max(1, centered.shape[0] - 1)
        return Calibrator(
            mean=mean,
            projection=components[:keep].T,
            scale=(1.0 / np.sqrt(variance + 1e-6)).astype(np.float32),
        )
    raise ValueError(f"unknown calibration mode: {mode}")


@dataclass
class TemplateBank:
    words: list[str]
    pooled: np.ndarray
    takes: np.ndarray
    take_offsets: np.ndarray
    durations: dict[str, float]
    windows: dict[str, int] = field(default_factory=dict)
    index: dict[str, int] = field(default_factory=dict)
    prototype_vectors: np.ndarray | None = None
    prototype_offsets: np.ndarray | None = None

    def window_for(self, word: str) -> int:
        return self.windows.get(word) or int(
            np.clip(round(self.durations[word] / FRAME_S), 2, 12)
        )


def build_bank(
    rows: list[dict],
    spans: dict[str, np.ndarray],
    calibrator: Calibrator,
    fixed_window: int | None,
    kmeans: int = 0,
) -> TemplateBank | None:
    per_word: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    per_duration: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        clip = row["clip_id"]
        if clip not in spans:
            continue
        span = calibrator.apply(spans[clip])
        per_word[row["word"]].append(l2(span.mean(axis=0)))
        per_duration[row["word"]].append(float(row["kw_t1"]) - float(row["kw_t0"]))
    words = sorted(word for word, takes in per_word.items() if takes)
    if not words:
        return None
    take_rows, offsets = [], []
    for word in words:
        offsets.append(len(take_rows))
        take_rows.extend(per_word[word])
    offsets.append(len(take_rows))
    takes = np.stack(take_rows).astype(np.float32)
    pooled = l2(
        np.stack(
            [takes[offsets[i] : offsets[i + 1]].mean(axis=0) for i in range(len(words))]
        )
    )
    durations = {word: float(np.median(per_duration[word])) for word in words}
    windows = {
        word: (fixed_window if fixed_window else int(np.clip(round(durations[word] / FRAME_S), 2, 12)))
        for word in words
    }
    prototype_vectors = None
    prototype_offsets = None
    if kmeans:
        prototype_rows = []
        prototype_offsets_list = [0]
        for word in words:
            centers = _kmeans_prototypes(np.stack(per_word[word]), int(kmeans))
            prototype_rows.extend(centers)
            prototype_offsets_list.append(len(prototype_rows))
        prototype_vectors = l2(np.stack(prototype_rows))
        prototype_offsets = np.asarray(prototype_offsets_list, dtype=np.int64)
    return TemplateBank(
        words=words,
        pooled=pooled,
        takes=takes,
        take_offsets=np.asarray(offsets, dtype=np.int64),
        durations=durations,
        windows=windows,
        index={word: position for position, word in enumerate(words)},
        prototype_vectors=prototype_vectors,
        prototype_offsets=prototype_offsets,
    )


def score_utterance(
    feature: np.ndarray,
    bank: TemplateBank,
    fixed_window: int | None,
    prototypes: bool,
    multi_scale: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = feature.shape[0]
    words_by_width: dict[int, list[int]] = collections.defaultdict(list)
    for position, word in enumerate(bank.words):
        width = fixed_window if fixed_window else bank.window_for(word)
        words_by_width[width].append(position)

    cache: dict[int, np.ndarray | None] = {}

    def normalized_windows(width: int) -> np.ndarray | None:
        if width not in cache:
            means = window_means(feature, width)
            cache[width] = None if means is None else l2(means)
        return cache[width]

    best = np.full(len(bank.words), -np.inf, dtype=np.float32)
    starts = np.zeros(len(bank.words), dtype=np.int64)
    widths = np.zeros(len(bank.words), dtype=np.int64)
    for width, positions in words_by_width.items():
        candidates = [width - 1, width, width + 1] if multi_scale else [width]
        for candidate_width in candidates:
            candidate_width = max(2, candidate_width)
            windows = normalized_windows(candidate_width)
            if windows is None:
                continue
            index = np.asarray(positions, dtype=np.int64)
            if prototypes:
                prototype_matrix = (
                    bank.prototype_vectors
                    if bank.prototype_vectors is not None
                    else bank.takes
                )
                prototype_offsets = (
                    bank.prototype_offsets
                    if bank.prototype_offsets is not None
                    else bank.take_offsets
                )
                scores = np.maximum.reduceat(
                    windows @ prototype_matrix.T, prototype_offsets[:-1], axis=1
                )[:, index]
            else:
                scores = windows @ bank.pooled[index].T
            peaks = scores.max(axis=0)
            peak_start = scores.argmax(axis=0)
            better = peaks > best[index]
            best[index[better]] = peaks[better]
            starts[index[better]] = peak_start[better]
            widths[index[better]] = candidate_width
    return best, starts, widths


def overlap(pred_start: int, pred_end: int, true_start: int, true_end: int) -> tuple[float, float]:
    inter = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    true_len = max(1, true_end - true_start)
    pred_len = max(1, pred_end - pred_start)
    union = pred_len + true_len - inter
    return inter / float(true_len), inter / float(max(1, union))


CONFIGS = [
    {"name": "k5_mean", "fixed_window": 5},
    {"name": "k8_mean", "fixed_window": 8},
    {"name": "adapt_mean"},
    {"name": "adapt_ms_mean", "multi_scale": True},
    {"name": "adapt_ms_takes", "multi_scale": True, "prototypes": True},
    {"name": "adapt_ms_kmeans2", "multi_scale": True, "kmeans": 2},
    {"name": "adapt_ms_kmeans3", "multi_scale": True, "kmeans": 3},
]
CALIBRATIONS = ("none", "center", "abtt", "whiten")


def summarise(records: dict[str, dict]) -> dict:
    report = {}
    for key, record in records.items():
        if not record["top1"]:
            continue
        per_word = collections.defaultdict(list)
        for value, word in zip(record["top1"], record["words"]):
            per_word[word].append(value)
        report[key] = {
            "layer": record["layer"],
            "calibration": record["calibration"],
            "config": record["config"],
            "n_occurrences": len(record["top1"]),
            "top1": float(np.mean(record["top1"])),
            "top1_words": float(np.mean([np.mean(v) for v in per_word.values()])),
            "top5": float(np.mean(record["top5"])),
            "top20": float(np.mean(record["top20"])),
            "rank_median": float(np.median(record["rank"])),
            "coverage_mean": float(np.mean(record["coverage"])),
            "iou_median": float(np.median(record["iou"])),
            "coverage_gt0": float(np.mean(np.asarray(record["coverage"]) > 0)),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="v94 experiment directory")
    parser.add_argument("--output", default="reports/stage1_ablation.json")
    parser.add_argument("--layers", default="llm1,llm2,llm6")
    parser.add_argument("--calibrations", default="none,center,abtt,whiten")
    parser.add_argument("--configs", default="", help="comma separated config names")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    feature_dir = root / "feats" / "h2"
    templates = read_jsonl(root / "data" / "h2" / "templates.jsonl")
    search = read_jsonl(root / "data" / "h2" / "search.jsonl")
    if args.limit:
        search = search[: args.limit]
    layers = [value.strip() for value in args.layers.split(",") if value.strip()]
    calibrations = [value.strip() for value in args.calibrations.split(",") if value.strip()]
    if args.configs:
        wanted = {value.strip() for value in args.configs.split(",") if value.strip()}
        configs = [config for config in CONFIGS if config["name"] in wanted]
    else:
        configs = CONFIGS
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    for layer in layers:
        spans = {}
        for row in templates:
            path = feature_dir / "templates" / layer / f"{row['clip_id']}.npy"
            if not path.exists():
                continue
            feature = load_feature(path)
            start, end = span_frames(float(row["kw_t0"]), float(row["kw_t1"]), feature.shape[0])
            spans[row["clip_id"]] = feature[start:end]
        features = []
        for row in search:
            path = feature_dir / "search" / layer / f"{row['clip_id']}.npy"
            features.append(load_feature(path) if path.exists() else None)
        raw_takes = np.stack([span.mean(axis=0) for span in spans.values()])

        for calibration in calibrations:
            calibrator = fit_calibrator(raw_takes, calibration, 32)
            banks = {
                config["name"]: build_bank(
                    templates,
                    spans,
                    calibrator,
                    config.get("fixed_window"),
                    int(config.get("kmeans", 0)),
                )
                for config in configs
            }
            for config in configs:
                records[f"{layer}|{calibration}|{config['name']}"] = {
                    "layer": layer, "calibration": calibration, "config": config["name"],
                    "top1": [], "top5": [], "top20": [], "rank": [],
                    "coverage": [], "iou": [], "words": [],
                }
            for row, feature in zip(search, features):
                if feature is None:
                    continue
                calibrated = calibrator.apply(feature)
                frames = calibrated.shape[0]
                scored = {}
                for config in configs:
                    bank = banks[config["name"]]
                    if bank is None:
                        continue
                    scored[config["name"]] = (bank,) + score_utterance(
                        calibrated,
                        bank,
                        fixed_window=config.get("fixed_window"),
                        prototypes=bool(config.get("prototypes") or config.get("kmeans")),
                        multi_scale=bool(config.get("multi_scale")),
                    )
                for occ in row["occurrences"]:
                    word = occ["word"]
                    true_start, true_end = span_frames(float(occ["t0"]), float(occ["t1"]), frames)
                    for config in configs:
                        if config["name"] not in scored:
                            continue
                        bank, scores, starts, widths = scored[config["name"]]
                        column = bank.index.get(word)
                        if column is None:
                            continue
                        rank = 1 + int(np.count_nonzero(scores > scores[column]))
                        record = records[f"{layer}|{calibration}|{config['name']}"]
                        record["top1"].append(1.0 if rank == 1 else 0.0)
                        record["top5"].append(1.0 if rank <= 5 else 0.0)
                        record["top20"].append(1.0 if rank <= 20 else 0.0)
                        record["rank"].append(float(rank))
                        record["words"].append(word)
                        width = int(widths[column])
                        start = int(starts[column])
                        coverage, iou = overlap(start, start + width, true_start, true_end)
                        record["coverage"].append(coverage)
                        record["iou"].append(iou)
            print(f"done {layer} {calibration}", flush=True)
            destination.write_text(
                json.dumps(summarise(records), ensure_ascii=False, indent=2), encoding="utf-8"
            )

    report = summarise(records)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, row in sorted(report.items(), key=lambda item: -item[1]["top20"])[:15]:
        print(
            f"{key:42s} top1={row['top1']:.4f} words={row['top1_words']:.4f} "
            f"top20={row['top20']:.4f} cov={row['coverage_mean']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
