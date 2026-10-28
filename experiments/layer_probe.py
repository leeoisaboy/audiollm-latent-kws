from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.io import read_jsonl
from common.config import FRAME_S
from experiments.legacy.layers import LayerViewStore, parse_fusion_specs
from locator import DirectoryLatentStore, build_pool, build_templates
from locator.templates import l2_normalize


def _pool_recall(pool_rows: Sequence[Mapping[str, Any]], pool_top: int) -> float:
    values = []
    for row in pool_rows:
        words = list(row["words"])[:pool_top]
        values.extend(1.0 if item["word"] in words else 0.0 for item in row.get("gold", []))
    return float(np.mean(values)) if values else 0.0


def _safe_pairwise_mean(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    return float(np.mean([float(vectors[left] @ vectors[right]) for left, right in combinations(range(len(vectors)), 2)]))


def template_discriminability(
    template_rows: Sequence[Mapping[str, Any]],
    enroll_dir: str | Path,
    latent_width: int = 3584,
) -> dict[str, float | int]:
    """Compare class-template cosine with within-class enrollment-template cosine."""
    store = DirectoryLatentStore(enroll_dir, expected_width=latent_width)
    return template_discriminability_from_store(template_rows, store)


def template_discriminability_from_store(
    template_rows: Sequence[Mapping[str, Any]],
    store,
) -> dict[str, float | int]:
    take_vectors: dict[str, list[np.ndarray]] = {}
    durations: dict[str, list[float]] = {}
    for row in template_rows:
        word = str(row["word"])
        try:
            features = store.get(str(row["clip_id"]))
        except (FileNotFoundError, OSError):
            continue
        start = int(np.floor(float(row["kw_t0"]) / FRAME_S))
        end = int(np.ceil(float(row["kw_t1"]) / FRAME_S))
        start, end = max(0, start), min(len(features), max(end, start + 1))
        if end <= start:
            continue
        take_vectors.setdefault(word, []).append(l2_normalize(features[start:end].mean(axis=0)))
        durations.setdefault(word, []).append(float(row["kw_t1"]) - float(row["kw_t0"]))
    if not take_vectors:
        raise ValueError("no usable enrollment features for template metrics")

    words = sorted(take_vectors)
    centroids = np.stack([l2_normalize(np.mean(take_vectors[word], axis=0)) for word in words])
    intra_values: list[float] = []
    for word, centroid in zip(words, centroids):
        intra_values.extend(float(vector @ centroid) for vector in take_vectors[word])
    inter = _safe_pairwise_mean(centroids)
    intra = float(np.mean(intra_values))
    return {
        "words": len(words),
        "takes": sum(len(values) for values in take_vectors.values()),
        "inter_template_cosine": inter,
        "intra_template_cosine": intra,
        "inter_over_intra": inter / max(intra, 1e-12),
        "margin_intra_minus_inter": intra - inter,
        "mean_duration_s": float(np.mean([value for values in durations.values() for value in values])) if durations else 0.0,
    }


def _stage1_for_store(template_rows, search_rows, enroll_store, search_store, pool_top):
    bank = build_templates(template_rows, enroll_store)
    pools = []
    for row in search_rows:
        features = search_store.get(str(row["utt"]))
        pool = build_pool(
            str(row["utt"]),
            features,
            bank,
            pool_top=pool_top,
            gold=row.get("gold"),
        )
        pools.append(pool.to_row(include_gold=True))
    return bank, stage1_metrics(pools, pool_top=pool_top)


def stage1_metrics(
    pool_rows: Sequence[Mapping[str, Any]],
    pool_top: int = 20,
) -> dict[str, float | int]:
    """Report pool recall and acquisition ranks without requiring Stage-2 scores."""
    ranks: list[float] = []
    occurrences = 0
    recovered = 0
    any_hit_utterances = 0
    complete_utterances = 0
    for row in pool_rows:
        words = list(row["words"])[:pool_top]
        gold = list(row.get("gold", []))
        occurrences += len(gold)
        utterance_hits = 0
        for item in gold:
            hit = int(item["word"] in words)
            recovered += hit
            utterance_hits += hit
            if hit:
                ranks.append(1.0 + words.index(item["word"]))
        any_hit_utterances += int(utterance_hits > 0)
        complete_utterances += int(utterance_hits == len(gold))
    return {
        "utterances": len(pool_rows),
        "occurrences": occurrences,
        "pool_recall": _pool_recall(pool_rows, pool_top),
        "candidate_pool_rank_median": float(np.median(ranks)) if ranks else 0.0,
        "utterance_any_hit": any_hit_utterances / max(len(pool_rows), 1),
        "utterance_all_hits": complete_utterances / max(len(pool_rows), 1),
    }


def compare_layers(
    layers: Sequence[int],
    features_root: str | Path,
    templates_path: str | Path,
    search_manifest_path: str | Path,
    latent_width: int = 3584,
    pool_top: int = 20,
    include_gold: bool = True,
    layer_pattern: str = "layer_{layer}",
) -> dict[str, Any]:
    """Build and score an independent Stage-1 pool for every requested layer."""
    template_rows = read_jsonl(templates_path)
    search_rows = read_jsonl(search_manifest_path)
    report: dict[str, Any] = {"pool_top": pool_top, "latent_width": latent_width, "layers": {}}
    for layer in layers:
        layer_dir = Path(features_root) / layer_pattern.format(layer=layer)
        enroll_dir = layer_dir / "enroll"
        search_dir = layer_dir / "search"
        metrics: dict[str, Any] = {
            "feature_root": str(layer_dir),
            "template_discriminability": template_discriminability(
                template_rows, enroll_dir, latent_width=latent_width
            ),
        }
        bank = build_templates(template_rows, DirectoryLatentStore(enroll_dir, expected_width=latent_width))
        pools = []
        for row in search_rows:
            features = np.load(search_dir / f"{row['utt']}.npy")
            pool = build_pool(
                str(row["utt"]),
                features,
                bank,
                pool_top=pool_top,
                gold=row.get("gold") if include_gold else None,
            )
            pools.append(pool.to_row(include_gold=include_gold))
        metrics["stage1"] = stage1_metrics(pools, pool_top=pool_top)
        report["layers"][str(layer)] = metrics

    ranked = sorted(
        (str(layer) for layer in layers),
        key=lambda layer: (
            -float(report["layers"][layer]["stage1"]["pool_recall"]),
            float(report["layers"][layer]["template_discriminability"]["inter_over_intra"]),
            int(layer),
        ),
    )
    report["ranking_by_pool_recall_then_discriminability"] = ranked
    report["best_layer"] = ranked[0] if ranked else None
    return report


def compare_layer_fusions(
    fusion_specs: str | Sequence[str],
    features_root: str | Path,
    templates_path: str | Path,
    search_manifest_path: str | Path,
    latent_width: int = 3584,
    pool_top: int = 20,
    layer_pattern: str = "layer_{layer}",
) -> dict[str, Any]:
    """Evaluate weighted and concatenated multi-layer Stage-1 feature views."""
    template_rows = read_jsonl(templates_path)
    search_rows = read_jsonl(search_manifest_path)
    specifications = parse_fusion_specs(fusion_specs)
    report: dict[str, Any] = {"pool_top": pool_top, "latent_width": latent_width, "views": {}}
    for specification in specifications:
        enroll_store = LayerViewStore(
            features_root,
            "enroll",
            specification,
            expected_layer_width=latent_width,
            layer_pattern=layer_pattern,
        )
        search_store = LayerViewStore(
            features_root,
            "search",
            specification,
            expected_layer_width=latent_width,
            layer_pattern=layer_pattern,
        )
        bank, stage1 = _stage1_for_store(
            template_rows, search_rows, enroll_store, search_store, pool_top
        )
        report["views"][specification.name] = {
            "layers": list(specification.layers),
            "weights": list(specification.weights),
            "concat": specification.concat,
            "view_width": int(bank.vectors.shape[1]),
            "template_discriminability": template_discriminability_from_store(
                template_rows, enroll_store
            ),
            "stage1": stage1,
        }
    ranked = sorted(
        report["views"],
        key=lambda name: (
            -float(report["views"][name]["stage1"]["pool_recall"]),
            float(report["views"][name]["template_discriminability"]["inter_over_intra"]),
            name,
        ),
    )
    report["ranking_by_pool_recall_then_discriminability"] = ranked
    report["best_view"] = ranked[0] if ranked else None
    return report
