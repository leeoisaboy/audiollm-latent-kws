#!/usr/bin/env python3
"""Real-corpus Stage-2 ablations: phonetic snapping, Top-2 starts and PMI.

This harness keeps the candidate pool, acoustic scores, fusion weight and
evaluation occurrences fixed.  It only changes the Stage-2 conditioning:

* exact grid snap (published arm C, recomputed under the cached scorer);
* Mandarin-pinyin fuzzy snap, which guards against homophone first characters;
* Top-2 NMS starts for the same candidate, resolved with max-over-starts;
* audio-conditional minus text-only log-probability (PMI calibration).

Audio is encoded once per utterance and all contexts are scored in one call,
so the comparisons share the same KV cache and audio representation.

Usage::

    python scripts/stage2_real_ablation.py \
        --root /path/to/final_ctc_latent_lm_kws \
        --features-root /path/to/v94_latent_mean_locator \
        --model /path/to/Step-Audio-2-mini \
        --adapter /path/to/stepaudio2-ASR \
        --wav-dir /path/to/audio_target \
        --report reports/stage2_real_ablation.json \
        --raw-output reports/stage2_real_lm.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.evaluation import occurrence_metrics
from common.scoring import fuse, rank_of, zpool
from locator import DirectoryLatentStore, build_templates, nms_starts, window_means
from locator.templates import l2_normalize
from reranker.align import build_contexts, load_reading


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def published_scores(rows: list[dict]) -> dict[str, dict]:
    """Convert ``word -> arm -> score`` rows to the scorer's arm layout."""
    return {
        row["utt"]: {
            "scores": {
                arm: {
                    word: float(arms[arm])
                    for word, arms in row.get("scores", {}).items()
                    if arm in arms
                }
                for arm in ("P", "C")
            }
        }
        for row in rows
    }


def gold_word(item) -> str:
    return str(item["word"] if isinstance(item, dict) else item[0])


def alternate_starts(
    feature: np.ndarray,
    words: list[str],
    primary: list[int],
    bank,
    count: int,
    radius: int,
) -> list[list[int]]:
    means = window_means(feature, 5)
    if means is None:
        return [[int(start)] for start in primary]
    normalized = l2_normalize(means)
    output = []
    for word, start in zip(words, primary):
        response = (normalized @ bank.prototypes_for(word).T).max(axis=1)
        starts = nms_starts(response, count=count, radius=radius)
        if int(start) not in starts:
            starts.insert(0, int(start))
        output.append(sorted(dict.fromkeys(starts), key=starts.index)[:count])
    return output


def build_request(
    row: dict,
    chars: list[tuple[int, str]],
    read_frame_s: float,
    feature: np.ndarray,
    bank,
    alt_count: int,
    alt_radius: int,
    pool_top: int,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    words = list(row["words"][:pool_top])
    starts = [int(value) for value in row["ps"][:pool_top]]
    exact = build_contexts(
        words,
        starts,
        chars,
        read_frame_s,
        variants=("P", "C"),
    )
    phonetic = build_contexts(
        words,
        starts,
        chars,
        read_frame_s,
        variants=("C",),
        phonetic=True,
    )
    alt_starts = alternate_starts(
        feature, words, starts, bank, count=alt_count, radius=alt_radius
    )
    expanded_words: list[str] = []
    expanded_starts: list[int] = []
    owners: list[int] = []
    for word_index, siblings in enumerate(alt_starts):
        for start in siblings:
            expanded_words.append(words[word_index])
            expanded_starts.append(int(start))
            owners.append(word_index)
    alternate = build_contexts(
        expanded_words,
        expanded_starts,
        chars,
        read_frame_s,
        variants=("C",),
    )["C"]

    prefixes = list(exact["P"]) + list(exact["C"]) + list(phonetic["C"]) + alternate
    candidates = words + words + words + expanded_words
    metadata = {
        "words": words,
        "ps_alt": alt_starts,
        "owners": owners,
        "prefix_P": exact["P"],
        "prefix_C": exact["C"],
        "prefix_C_phonetic": phonetic["C"],
    }
    return prefixes, candidates, metadata


def collect_metrics(
    pool_rows: list[dict],
    score_rows: dict[str, dict],
    key: str,
    calibration: str = "none",
    gamma: float = 1.0,
    lambda_lm: float = 1.5,
    pool_top: int = 20,
) -> tuple[dict, list[tuple[str, float]]]:
    correct_by_utt: dict[str, list[float]] = collections.defaultdict(list)
    for pool_row in pool_rows:
        row = score_rows.get(pool_row["utt"])
        if row is None:
            continue
        words = list(pool_row["words"][:pool_top])
        acoustic = np.asarray(pool_row["score"][:pool_top], dtype=np.float64)
        language_model = np.asarray(
            [row["scores"][key].get(word, np.nan) for word in words], dtype=np.float64
        )
        if calibration == "pmi":
            prior = np.asarray(
                [row.get("text_scores", {}).get(word, np.nan) for word in words],
                dtype=np.float64,
            )
            language_model = language_model - gamma * prior
        fused = fuse(acoustic, language_model, lambda_lm=lambda_lm)
        for item in pool_row.get("gold", []):
            word = gold_word(item)
            if word not in words:
                correct_by_utt[pool_row["utt"]].append(0.0)
                continue
            correct = float(rank_of(fused, words.index(word)) <= 1.0)
            correct_by_utt[pool_row["utt"]].append(correct)
    occurrences = [
        (utterance, value)
        for utterance, values in correct_by_utt.items()
        for value in values
    ]
    values = [value for _, value in occurrences]
    per_word: dict[str, list[float]] = collections.defaultdict(list)
    for pool_row in pool_rows:
        row = score_rows.get(pool_row["utt"])
        if row is None:
            continue
        words = list(pool_row["words"][:pool_top])
        acoustic = np.asarray(pool_row["score"][:pool_top], dtype=np.float64)
        language_model = np.asarray(
            [row["scores"][key].get(word, np.nan) for word in words], dtype=np.float64
        )
        if calibration == "pmi":
            prior = np.asarray(
                [row.get("text_scores", {}).get(word, np.nan) for word in words],
                dtype=np.float64,
            )
            language_model = language_model - gamma * prior
        fused = fuse(acoustic, language_model, lambda_lm=lambda_lm)
        for item in pool_row.get("gold", []):
            word = gold_word(item)
            per_word[word].append(
                float(word in words and rank_of(fused, words.index(word)) <= 1.0)
            )
    return {
        "n_occurrences": len(values),
        "top1": float(np.mean(values)) if values else 0.0,
        "top1_words": float(np.mean([np.mean(items) for items in per_word.values()]))
        if per_word
        else 0.0,
    }, occurrences


def paired_bootstrap(
    baseline: list[tuple[str, float]],
    candidate: list[tuple[str, float]],
    samples: int = 2000,
    seed: int = 20261027,
) -> dict[str, float]:
    by_utt_base: dict[str, list[float]] = collections.defaultdict(list)
    by_utt_cand: dict[str, list[float]] = collections.defaultdict(list)
    for (utt, value), (other_utt, other) in zip(baseline, candidate):
        if utt != other_utt:
            raise ValueError("paired bootstrap requires aligned occurrences")
        by_utt_base[utt].append(value)
        by_utt_cand[utt].append(other)
    utterances = sorted(by_utt_base)
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(samples):
        chosen = rng.choice(len(utterances), size=len(utterances), replace=True)
        base_values = [value for index in chosen for value in by_utt_base[utterances[index]]]
        cand_values = [value for index in chosen for value in by_utt_cand[utterances[index]]]
        gains.append(float(np.mean(cand_values) - np.mean(base_values)))
    return {
        "mean_gain_pts": 100.0
        * (float(np.mean([value for _, value in candidate])) - float(np.mean([value for _, value in baseline]))),
        "ci95_low_pts": 100.0 * float(np.percentile(gains, 2.5)),
        "ci95_high_pts": 100.0 * float(np.percentile(gains, 97.5)),
    }


def prefix_leak_rate(
    score_rows: dict[str, dict],
    pool_rows: list[dict],
    key: str,
) -> float:
    hits = total = 0
    for pool_row in pool_rows:
        row = score_rows.get(pool_row["utt"])
        if row is None:
            continue
        for word, prefix in zip(row["words"], row[key]):
            total += 1
            hits += int(bool(word) and word[0] in prefix)
    return hits / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-kv-cache", dest="kv_cache", action="store_false")
    parser.set_defaults(kv_cache=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pool-top", type=int, default=20)
    parser.add_argument("--alt-count", type=int, default=2)
    parser.add_argument("--alt-radius", type=int, default=2)
    parser.add_argument("--lambda-lm", type=float, default=1.5)
    parser.add_argument("--report", default="reports/stage2_real_ablation.json")
    parser.add_argument("--raw-output", default="reports/stage2_real_lm.jsonl")
    args = parser.parse_args()

    root = Path(args.root)
    features_root = Path(args.features_root)
    pool_rows = read_jsonl(root / "data" / "pool.jsonl")
    if args.limit:
        pool_rows = pool_rows[: args.limit]
    existing_rows = {
        row["utt"]: row for row in read_jsonl(root / "data" / "lm_all.jsonl")
    }
    readings, read_frame_s = load_reading(root / "data" / "read_u2pp.jsonl")
    templates = read_jsonl(features_root / "data" / "h2" / "templates.jsonl")
    bank = build_templates(
        templates,
        DirectoryLatentStore(features_root / "feats" / "h2" / "templates" / "llm2"),
    )

    raw_path = Path(args.raw_output)
    previous = read_jsonl(raw_path) if raw_path.exists() else []
    output_rows = {row["utt"]: row for row in previous}

    pending = [row for row in pool_rows if row["utt"] not in output_rows]
    if pending:
        from integration.step_audio2 import create_scorer

        scorer = create_scorer(
            model=args.model,
            adapter=args.adapter,
            device=args.device,
            kv_cache=args.kv_cache,
        )
    else:
        scorer = None
    for index, pool_row in enumerate(pool_rows, 1):
        utt = pool_row["utt"]
        if utt in output_rows:
            continue
        feature_path = (
            features_root / "feats" / "h2" / "search" / "llm2" / f"{utt}.npy"
        )
        feature = np.load(feature_path).astype(np.float32)
        prefixes, candidates, metadata = build_request(
            pool_row,
            readings.get(utt, []),
            read_frame_s,
            feature,
            bank,
            args.alt_count,
            args.alt_radius,
            args.pool_top,
        )
        words = metadata["words"]
        assert scorer is not None
        values = scorer.score(
            utterance=utt,
            wav_path=str(Path(args.wav_dir) / f"{utt}.wav"),
            prefixes=prefixes,
            candidates=candidates,
        )
        count = len(words)
        split = {
            "P": values[:count],
            "C": values[count : 2 * count],
            "C_phonetic": values[2 * count : 3 * count],
        }
        alternate = np.full(count, -np.inf, dtype=np.float64)
        for owner, value in zip(metadata["owners"], values[3 * count :]):
            alternate[owner] = max(alternate[owner], float(value))
        split["C_alt"] = alternate
        text_scores = (
            scorer.score_text_only(prefixes=metadata["prefix_C"], candidates=words)
            if args.kv_cache
            else {}
        )
        output_rows[utt] = {
            "utt": utt,
            "words": words,
            "scores": {key: dict(zip(words, map(float, value))) for key, value in split.items()},
            "text_scores": dict(zip(words, map(float, text_scores))),
            "prefix_C": metadata["prefix_C"],
            "prefix_C_phonetic": metadata["prefix_C_phonetic"],
            "ps_alt": metadata["ps_alt"],
            "stats": dict(scorer.last_stats),
        }
        if index % 25 == 0:
            write_jsonl(raw_path, list(output_rows.values()))
            print(
                json.dumps(
                    {"done": index, "total": len(pool_rows), "utt": utt},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    write_jsonl(raw_path, list(output_rows.values()))

    existing = published_scores(list(existing_rows.values()))
    metrics = {}
    occurrences = {}
    for key, rows, name in (
        ("existing_P", existing, "P"),
        ("existing_C", existing, "C"),
        ("C", output_rows, "C"),
        ("P", output_rows, "P"),
        ("C_phonetic", output_rows, "C_phonetic"),
        ("C_alt", output_rows, "C_alt"),
    ):
        metrics[key], occurrences[key] = collect_metrics(
            pool_rows,
            rows,
            name,
            lambda_lm=args.lambda_lm,
            pool_top=args.pool_top,
        )
    # PMI needs the text-only prior; without the cache path it is not computed
    # and the arm is omitted rather than silently collapsing to plain C.
    if args.kv_cache:
        for gamma in (0.25, 0.5, 0.75, 1.0, 1.5):
            key = f"C_pmi_gamma{gamma:g}"
            metrics[key], occurrences[key] = collect_metrics(
                pool_rows,
                output_rows,
                "C",
                calibration="pmi",
                gamma=gamma,
                lambda_lm=args.lambda_lm,
                pool_top=args.pool_top,
            )
    report = {
        "config": {
            "n_utterances": len(pool_rows),
            "pool_top": args.pool_top,
            "alt_count": args.alt_count,
            "alt_radius": args.alt_radius,
            "lambda_lm": args.lambda_lm,
            "device": args.device,
            "kv_cache": args.kv_cache,
            "pmi_available": args.kv_cache,
        },
        "metrics": metrics,
        "prefix_leak_rate": {
            "C": prefix_leak_rate(output_rows, pool_rows, "prefix_C"),
            "C_phonetic": prefix_leak_rate(output_rows, pool_rows, "prefix_C_phonetic"),
        },
        "bootstrap_vs_recomputed_C": {},
    }
    for key in ("C_phonetic", "C_alt", "C_pmi_gamma1"):
        if key not in occurrences:
            continue
        report["bootstrap_vs_recomputed_C"][key] = paired_bootstrap(
            occurrences["C"], occurrences[key]
        )
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
