#!/usr/bin/env python3
"""Benchmark full-sequence and prefix-KV Stage-2 scoring on identical WAVs.

Example::

    python scripts/benchmark_kv_cache.py \
        --model /path/to/Step-Audio-2-mini \
        --adapter /path/to/wu-adapter \
        --wav short=/path/short.wav --wav long=/path/long.wav \
        --candidates "开灯,关窗,打开电视,关闭电视" \
        --prefix 请帮我 --repetitions 5 \
        --output reports/cache_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def parse_name_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=/absolute/path.wav")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=/absolute/path.wav")
    return name, path


def measure(scorer, wav: str, prefixes: list[str], candidates: list[str], repetitions: int):
    import torch

    times = []
    peaks = []
    values = None
    for _ in range(repetitions):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        values = scorer.score("benchmark", wav, prefixes, candidates)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        peaks.append(int(torch.cuda.max_memory_allocated() - baseline))
    return np.asarray(values), statistics.median(times), max(peaks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wav", action="append", type=parse_name_path, required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", default="reports/cache_benchmark.json")
    args = parser.parse_args()

    from scipy.stats import spearmanr

    from integration.step_audio2 import StepAudio2Scorer, load_model_and_adapter

    candidates = [value.strip() for value in args.candidates.split(",") if value.strip()]
    if not candidates:
        raise ValueError("--candidates must contain at least one item")
    prefixes = [args.prefix] * len(candidates)
    model, tokenizer = load_model_and_adapter(args.model, args.adapter, args.device)
    report = {"candidates": candidates, "prefix": args.prefix, "results": {}}
    for name, wav in args.wav:
        values = {}
        for mode, use_cache in (("full", False), ("cached", True)):
            scorer = StepAudio2Scorer(
                model, tokenizer, args.device, use_kv_cache=use_cache
            )
            score_values, median_s, peak_bytes = measure(
                scorer, wav, prefixes, candidates, args.repetitions
            )
            scorer._audio_key = None
            scorer._audio_cache = None
            import torch

            torch.cuda.synchronize()
            started = time.perf_counter()
            scorer.score("cold", wav, prefixes, candidates)
            torch.cuda.synchronize()
            cold_s = time.perf_counter() - started
            values[mode] = score_values
            report["results"][f"{name}_{mode}"] = {
                "median_s": median_s,
                "cold_s": cold_s,
                "peak_extra_bytes": peak_bytes,
                "scores": score_values.tolist(),
            }
        full = values["full"]
        cached = values["cached"]
        result = report["results"][f"{name}_cached"]
        result["speedup_warm"] = (
            report["results"][f"{name}_full"]["median_s"] / result["median_s"]
        )
        result["memory_ratio"] = (
            result["peak_extra_bytes"]
            / report["results"][f"{name}_full"]["peak_extra_bytes"]
        )
        result["max_abs_diff"] = float(np.max(np.abs(cached - full)))
        result["spearman"] = float(spearmanr(cached, full).statistic)
        result["top1_identical"] = bool(np.argmax(cached) == np.argmax(full))
        result["top5_identical"] = bool(
            set(np.argsort(-cached)[:5]) == set(np.argsort(-full)[:5])
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = {
        key: {name: value for name, value in item.items() if name != "scores"}
        for key, item in report["results"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
