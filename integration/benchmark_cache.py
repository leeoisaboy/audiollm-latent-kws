from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integration.step_audio2 import StepAudio2Scorer, load_model_and_adapter


def parse_candidates(value: str) -> list[str]:
    candidates = [item.strip() for item in value.split(",") if item.strip()]
    if not candidates:
        raise ValueError("at least one candidate is required")
    return candidates


def run_once(scorer, wav_path: str, prefixes: list[str], candidates: list[str]):
    import torch

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = scorer.score("benchmark", wav_path, prefixes, candidates)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() - baseline
    return np.asarray(values, dtype=np.float64), elapsed, int(peak)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare cached and full-sequence Step-Audio2 scoring")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--prefix", default="请帮我")
    parser.add_argument("--candidates", default="开灯,关窗,拉帘,调温,打开电视,关闭风扇")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    candidates = parse_candidates(args.candidates)
    prefixes = [args.prefix] * len(candidates)
    model, tokenizer = load_model_and_adapter(args.model, args.adapter, args.device)
    cached = StepAudio2Scorer(model, tokenizer, args.device, use_kv_cache=True)
    full = StepAudio2Scorer(model, tokenizer, args.device, use_kv_cache=False)

    cached_score, cached_time, cached_memory = run_once(cached, args.wav, prefixes, candidates)
    full_score, full_time, full_memory = run_once(full, args.wav, prefixes, candidates)
    for _ in range(max(0, args.repeat - 1)):
        cached_score, cached_time, cached_memory = run_once(
            cached, args.wav, prefixes, candidates
        )
        full_score, full_time, full_memory = run_once(
            full, args.wav, prefixes, candidates
        )
    report = {
        "cached_scores": cached_score.tolist(),
        "full_scores": full_score.tolist(),
        "max_abs_diff": float(np.max(np.abs(cached_score - full_score))),
        "cached_seconds": cached_time,
        "full_seconds": full_time,
        "speedup": full_time / max(cached_time, 1e-9),
        "cached_peak_extra_bytes": cached_memory,
        "full_peak_extra_bytes": full_memory,
        "peak_memory_ratio": cached_memory / max(full_memory, 1),
        "cached_stats": cached.last_stats,
        "full_stats": full.last_stats,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
