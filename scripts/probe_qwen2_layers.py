"""Layer sweep for frozen Qwen2-Audio-7B-Instruct across L1, L2, and L3.

Extracts all candidate decoder layers (including ``layer=-1`` = multimodal
projector output before decoder layer 0) in ONE forward pass per WAV, then
reports Stage-1 Top-20 Pool Recall and Top-1 accuracy for every layer so the
optimal layer is chosen empirically rather than hardcoded to ``layer=2``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.io import read_jsonl, write_json
from common.scoring import rank_of
from core import enroll, locate
from integration.qwen2_audio_backend import Qwen2AudioBackend
from locator.calibration import fit_latent_calibrator


def extract_multi_layers(backend: Qwen2AudioBackend, wav_path: str, layers: list[int]) -> dict[int, np.ndarray]:
    inputs = backend._prepare_inputs(wav_path)
    with torch.inference_mode():
        outputs = backend.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )
        audio_mask = inputs["input_ids"][0] == backend.audio_token_id
        out: dict[int, np.ndarray] = {}
        for layer in layers:
            # layer = -1 -> hidden_states[0] (multi_modal_projector output)
            # layer = k >= 0 -> hidden_states[k + 1] (output of decoder block k)
            state_idx = layer + 1
            span = outputs.hidden_states[state_idx][0, audio_mask]
            out[layer] = span.detach().to(torch.float32).cpu().numpy()
        return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks-root", required=True)
    parser.add_argument("--qwen2-audio-dir", required=True)
    parser.add_argument("--layers", default="-1,0,2,4,8,12,16,20,24,28")
    parser.add_argument("--tiers", default="L1,L2,L3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]
    benchmarks = Path(args.benchmarks_root)

    backend = Qwen2AudioBackend(args.qwen2_audio_dir, device=args.device)
    report: dict[str, dict] = {}

    for tier in tiers:
        t0 = time.perf_counter()
        tdir = benchmarks / tier
        templates = read_jsonl(tdir / "templates.jsonl")
        pos_rows = read_jsonl(tdir / "search_pos.jsonl")

        enroll_by_layer: dict[int, dict[str, np.ndarray]] = {l: {} for l in layers}
        pos_by_layer: dict[int, dict[str, np.ndarray]] = {l: {} for l in layers}

        for row in templates:
            res = extract_multi_layers(backend, row["wav"], layers)
            for l, arr in res.items():
                enroll_by_layer[l][row["clip_id"]] = arr.astype(np.float64)

        for row in pos_rows:
            res = extract_multi_layers(backend, row["wav"], layers)
            for l, arr in res.items():
                pos_by_layer[l][row["utt"]] = arr.astype(np.float64)

        print(f"\n=== Qwen2-Audio Layer Sweep on {tier} ({time.perf_counter()-t0:.1f}s) ===", flush=True)
        tier_rep = {}
        for l in layers:
            ef = enroll_by_layer[l]
            pf = pos_by_layer[l]
            cal = fit_latent_calibrator(list(ef.values()), mode="center")
            bank = enroll(templates, lambda cid, _ef=ef: _ef[cid], frame_s=backend.frame_s, calibrator=cal)
            in_pool, top1 = [], []
            for row in pos_rows:
                pool = locate(row["utt"], cal.transform(pf[row["utt"]]), bank, pool_top=20, gold=row["gold"])
                w2i = {w: i for i, w in enumerate(pool.words)}
                for g in row["gold"]:
                    gw = g["word"]
                    if gw not in w2i:
                        in_pool.append(0.0)
                        top1.append(0.0)
                    else:
                        in_pool.append(1.0)
                        top1.append(float(rank_of(pool.acoustic_scores, w2i[gw]) <= 1.0))
            r20 = float(np.mean(in_pool))
            t1 = float(np.mean(top1))
            tier_rep[str(l)] = {"r20": round(r20, 4), "s1_top1": round(t1, 4), "window": bank.auto_window_frames}
            tag = "projector(-1)" if l == -1 else f"layer_{l:02d}"
            print(f"  {tag:14s} | win={bank.auto_window_frames:2d}f | R@20={r20*100:5.1f}% | S1 Top-1={t1*100:5.1f}%", flush=True)
        report[tier] = tier_rep

    # Compute mean R@20 across tiers per layer
    summary = {}
    for l in layers:
        vals = [report[t][str(l)]["r20"] for t in tiers]
        t1s = [report[t][str(l)]["s1_top1"] for t in tiers]
        summary[str(l)] = {"mean_r20": round(float(np.mean(vals)), 4), "mean_s1_top1": round(float(np.mean(t1s)), 4)}
    best_layer = max(layers, key=lambda l: (summary[str(l)]["mean_r20"], summary[str(l)]["mean_s1_top1"]))
    out = {"best_layer": best_layer, "summary": summary, "per_tier": report}
    write_json(args.output, out)
    print(f"\n>>> BEST QWEN2-AUDIO LAYER: {best_layer} (mean R@20={summary[str(best_layer)]['mean_r20']*100:.1f}%, mean S1 Top-1={summary[str(best_layer)]['mean_s1_top1']*100:.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
