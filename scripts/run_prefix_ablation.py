"""Subtractive ablation on L1 (Shanghainese) comparing:
  1. Arm C (external 40ms Whisper reader + magnetic character-grid snap)
  2. Arm C + Pinyin (external 40ms reader + Mandarin pinyin fuzzy snap)
  3. Acoustic Backoff (1 frame = 80ms, NO external reader, self-decoded prefix)
  4. Acoustic Backoff (2 frames = 160ms, NO external reader, self-decoded prefix)
  5. Empty Prefix (0ms, NO external reader, pure audio-conditioned teacher forcing)

This directly tests whether cutting the external 40ms reader + pinyin keeps
prefix leakage < 5% and Top-1 drop within 1-2 pp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.open_set import check_prefix_leak, evaluate_speaker_cv
from common.io import read_jsonl, write_json
from common.scoring import fuse, rank_of
from core import enroll, kv_rerank, locate, safe_prefix_cut
from integration.backend import BackendScorerAdapter
from locator.calibration import fit_latent_calibrator
from reranker.align import build_prefixes


def _generate_whisper_40ms_readings(
    pos_rows: list[dict[str, Any]],
    cache_path: Path,
    device: str = "cuda:0",
) -> dict[str, list[tuple[int, str]]]:
    """Generate 40ms character timestamps via Whisper to emulate the legacy external reader."""
    if cache_path.exists():
        readings: dict[str, list[tuple[int, str]]] = {}
        for row in read_jsonl(cache_path):
            readings[row["utt_id"]] = [(int(f), str(s)) for f, s in row["tokens"]]
        return readings

    import whisper

    model = whisper.load_model("tiny", device=device)
    readings = {}
    out_rows = []
    for row in pos_rows:
        res = model.transcribe(
            row["wav"], language="zh", fp16=True, word_timestamps=True, verbose=False
        )
        tokens: list[tuple[int, str]] = []
        for seg in res.get("segments", []):
            words = seg.get("words", [])
            if words:
                for w in words:
                    txt = str(w.get("word", "")).strip()
                    t0 = float(w.get("start", seg["start"]))
                    t1 = float(w.get("end", seg["end"]))
                    chars = [c for c in txt if not c.isspace()]
                    if not chars:
                        continue
                    dt = (t1 - t0) / max(len(chars), 1)
                    for ci, ch in enumerate(chars):
                        frame = int(round((t0 + ci * dt) / 0.04))
                        tokens.append((frame, ch))
            else:
                txt = str(seg.get("text", "")).strip()
                t0 = float(seg["start"])
                t1 = float(seg["end"])
                chars = [c for c in txt if not c.isspace()]
                dt = (t1 - t0) / max(len(chars), 1)
                for ci, ch in enumerate(chars):
                    frame = int(round((t0 + ci * dt) / 0.04))
                    tokens.append((frame, ch))
        readings[row["utt"]] = tokens
        out_rows.append({"utt_id": row["utt"], "frame_time": 0.04, "tokens": tokens})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        for r in out_rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    return readings


def main() -> int:
    parser = argparse.ArgumentParser(description="Subtractive prefix ablation on L1")
    parser.add_argument("--tier-dir", required=True)
    parser.add_argument("--feat-cache-root", required=True)
    parser.add_argument("--step-audio2-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lambda-lm", type=float, default=4.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from integration.step_audio2_backend import StepAudio2Backend

    tier_dir = Path(args.tier_dir)
    templates = read_jsonl(tier_dir / "templates.jsonl")
    pos_rows = read_jsonl(tier_dir / "search_pos.jsonl")

    readings_cache = Path(args.feat_cache_root) / "l1_whisper_40ms_readings.jsonl"
    readings = _generate_whisper_40ms_readings(pos_rows, readings_cache, device=args.device)

    backend = StepAudio2Backend(args.step_audio2_dir, device=args.device)
    scorer = BackendScorerAdapter(backend)

    feat_root = Path(args.feat_cache_root) / "step_audio2" / "layer_2" / "L1"
    enroll_feats = {}
    for row in templates:
        p = feat_root / "enroll" / f"{row['clip_id']}.npy"
        if p.exists():
            enroll_feats[row["clip_id"]] = np.load(p).astype(np.float64)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            arr = backend.extract_shallow(row["wav"])
            np.save(p, arr.astype(np.float16))
            enroll_feats[row["clip_id"]] = arr.astype(np.float64)

    pos_feats = {}
    for row in pos_rows:
        p = feat_root / "pos" / f"{row['utt']}.npy"
        if p.exists():
            pos_feats[row["utt"]] = np.load(p).astype(np.float64)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            arr = backend.extract_shallow(row["wav"])
            np.save(p, arr.astype(np.float16))
            pos_feats[row["utt"]] = arr.astype(np.float64)

    calibrator = fit_latent_calibrator(list(enroll_feats.values()), mode="center")
    bank = enroll(templates, lambda cid: enroll_feats[cid], frame_s=backend.frame_s, calibrator=calibrator)

    arms = ("arm_C_40ms", "arm_C_pinyin", "backoff_1f_80ms", "backoff_2f_160ms", "empty_prefix")
    arm_records: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}

    for row in pos_rows:
        utt = row["utt"]
        feats = calibrator.transform(pos_feats[utt])
        pool = locate(utt, feats, bank, pool_top=20, window=None, gold=row.get("gold"))
        word_to_idx = {w: i for i, w in enumerate(pool.words)}
        chars = readings.get(utt, [])

        # Arm C & Arm C + Pinyin (external 40ms reader)
        for arm_name, use_phonetic in (("arm_C_40ms", False), ("arm_C_pinyin", True)):
            prefixes = build_prefixes(
                pool.words,
                pool.starts.tolist(),
                chars,
                read_frame_s=0.04,
                frame_s=backend.frame_s,
                phonetic=use_phonetic,
            )
            lm_scores = scorer.score(utt, row["wav"], prefixes, pool.words)
            fused = fuse(pool.acoustic_scores, lm_scores, lambda_lm=args.lambda_lm)
            for g in row.get("gold", []):
                gw = str(g["word"])
                if gw not in word_to_idx:
                    arm_records[arm_name].append(
                        {"speaker": row["speaker"], "in_pool": False, "s1_top1": 0.0, "s2_top1": 0.0, "leaked": False, "ends_leaked": False}
                    )
                    continue
                idx = word_to_idx[gw]
                leaked, ends_leaked = check_prefix_leak(prefixes[idx], gw)
                arm_records[arm_name].append(
                    {
                        "speaker": row["speaker"],
                        "in_pool": True,
                        "s1_top1": float(rank_of(pool.acoustic_scores, idx) <= 1.0),
                        "s2_top1": float(rank_of(fused, idx) <= 1.0),
                        "leaked": bool(leaked),
                        "ends_leaked": bool(ends_leaked),
                    }
                )

        # Pure acoustic backoff (1f, 2f) and empty prefix (NO external 40ms reader!)
        for arm_name, mode, bf, self_pref in (
            ("backoff_1f_80ms", "acoustic_backoff", 1, True),
            ("backoff_2f_160ms", "acoustic_backoff", 2, True),
            ("empty_prefix", "empty", 0, False),
        ):
            reranked = kv_rerank(
                pool,
                wav_path=row["wav"],
                backend=backend,
                lambda_lm=args.lambda_lm,
                backoff_frames=bf,
                mode=mode,
                use_self_prefix=self_pref,
            )
            for g in row.get("gold", []):
                gw = str(g["word"])
                if gw not in word_to_idx:
                    arm_records[arm_name].append(
                        {"speaker": row["speaker"], "in_pool": False, "s1_top1": 0.0, "s2_top1": 0.0, "leaked": False, "ends_leaked": False}
                    )
                    continue
                idx = word_to_idx[gw]
                leaked, ends_leaked = check_prefix_leak(reranked.prefixes[idx], gw)
                arm_records[arm_name].append(
                    {
                        "speaker": row["speaker"],
                        "in_pool": True,
                        "s1_top1": float(rank_of(pool.acoustic_scores, idx) <= 1.0),
                        "s2_top1": float(rank_of(reranked.fused_scores, idx) <= 1.0),
                        "leaked": bool(leaked),
                        "ends_leaked": bool(ends_leaked),
                    }
                )

    report = {
        arm: evaluate_speaker_cv(recs, n_folds=5)
        for arm, recs in arm_records.items()
    }
    write_json(args.output, report)
    print(json.dumps({
        arm: {
            "pool_recall_pct": round(r["pool_recall"] * 100, 2),
            "s1_top1_pct": round(r["s1_top1"] * 100, 2),
            "s2_top1_pct": round(r["s2_top1"] * 100, 2),
            "gain_pp": round(r["bootstrap"]["mean_gain_pp"], 2),
            "ci95_low_pp": round(r["bootstrap"]["ci95_low_pp"], 2),
            "leak_pct": round(r["leak_rate"] * 100, 2),
        }
        for arm, r in report.items()
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
