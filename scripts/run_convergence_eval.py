"""One-click public benchmark evaluation, subtractive ablation, ASR baseline,
KV-cache parity/speedup audit, and automated 6-gate Release Gate verification.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.gates import evaluate_release_gates
from analysis.open_set import (
    check_prefix_leak,
    compute_roc_metrics,
    evaluate_speaker_cv,
)
from common.io import read_jsonl, write_json
from common.scoring import fuse, rank_of
from core import enroll, kv_rerank, locate, safe_prefix_cut
from locator.calibration import fit_latent_calibrator


def _load_backend(name: str, model_dir: str, device: str):
    if name == "step_audio2":
        from integration.step_audio2_backend import StepAudio2Backend

        return StepAudio2Backend(model_dir=model_dir, device=device)
    if name == "qwen2_audio":
        from integration.qwen2_audio_backend import Qwen2AudioBackend

        return Qwen2AudioBackend(model_dir=model_dir, device=device)
    raise ValueError(f"unknown backend: {name}")


def _extract_or_load(
    backend,
    items: list[dict[str, Any]],
    id_key: str,
    feat_dir: Path,
    layer: int | None = None,
) -> dict[str, np.ndarray]:
    feat_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    for idx, row in enumerate(items, 1):
        cid = str(row[id_key])
        target = feat_dir / f"{cid}.npy"
        if target.exists():
            arr = np.load(target)
        else:
            arr = backend.extract_shallow(row["wav"], layer=layer)
            np.save(target, arr.astype(np.float16))
        out[cid] = arr.astype(np.float64)
    return out


def _benchmark_kv_cache(
    backend,
    pos_rows: list[dict[str, Any]],
    keywords: list[str],
    n_samples: int = 8,
) -> dict[str, Any]:
    import torch

    sample_rows = pos_rows[: min(n_samples, len(pos_rows))]
    cands = keywords[: min(20, len(keywords))]
    kv_all: list[float] = []
    full_all: list[float] = []
    t_kv_list: list[float] = []
    t_full_list: list[float] = []

    for i, row in enumerate(sample_rows):
        wav = row["wav"]
        pref = ""
        # Warmup on first sample
        if i == 0:
            st = backend.prefill_prefix_kv(wav, pref)
            _ = backend.score_suffixes(st, cands)
            _ = backend.score_full(wav, [pref] * len(cands), cands)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        st = backend.prefill_prefix_kv(wav, pref)
        s_kv = backend.score_suffixes(st, cands)
        torch.cuda.synchronize()
        t_kv = time.perf_counter() - t0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s_full = backend.score_full(wav, [pref] * len(cands), cands)
        torch.cuda.synchronize()
        t_full = time.perf_counter() - t0

        kv_all.extend(s_kv.tolist())
        full_all.extend(s_full.tolist())
        t_kv_list.append(t_kv)
        t_full_list.append(t_full)

    kv_arr = np.asarray(kv_all, dtype=np.float64)
    full_arr = np.asarray(full_all, dtype=np.float64)
    r_p = float(pearsonr(kv_arr, full_arr).statistic)
    r_s = float(spearmanr(kv_arr, full_arr).statistic)
    med_kv = float(np.median(t_kv_list))
    med_full = float(np.median(t_full_list))
    return {
        "n_utterances": len(sample_rows),
        "n_candidates": len(cands),
        "correlation_r": round(r_p, 6),
        "spearman_r": round(r_s, 6),
        "max_abs_diff": round(float(np.max(np.abs(kv_arr - full_arr))), 5),
        "median_kv_s": round(med_kv, 4),
        "median_full_s": round(med_full, 4),
        "speedup": round(med_full / max(med_kv, 1e-6), 3),
    }


def _evaluate_tier(
    tier: str,
    tier_dir: Path,
    backend,
    feat_cache_root: Path,
    pool_top: int = 20,
    lambda_lm: float = 4.0,
    backoff_frames: int = 2,
    mode: str = "acoustic_backoff",
    use_self_prefix: bool = True,
    open_beta: float = 0.25,
    layer: int | None = None,
) -> dict[str, Any]:
    templates = read_jsonl(tier_dir / "templates.jsonl")
    pos_rows = read_jsonl(tier_dir / "search_pos.jsonl")
    neg_rows = read_jsonl(tier_dir / "search_neg.jsonl")

    feat_root = feat_cache_root / backend.name / f"layer_{backend.default_layer if layer is None else layer}" / tier
    t0 = time.perf_counter()
    enroll_feats = _extract_or_load(backend, templates, "clip_id", feat_root / "enroll", layer=layer)
    pos_feats = _extract_or_load(backend, pos_rows, "utt", feat_root / "pos", layer=layer)
    neg_feats = _extract_or_load(backend, neg_rows, "utt", feat_root / "neg", layer=layer)
    t_extract = time.perf_counter() - t0

    # Fit enrollment mean-centering calibrator (zero test leakage)
    calibrator = fit_latent_calibrator(list(enroll_feats.values()), mode="center")

    # Step 1: enroll (auto window from median enrollment duration!)
    bank = enroll(
        templates,
        feature_getter=lambda cid: enroll_feats[cid],
        frame_s=backend.frame_s,
        calibrator=calibrator,
    )

    # Step 2-4 on positive queries
    cv_records: list[dict[str, Any]] = []
    pos_s1_open: list[float] = []
    pos_lm_open: list[float] = []
    pos_fused_open: list[float] = []
    rerank_times: list[float] = []

    for row in pos_rows:
        utt = row["utt"]
        feats = calibrator.transform(pos_feats[utt])
        pool = locate(utt, feats, bank, pool_top=pool_top, window=None, gold=row.get("gold"))

        tr0 = time.perf_counter()
        reranked = kv_rerank(
            pool,
            wav_path=row["wav"],
            backend=backend,
            lambda_lm=lambda_lm,
            backoff_frames=backoff_frames,
            open_beta=open_beta,
            mode=mode,
            use_self_prefix=use_self_prefix,
        )
        rerank_times.append(time.perf_counter() - tr0)

        # Open-set detection scores (top fused candidate)
        best_idx = int(reranked.order[0])
        pos_s1_open.append(float(np.max(pool.acoustic_scores)))
        pos_lm_open.append(float(reranked.lm_scores[best_idx]))
        pos_fused_open.append(float(reranked.open_scores[best_idx]))

        # Occurrence-level metrics
        word_to_idx = {w: i for i, w in enumerate(pool.words)}
        for g in row.get("gold", []):
            gw = str(g["word"])
            if gw not in word_to_idx:
                cv_records.append(
                    {
                        "utt": utt,
                        "speaker": row["speaker"],
                        "word": gw,
                        "in_pool": False,
                        "s1_top1": 0.0,
                        "s2_top1": 0.0,
                        "leaked": False,
                        "ends_leaked": False,
                    }
                )
                continue
            idx = word_to_idx[gw]
            s1_rank = rank_of(pool.acoustic_scores, idx)
            s2_rank = rank_of(reranked.fused_scores, idx)
            leaked, ends_leaked = check_prefix_leak(reranked.prefixes[idx], gw)
            cv_records.append(
                {
                    "utt": utt,
                    "speaker": row["speaker"],
                    "word": gw,
                    "in_pool": True,
                    "s1_top1": float(s1_rank <= 1.0),
                    "s2_top1": float(s2_rank <= 1.0),
                    "leaked": bool(leaked),
                    "ends_leaked": bool(ends_leaked),
                }
            )

    # Step 2-4 on negative queries (1:1 open-set evaluation)
    neg_s1_open: list[float] = []
    neg_lm_open: list[float] = []
    neg_fused_open: list[float] = []
    for row in neg_rows:
        utt = row["utt"]
        feats = calibrator.transform(neg_feats[utt])
        pool = locate(utt, feats, bank, pool_top=pool_top, window=None)
        reranked = kv_rerank(
            pool,
            wav_path=row["wav"],
            backend=backend,
            lambda_lm=lambda_lm,
            backoff_frames=backoff_frames,
            open_beta=open_beta,
            mode=mode,
            use_self_prefix=use_self_prefix,
        )
        best_idx = int(reranked.order[0])
        neg_s1_open.append(float(np.max(pool.acoustic_scores)))
        neg_lm_open.append(float(reranked.lm_scores[best_idx]))
        neg_fused_open.append(float(reranked.open_scores[best_idx]))

    y_true = [1] * len(pos_s1_open) + [0] * len(neg_s1_open)
    open_set = {
        "stage1": compute_roc_metrics(y_true, pos_s1_open + neg_s1_open),
        "stage2_lm": compute_roc_metrics(y_true, pos_lm_open + neg_lm_open),
        "fused": compute_roc_metrics(y_true, pos_fused_open + neg_fused_open),
    }

    cv_summary = evaluate_speaker_cv(cv_records, n_folds=5)
    return {
        "tier": tier,
        "auto_window_frames": bank.auto_window_frames,
        "median_enroll_duration_s": round(bank.median_duration_s, 4),
        "frame_s": backend.frame_s,
        "mode": mode,
        "backoff_frames": backoff_frames,
        "use_self_prefix": use_self_prefix,
        "lambda_lm": lambda_lm,
        "median_rerank_s": round(float(np.median(rerank_times)) if rerank_times else 0.0, 4),
        "extract_total_s": round(t_extract, 2),
        "cv": cv_summary,
        "open_set": open_set,
    }


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + int(ca != cb)))
        prev = curr
    return prev[-1]


def _best_fuzzy_score(transcript: str, keyword: str) -> float:
    """Best normalized substring similarity in [0, 1] between transcript and keyword."""
    t = transcript.strip().lower()
    k = keyword.strip().lower()
    if not t or not k:
        return 0.0
    if k in t:
        return 1.0
    lk = len(k)
    if len(t) <= lk:
        return max(0.0, 1.0 - _levenshtein(t, k) / max(len(t), lk))
    best = 0.0
    for i in range(0, len(t) - lk + 1):
        sub = t[i : i + lk]
        sim = 1.0 - _levenshtein(sub, k) / float(lk)
        if sim > best:
            best = sim
    return max(0.0, best)


def _run_asr_fuzzy_baseline(
    benchmarks_root: Path,
    tiers: list[str],
    whisper_model: str = "tiny",
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Strong baseline: full-utterance Whisper ASR + substring fuzzy match."""
    import whisper

    model = whisper.load_model(whisper_model, device=device)
    out: dict[str, Any] = {"asr_model": f"whisper-{whisper_model}", "tiers": {}}
    for tier in tiers:
        tier_dir = benchmarks_root / tier
        keywords = json.loads((tier_dir / "keywords.json").read_text(encoding="utf-8"))
        pos_rows = read_jsonl(tier_dir / "search_pos.jsonl")
        neg_rows = read_jsonl(tier_dir / "search_neg.jsonl")
        lang = "en" if tier == "L3" else "zh"

        pos_hits: list[float] = []
        pos_scores: list[float] = []
        neg_scores: list[float] = []
        latencies: list[float] = []

        for row in pos_rows:
            t0 = time.perf_counter()
            res = model.transcribe(row["wav"], language=lang, fp16=True, verbose=False)
            latencies.append(time.perf_counter() - t0)
            hyp = str(res.get("text", "")).strip()
            sims = np.array([_best_fuzzy_score(hyp, kw) for kw in keywords], dtype=np.float64)
            best_idx = int(np.argmax(sims))
            pos_scores.append(float(sims[best_idx]))
            gold_words = {g["word"] for g in row.get("gold", [])}
            pos_hits.append(1.0 if keywords[best_idx] in gold_words else 0.0)

        for row in neg_rows:
            res = model.transcribe(row["wav"], language=lang, fp16=True, verbose=False)
            hyp = str(res.get("text", "")).strip()
            sims = np.array([_best_fuzzy_score(hyp, kw) for kw in keywords], dtype=np.float64)
            neg_scores.append(float(np.max(sims)))

        y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
        out["tiers"][tier] = {
            "top1": round(float(np.mean(pos_hits)), 4),
            "median_latency_s": round(float(np.median(latencies)), 4),
            "open_set": compute_roc_metrics(y_true, pos_scores + neg_scores),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full convergence evaluation and release gates")
    parser.add_argument("--benchmarks-root", required=True)
    parser.add_argument("--feat-cache-root", required=True)
    parser.add_argument("--step-audio2-dir", required=True)
    parser.add_argument("--qwen2-audio-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tiers", default="L1,L2,L3")
    parser.add_argument("--mode", default="acoustic_backoff", choices=("acoustic_backoff", "empty"))
    parser.add_argument("--backoff-frames", type=int, default=2)
    parser.add_argument("--no-self-prefix", action="store_true")
    parser.add_argument("--lambda-lm", type=float, default=4.0)
    parser.add_argument("--open-beta", type=float, default=0.25)
    parser.add_argument("--skip-asr-baseline", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    benchmarks_root = Path(args.benchmarks_root)
    feat_cache_root = Path(args.feat_cache_root)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    backends_cfg = [
        ("step_audio2", args.step_audio2_dir),
        ("qwen2_audio", args.qwen2_audio_dir),
    ]

    backends_reports: dict[str, Any] = {}
    for b_name, m_dir in backends_cfg:
        print(f"\n================ BACKEND: {b_name} ================", flush=True)
        backend = _load_backend(b_name, m_dir, device=args.device)
        tier_reports: dict[str, Any] = {}
        for tier in tiers:
            print(f"--- Evaluating {b_name} on {tier} ---", flush=True)
            rep = _evaluate_tier(
                tier=tier,
                tier_dir=benchmarks_root / tier,
                backend=backend,
                feat_cache_root=feat_cache_root,
                lambda_lm=args.lambda_lm,
                backoff_frames=args.backoff_frames,
                mode=args.mode,
                use_self_prefix=not args.no_self_prefix,
                open_beta=args.open_beta,
            )
            tier_reports[tier] = rep
            cv = rep["cv"]
            os_f = rep["open_set"]["fused"]
            os_s1 = rep["open_set"]["stage1"]
            print(
                f"  [{b_name}/{tier}] win={rep['auto_window_frames']}f | "
                f"R@20={cv['pool_recall']*100:.1f}% | S1={cv['s1_top1']*100:.1f}% -> "
                f"S2={cv['s2_top1']*100:.1f}% (gain={cv['bootstrap']['mean_gain_pp']:+.2f}pp, "
                f"CI95=[{cv['bootstrap']['ci95_low_pp']:+.2f}, {cv['bootstrap']['ci95_high_pp']:+.2f}]) | "
                f"leak={cv['leak_rate']*100:.2f}% | "
                f"AUC: S1={os_s1['auc']:.3f} -> Fused={os_f['auc']:.3f} | "
                f"R@5%FAR: {os_s1['recall_at_far_5pct']*100:.1f}% -> {os_f['recall_at_far_5pct']*100:.1f}%",
                flush=True,
            )

        # KV-cache parity & speedup benchmark on L1
        l1_pos = read_jsonl(benchmarks_root / tiers[0] / "search_pos.jsonl")
        l1_kws = json.loads((benchmarks_root / tiers[0] / "keywords.json").read_text(encoding="utf-8"))
        kv_rep = _benchmark_kv_cache(backend, l1_pos, l1_kws, n_samples=8)
        print(
            f"  [{b_name}/KV] r={kv_rep['correlation_r']:.5f}, speedup={kv_rep['speedup']:.2f}x "
            f"(KV={kv_rep['median_kv_s']:.3f}s vs Full={kv_rep['median_full_s']:.3f}s)",
            flush=True,
        )
        backends_reports[b_name] = {
            "backend": b_name,
            "frame_s": backend.frame_s,
            "default_layer": backend.default_layer,
            "datasets": tier_reports,
            "kv_cache": kv_rep,
        }
        del backend
        import torch

        torch.cuda.empty_cache()

    asr_baseline = None
    if not args.skip_asr_baseline:
        print("\n================ BASELINE: Whisper ASR + Fuzzy ================", flush=True)
        asr_baseline = _run_asr_fuzzy_baseline(benchmarks_root, tiers, device=args.device)
        print(json.dumps(asr_baseline, ensure_ascii=False, indent=2), flush=True)

    gates = evaluate_release_gates(backends_reports, core_dir=REPO_ROOT / "core", tiers=tiers)
    final_report = {
        "config": {
            "tiers": tiers,
            "mode": args.mode,
            "backoff_frames": args.backoff_frames,
            "use_self_prefix": not args.no_self_prefix,
            "lambda_lm": args.lambda_lm,
            "open_beta": args.open_beta,
        },
        "backends": backends_reports,
        "asr_fuzzy_baseline": asr_baseline,
        "release_gates": gates,
    }
    write_json(args.output, final_report)
    print("\n================ RELEASE GATES SUMMARY ================", flush=True)
    print(json.dumps(gates, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
