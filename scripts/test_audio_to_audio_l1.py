"""Multimodal In-Context Audio-to-Audio (A2A) Verification on L1 Shanghainese.

Compares:
  1. Stage-1 Mean Cosine (Step-Audio-2 layer=2)
  2. Stage-1 3-Part Temporal Pyramid (order-preserving, zero extra GPU cost)
  3. Pure Audio-to-Audio (A2A) In-Context Verification (Qwen2-Audio, 1 Query + Top-5 Enroll Clips, ZERO text!)
  4. Multimodal Audio+Text (A2A+Text) In-Context Verification
  5. Fused (Stage-1 Pyramid + A2A)

Evaluates with upgraded multi-keyword metrics:
  - Utterance-level Hit@1 (Oracle ceiling = 100%)
  - Word-level mAP (mean Average Precision across all 40 keywords)
  - Word-level Macro-F1 & Micro-F1
  - Open-Set ROC-AUC & Recall@5%FAR (180 pos vs 180 neg)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.open_set import compute_roc_metrics
from common.io import read_jsonl, write_json
from common.scoring import fuse
from integration.qwen2_audio_backend import (
    QWEN2_SAMPLE_RATE,
    Qwen2AudioBackend,
    _load_mono_16k,
)
from locator.calibration import fit_latent_calibrator
from locator.templates import l2_normalize, span_frames

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _seg_vec(arr: np.ndarray, parts: int) -> np.ndarray:
    T, _ = arr.shape
    idxs = np.linspace(0, T, parts + 1).astype(int)
    chunks = []
    for i in range(parts):
        a = min(int(idxs[i]), T - 1)
        b = min(max(a + 1, int(idxs[i + 1])), T)
        chunks.append(l2_normalize(arr[a:b].mean(axis=0)))
    return l2_normalize(np.concatenate(chunks, axis=0))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    sorted_y = y_true[order]
    tp = np.cumsum(sorted_y)
    ranks = np.arange(1, len(sorted_y) + 1)
    precisions = tp / ranks
    return float(np.sum(precisions[sorted_y == 1]) / n_pos)


def _best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    sorted_y = y_true[order]
    tp = np.cumsum(sorted_y)
    fp = np.arange(1, len(sorted_y) + 1) - tp
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / float(n_pos)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(np.max(f1))


def compute_upgraded_metrics(
    all_rows: list[dict[str, Any]],
    score_maps: list[dict[str, float]],
    keywords: list[str],
    open_scores: list[float],
) -> dict[str, Any]:
    """Compute Utterance Hit@1, Top-5 Recall, Word-level mAP, Macro-F1, Micro-F1, and Open-Set AUC."""
    pos_hit1, pos_hit5 = [], []
    y_open_true = []
    for row, smap in zip(all_rows, score_maps):
        gold_set = {g["word"] for g in row.get("gold", [])}
        is_pos = len(gold_set) > 0
        y_open_true.append(1 if is_pos else 0)
        if is_pos:
            ranked = sorted(smap.items(), key=lambda kv: -kv[1])
            top1_w = ranked[0][0]
            top5_w = {w for w, _ in ranked[:5]}
            pos_hit1.append(1.0 if top1_w in gold_set else 0.0)
            pos_hit5.append(1.0 if bool(top5_w & gold_set) else 0.0)

    # Per-keyword AP and F1 across all (pos + neg) utterances
    per_kw_ap = []
    per_kw_f1 = []
    micro_y, micro_s = [], []
    for kw in keywords:
        yk = np.array([1 if kw in {g["word"] for g in r.get("gold", [])} else 0 for r in all_rows], dtype=np.int64)
        sk = np.array([smap.get(kw, -1e4) for smap in score_maps], dtype=np.float64)
        micro_y.append(yk)
        micro_s.append(sk)
        if int(np.sum(yk)) > 0:
            per_kw_ap.append(_average_precision(yk, sk))
            per_kw_f1.append(_best_f1(yk, sk))

    micro_y_all = np.concatenate(micro_y)
    micro_s_all = np.concatenate(micro_s)
    open_roc = compute_roc_metrics(y_open_true, open_scores)

    return {
        "utt_hit1": round(float(np.mean(pos_hit1)), 4),
        "utt_hit5": round(float(np.mean(pos_hit5)), 4),
        "word_mAP": round(float(np.mean(per_kw_ap)), 4),
        "word_macro_f1": round(float(np.mean(per_kw_f1)), 4),
        "word_micro_f1": round(_best_f1(micro_y_all, micro_s_all), 4),
        "open_auc": round(open_roc["auc"], 4),
        "open_r_at_5pct_far": round(open_roc["recall_at_far_5pct"], 4),
    }


def _crop_enroll_clip(wav_path: str, t0: float, t1: float, pad_s: float = 0.05) -> np.ndarray:
    audio = _load_mono_16k(wav_path, max_duration=30.0)
    s0 = max(0, int(round((t0 - pad_s) * QWEN2_SAMPLE_RATE)))
    s1 = min(len(audio), int(round((t1 + pad_s) * QWEN2_SAMPLE_RATE)))
    clip = audio[s0:s1]
    if len(clip) < int(0.4 * QWEN2_SAMPLE_RATE):
        pad = int(0.4 * QWEN2_SAMPLE_RATE) - len(clip)
        clip = np.pad(clip, (pad // 2, pad - pad // 2))
    return clip.astype(np.float32)


def _score_qwen_multiaudio_one_order(
    backend: Qwen2AudioBackend,
    query_audio: np.ndarray,
    cand_words: list[str],
    cand_audios: list[np.ndarray],
    include_text: bool = False,
) -> tuple[np.ndarray, float]:
    k = len(cand_words)
    opt_letters = LETTERS[:k]
    none_letter = LETTERS[k]
    all_letters = opt_letters + [none_letter]
    letter_ids = [int(backend.tokenizer.encode(l, add_special_tokens=False)[0]) for l in all_letters]

    lines = [
        "<|im_start|>user",
        "Audio 1 (Query utterance): <|audio_bos|><|AUDIO|><|audio_eos|>",
    ]
    for i, (letter, word) in enumerate(zip(opt_letters, cand_words), start=2):
        if include_text:
            lines.append(f'Audio {i} (Option {letter}, "{word}"): <|audio_bos|><|AUDIO|><|audio_eos|>')
        else:
            lines.append(f"Audio {i} (Option {letter}): <|audio_bos|><|AUDIO|><|audio_eos|>")
    lines.append(
        f"Listen to Audio 1 and compare the spoken sounds against Options {opt_letters[0]}-{opt_letters[-1]}."
    )
    lines.append(
        f"Which reference audio ({opt_letters[0]}-{opt_letters[-1]}) is spoken inside Audio 1? "
        f"If none of them is spoken in Audio 1, answer {none_letter}."
    )
    lines.append(f"Answer with a single letter ({opt_letters[0]}-{none_letter}) only.<|im_end|>")
    lines.append("<|im_start|>assistant\n")
    prompt = "\n".join(lines)

    audios = [query_audio] + list(cand_audios)
    inputs = backend.processor(
        text=prompt,
        audios=audios,
        sampling_rate=QWEN2_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: (
            val.to(backend.device, dtype=backend.compute_dtype)
            if key == "input_features"
            else val.to(backend.device)
        )
        for key, val in inputs.items()
    }
    with torch.inference_mode():
        out = backend.model(**inputs, use_cache=False)
        logits = out.logits[0, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        picked = logp[letter_ids].cpu().numpy().astype(np.float64)
        return picked[:-1], float(picked[-1])


def score_a2a_debias(
    backend: Qwen2AudioBackend,
    query_audio: np.ndarray,
    cand_words: list[str],
    enroll_audio_map: dict[str, np.ndarray],
    include_text: bool = False,
) -> tuple[np.ndarray, float]:
    cand_audios = [enroll_audio_map[w] for w in cand_words]
    fwd_opt, fwd_none = _score_qwen_multiaudio_one_order(
        backend, query_audio, cand_words, cand_audios, include_text=include_text
    )
    rev_words = list(reversed(cand_words))
    rev_audios = list(reversed(cand_audios))
    rev_opt, rev_none = _score_qwen_multiaudio_one_order(
        backend, query_audio, rev_words, rev_audios, include_text=include_text
    )
    rev_opt = rev_opt[::-1]
    avg_opt = 0.5 * (fwd_opt + rev_opt)
    avg_none = 0.5 * (fwd_none + rev_none)
    return avg_opt - avg_none, avg_none


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier-dir", required=True)
    parser.add_argument("--feat-dir", required=True)
    parser.add_argument("--qwen2-audio-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tdir = Path(args.tier_dir)
    fdir = Path(args.feat_dir)
    keywords = json.loads((tdir / "keywords.json").read_text(encoding="utf-8"))
    templates = read_jsonl(tdir / "templates.jsonl")
    pos_rows = read_jsonl(tdir / "search_pos.jsonl")
    neg_rows = read_jsonl(tdir / "search_neg.jsonl")
    all_rows = pos_rows + neg_rows

    # Build representative enrollment audio clip per keyword (longest/cleanest take)
    enroll_audio_map: dict[str, np.ndarray] = {}
    by_kw_takes: dict[str, list[dict]] = defaultdict(list)
    for r in templates:
        by_kw_takes[r["word"]].append(r)
    for kw, takes in by_kw_takes.items():
        # Pick take with median duration (avoids clipped outliers)
        takes_sorted = sorted(takes, key=lambda x: float(x["kw_t1"]) - float(x["kw_t0"]))
        chosen = takes_sorted[len(takes_sorted) // 2]
        enroll_audio_map[kw] = _crop_enroll_clip(chosen["wav"], float(chosen["kw_t0"]), float(chosen["kw_t1"]))

    # Stage-1: 1-part mean vs 3-part temporal pyramid on Step-Audio-2 layer=2
    ef = {r["clip_id"]: np.load(fdir / "enroll" / f"{r['clip_id']}.npy").astype(np.float64) for r in templates}
    pf = {r["utt"]: np.load(fdir / "pos" / f"{r['utt']}.npy").astype(np.float64) for r in pos_rows}
    nf = {r["utt"]: np.load(fdir / "neg" / f"{r['utt']}.npy").astype(np.float64) for r in neg_rows}
    all_f = {**pf, **nf}
    cal = fit_latent_calibrator(list(ef.values()), mode="center")

    def run_stage1(n_parts: int):
        take_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
        durs = []
        for r in templates:
            arr = cal.transform(ef[r["clip_id"]])
            s, e = span_frames(r["kw_t0"], r["kw_t1"], 0.08, len(arr))
            if e <= s:
                continue
            take_vecs[r["word"]].append(_seg_vec(arr[s:e], n_parts))
            durs.append(float(r["kw_t1"]) - float(r["kw_t0"]))
        words = sorted(take_vecs)
        mat = l2_normalize(np.stack([np.mean(take_vecs[w], axis=0) for w in words]))
        win = max(n_parts, int(round(np.median(durs) / 0.08)))
        score_maps = []
        open_s = []
        for r in all_rows:
            arr = cal.transform(all_f[r["utt"]])
            T = len(arr)
            if T < win:
                sc = np.full(len(words), -1.0)
            else:
                wins = np.stack([_seg_vec(arr[t : t + win], n_parts) for t in range(T - win + 1)])
                sc = (wins @ mat.T).max(axis=0)
            score_maps.append({w: float(v) for w, v in zip(words, sc)})
            open_s.append(float(np.max(sc)))
        return words, score_maps, open_s

    words, s1_mean_maps, s1_mean_open = run_stage1(1)
    _, s1_pyr_maps, s1_pyr_open = run_stage1(3)

    report: dict[str, Any] = {}
    report["stage1_mean"] = compute_upgraded_metrics(all_rows, s1_mean_maps, keywords, s1_mean_open)
    report["stage1_pyramid3"] = compute_upgraded_metrics(all_rows, s1_pyr_maps, keywords, s1_pyr_open)
    print("Stage-1 Mean:    ", json.dumps(report["stage1_mean"]), flush=True)
    print("Stage-1 Pyramid3:", json.dumps(report["stage1_pyramid3"]), flush=True)

    # Load Qwen2-Audio for Multi-Audio Audio-to-Audio In-Context Verification on Top-K
    backend = Qwen2AudioBackend(args.qwen2_audio_dir, device=args.device)
    k = int(args.top_k)

    for mode_name, inc_text in (("a2a_pure_audio", False), ("a2a_audio_plus_text", True)):
        t0 = time.perf_counter()
        raw_maps = []
        raw_open = []
        fused_maps_by_lam: dict[float, list[dict[str, float]]] = {0.5: [], 1.0: [], 1.5: []}
        fused_open_by_lam: dict[float, list[float]] = {0.5: [], 1.0: [], 1.5: []}

        for idx, (row, s1_map) in enumerate(zip(all_rows, s1_pyr_maps)):
            ranked = sorted(s1_map.items(), key=lambda kv: -kv[1])
            top_words = [w for w, _ in ranked[:k]]
            top_s1 = np.array([v for _, v in ranked[:k]], dtype=np.float64)
            q_audio = _load_mono_16k(row["wav"], max_duration=30.0)
            a2a_odds, _ = score_a2a_debias(backend, q_audio, top_words, enroll_audio_map, include_text=inc_text)

            # Build full-vocab score map (top-K get A2A scores, rest get -100 + s1)
            rmap = {w: float(s1_map[w] - 100.0) for w in words}
            for w, v in zip(top_words, a2a_odds):
                rmap[w] = float(v)
            raw_maps.append(rmap)
            raw_open.append(float(np.max(a2a_odds)))

            for lam in (0.5, 1.0, 1.5):
                f_scores = fuse(top_s1, a2a_odds, lambda_lm=lam)
                fmap = {w: float(s1_map[w] - 100.0) for w in words}
                for w, v in zip(top_words, f_scores):
                    fmap[w] = float(v)
                fused_maps_by_lam[lam].append(fmap)
                bi = int(np.argmax(f_scores))
                fused_open_by_lam[lam].append(float(top_s1[bi] + 0.15 * a2a_odds[bi]))

        elapsed = time.perf_counter() - t0
        report[mode_name] = {
            "median_s_per_utt": round(elapsed / len(all_rows), 4),
            "raw_a2a": compute_upgraded_metrics(all_rows, raw_maps, keywords, raw_open),
            "fused_by_lam": {
                str(lam): compute_upgraded_metrics(all_rows, fused_maps_by_lam[lam], keywords, fused_open_by_lam[lam])
                for lam in (0.5, 1.0, 1.5)
            },
        }
        print(f"\n=== {mode_name} ({elapsed/len(all_rows):.3f}s/utt) ===", flush=True)
        print("  raw_a2a:", json.dumps(report[mode_name]["raw_a2a"]), flush=True)
        for lam in (0.5, 1.0, 1.5):
            print(f"  fused lam={lam}:", json.dumps(report[mode_name]["fused_by_lam"][str(lam)]), flush=True)

    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
