"""Evaluate L1 Shanghainese with frozen Step-Audio-2-mini + WenetSpeech-Wu ASR LoRA adapter.

Runs ALL 4 methods under the upgraded metric suite (Utt-Hit@1, Utt-Hit@5, R@20,
Occ-Top1, Word mAP, Macro-F1, Micro-F1, Open-Set AUC, R@5%FAR):
  1. Stage-1 Mean (layer=2, Wu-ASR merged)
  2. Stage-1 3-Part Temporal Pyramid (layer=2, Wu-ASR merged)
  3. Shared-Audio-KV Listwise MCQ (Top-20 + NONE, forward + reversed debiasing)
  4. Shared-Audio-KV Pairwise 2-Audio A2A+Text (Top-5)
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
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.io import read_jsonl, write_json
from common.scoring import fuse, rank_of
from integration.step_audio2 import load_audio
from integration.step_audio2_backend import StepAudio2Backend
from locator.calibration import fit_latent_calibrator
from locator.templates import l2_normalize, span_frames
from scripts.test_audio_to_audio_l1 import _seg_vec, compute_upgraded_metrics
from scripts.test_listwise_mcq import _setup_audio_only_prompt, score_listwise_debias
from scripts.test_step_audio2_pairwise_a2a import (
    _crop_enroll_tensor,
    _encode_audio_block,
)


def _occ_metrics(pos_rows, score_maps, words):
    in_pool, top1 = [], []
    for row, smap in zip(pos_rows, score_maps):
        ranked = sorted(smap.items(), key=lambda kv: -kv[1])
        top20 = [w for w, _ in ranked[:20]]
        scores_arr = np.array([smap[w] for w in top20], dtype=np.float64)
        for g in row["gold"]:
            gw = g["word"]
            if gw not in top20:
                in_pool.append(0.0)
                top1.append(0.0)
            else:
                in_pool.append(1.0)
                top1.append(float(rank_of(scores_arr, top20.index(gw)) <= 1.0))
    return {"r20": round(float(np.mean(in_pool)), 4), "occ_top1": round(float(np.mean(top1)), 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier-dir", required=True)
    parser.add_argument("--feat-cache", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tdir = Path(args.tier_dir)
    fdir = Path(args.feat_cache)
    keywords = json.loads((tdir / "keywords.json").read_text(encoding="utf-8"))
    templates = read_jsonl(tdir / "templates.jsonl")
    pos_rows = read_jsonl(tdir / "search_pos.jsonl")
    neg_rows = read_jsonl(tdir / "search_neg.jsonl")
    all_rows = pos_rows + neg_rows

    print(f"Loading Step-Audio-2-mini + Wu-ASR adapter: {args.adapter_dir}", flush=True)
    backend = StepAudio2Backend(
        model_dir=args.model_dir,
        adapter_dir=args.adapter_dir,
        device=args.device,
    )

    # Extract/load layer-2 features from merged Wu-ASR model
    def _load_or_ext(rows, id_key, sub):
        d = fdir / sub
        d.mkdir(parents=True, exist_ok=True)
        out = {}
        for r in rows:
            cid = r[id_key]
            p = d / f"{cid}.npy"
            if p.exists():
                out[cid] = np.load(p).astype(np.float64)
            else:
                arr = backend.extract_shallow(r["wav"], layer=2)
                np.save(p, arr.astype(np.float16))
                out[cid] = arr.astype(np.float64)
        return out

    t0 = time.perf_counter()
    ef = _load_or_ext(templates, "clip_id", "enroll")
    pf = _load_or_ext(pos_rows, "utt", "pos")
    nf = _load_or_ext(neg_rows, "utt", "neg")
    all_f = {**pf, **nf}
    print(f"Extracted/loaded Wu-ASR layer=2 features in {time.perf_counter()-t0:.1f}s", flush=True)

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
        score_maps, open_s = [], []
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
    for tag, maps, op in (("stage1_mean", s1_mean_maps, s1_mean_open), ("stage1_pyramid3", s1_pyr_maps, s1_pyr_open)):
        m = compute_upgraded_metrics(all_rows, maps, keywords, op)
        m.update(_occ_metrics(pos_rows, maps[: len(pos_rows)], words))
        report[tag] = m
        print(f"{tag:18s}: {json.dumps(m)}", flush=True)

    # Method 3: Shared-Audio-KV Listwise MCQ (Top-20 + NONE, forward + reversed)
    _setup_audio_only_prompt(backend)
    t0 = time.perf_counter()
    mcq_raw_maps, mcq_raw_open = [], []
    mcq_fused_maps: dict[float, list[dict[str, float]]] = {1.0: [], 1.5: [], 3.0: []}
    mcq_fused_open: dict[float, list[float]] = {1.0: [], 1.5: [], 3.0: []}

    for row, s1_map in zip(all_rows, s1_pyr_maps):
        ranked = sorted(s1_map.items(), key=lambda kv: -kv[1])
        top20 = [w for w, _ in ranked[:20]]
        top_s1 = np.array([v for _, v in ranked[:20]], dtype=np.float64)
        mcq_odds, _ = score_listwise_debias(backend, row["wav"], top20)

        rmap = {w: float(s1_map[w] - 100.0) for w in words}
        for w, v in zip(top20, mcq_odds):
            rmap[w] = float(v)
        mcq_raw_maps.append(rmap)
        mcq_raw_open.append(float(np.max(mcq_odds)))

        for lam in (1.0, 1.5, 3.0):
            f_sc = fuse(top_s1, mcq_odds, lambda_lm=lam)
            fmap = {w: float(s1_map[w] - 100.0) for w in words}
            for w, v in zip(top20, f_sc):
                fmap[w] = float(v)
            mcq_fused_maps[lam].append(fmap)
            bi = int(np.argmax(f_sc))
            mcq_fused_open[lam].append(float(top_s1[bi] + 0.15 * mcq_odds[bi]))

    t_mcq = (time.perf_counter() - t0) / len(all_rows)
    m_mcq_raw = compute_upgraded_metrics(all_rows, mcq_raw_maps, keywords, mcq_raw_open)
    m_mcq_raw.update(_occ_metrics(pos_rows, mcq_raw_maps[: len(pos_rows)], words))
    report["listwise_mcq"] = {"s_per_utt": round(t_mcq, 4), "raw": m_mcq_raw, "fused": {}}
    print(f"\nListwise MCQ ({t_mcq:.3f}s/utt):", flush=True)
    print(f"  raw:        {json.dumps(m_mcq_raw)}", flush=True)
    for lam in (1.0, 1.5, 3.0):
        mf = compute_upgraded_metrics(all_rows, mcq_fused_maps[lam], keywords, mcq_fused_open[lam])
        mf.update(_occ_metrics(pos_rows, mcq_fused_maps[lam][: len(pos_rows)], words))
        report["listwise_mcq"]["fused"][str(lam)] = mf
        print(f"  fused {lam}:  {json.dumps(mf)}", flush=True)

    # Method 4: Pairwise 2-Audio A2A+Text (Top-5, shared Query Audio KV-Cache)
    tok = backend.tokenizer
    shi_id = int(tok.encode("是", add_special_tokens=False)[0])
    fou_id = int(tok.encode("否", add_special_tokens=False)[0])

    enroll_blocks: dict[str, torch.Tensor] = {}
    by_kw_takes: dict[str, list[dict]] = defaultdict(list)
    for r in templates:
        by_kw_takes[r["word"]].append(r)
    for kw, takes in by_kw_takes.items():
        takes_sorted = sorted(takes, key=lambda x: float(x["kw_t1"]) - float(x["kw_t0"]))
        chosen = takes_sorted[len(takes_sorted) // 2]
        wav_t = _crop_enroll_tensor(chosen["wav"], float(chosen["kw_t0"]), float(chosen["kw_t1"]))
        enroll_blocks[kw] = _encode_audio_block(backend, wav_t)

    def _embed_text(text: str) -> torch.Tensor:
        ids = tok.encode(text, add_special_tokens=False)
        return backend.model.model.embed_tokens(torch.tensor([ids], dtype=torch.long, device=args.device))

    head_emb = _embed_text("<|BOT|>human\n第一段完整语音：")
    mid_emb = _embed_text("\n请仔细比对发音。下面第二段是目标词参考语音：")

    def _score_logodds(last_h: torch.Tensor) -> float:
        padded = last_h.reshape(1, -1).expand(40, -1)
        logits = torch.matmul(backend.model.lm_head.weight, padded.transpose(0, 1)).float()[:, 0]
        logp = torch.log_softmax(logits, dim=0)
        return float((logp[shi_id] - logp[fou_id]).item())

    t0 = time.perf_counter()
    a2a_raw_maps, a2a_raw_open = [], []
    a2a_fused_maps: dict[float, list[dict[str, float]]] = {0.5: [], 1.0: [], 1.5: []}
    a2a_fused_open: dict[float, list[float]] = {0.5: [], 1.0: [], 1.5: []}

    with torch.inference_mode():
        for row, s1_map in zip(all_rows, s1_pyr_maps):
            ranked = sorted(s1_map.items(), key=lambda kv: -kv[1])
            top5 = [w for w, _ in ranked[:5]]
            top_s1 = np.array([v for _, v in ranked[:5]], dtype=np.float64)

            q_wav = load_audio(row["wav"], max_duration=30.0)
            q_block = _encode_audio_block(backend, q_wav)
            prefix_emb = torch.cat([head_emb, q_block, mid_emb], dim=1)
            _, base_cache = backend._scorer._forward_hidden(
                inputs_embeds=prefix_emb,
                attention_mask=torch.ones(1, prefix_emb.shape[1], dtype=torch.long, device=args.device),
                use_cache=True,
            )
            past_len = int(base_cache.get_seq_length())

            a2a_sc = np.empty(5, dtype=np.float64)
            for ci, w in enumerate(top5):
                tail_e = _embed_text(f"\n第二段语音（参考词“{w}”）是否在第一段语音中出现过？请只回答是或否。<|EOT|><|BOT|>assistant\n")
                cand_emb = torch.cat([enroll_blocks[w], tail_e], dim=1)
                cloned = backend._scorer._clone_cache(base_cache)
                mask = torch.ones(1, past_len + cand_emb.shape[1], dtype=torch.long, device=args.device)
                hidden, _ = backend._scorer._forward_hidden(
                    inputs_embeds=cand_emb,
                    attention_mask=mask,
                    past_key_values=cloned,
                    use_cache=False,
                )
                a2a_sc[ci] = _score_logodds(hidden[:, -1, :])

            rmap = {w: float(s1_map[w] - 100.0) for w in words}
            for w, v in zip(top5, a2a_sc):
                rmap[w] = float(v)
            a2a_raw_maps.append(rmap)
            a2a_raw_open.append(float(np.max(a2a_sc)))

            for lam in (0.5, 1.0, 1.5):
                f_sc = fuse(top_s1, a2a_sc, lambda_lm=lam)
                fmap = {w: float(s1_map[w] - 100.0) for w in words}
                for w, v in zip(top5, f_sc):
                    fmap[w] = float(v)
                a2a_fused_maps[lam].append(fmap)
                bi = int(np.argmax(f_sc))
                a2a_fused_open[lam].append(float(top_s1[bi] + 0.15 * (a2a_sc[bi] - float(np.median(a2a_sc)))))

    t_a2a = (time.perf_counter() - t0) / len(all_rows)
    m_a2a_raw = compute_upgraded_metrics(all_rows, a2a_raw_maps, keywords, a2a_raw_open)
    m_a2a_raw.update(_occ_metrics(pos_rows, a2a_raw_maps[: len(pos_rows)], words))
    report["pairwise_a2a_plus_text"] = {"s_per_utt": round(t_a2a, 4), "raw": m_a2a_raw, "fused": {}}
    print(f"\nPairwise A2A+Text ({t_a2a:.3f}s/utt):", flush=True)
    print(f"  raw:        {json.dumps(m_a2a_raw)}", flush=True)
    for lam in (0.5, 1.0, 1.5):
        mf = compute_upgraded_metrics(all_rows, a2a_fused_maps[lam], keywords, a2a_fused_open[lam])
        mf.update(_occ_metrics(pos_rows, a2a_fused_maps[lam][: len(pos_rows)], words))
        report["pairwise_a2a_plus_text"]["fused"][str(lam)] = mf
        print(f"  fused {lam}:  {json.dumps(mf)}", flush=True)

    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
