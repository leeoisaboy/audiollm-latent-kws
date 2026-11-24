"""Step-Audio-2 Pairwise 2-Audio KV-Cache A2A Verification on L1 Shanghainese.

Architecture:
  1. Pre-encode all 40 keyword enrollment audio clips ONCE via Step-Audio-2's
     frozen encoder + adapter (< 1 second total).
  2. For each query utterance, encode Audio 1 (Query) ONCE into a reusable
     Prefix KV-Cache (``DynamicCache``).
  3. For each Stage-1 Top-K candidate (K=5), clone the Query KV-Cache, append
     ONLY 1 candidate audio block (Audio 2) + binary verification question,
     and read ``log P("是") - log P("否")`` (and ``log P("Yes") - log P("No")``).
  4. Compare Pure Pairwise A2A (0 text!) and Audio+Text Pairwise A2A against
     Stage-1 Mean & Stage-1 3-Part Temporal Pyramid using upgraded metrics:
     Utt-Hit@1, Utt-Hit@5, Word mAP, Macro-F1, Micro-F1, Open-Set AUC, R@5%FAR.
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

from common.io import read_jsonl, write_json
from common.scoring import fuse
from integration.step_audio2 import (
    AUDIO_END,
    AUDIO_PATCH,
    AUDIO_START,
    SAMPLE_RATE,
    audio_token_count,
    load_audio,
    log_mel_spectrogram,
)
from integration.step_audio2_backend import StepAudio2Backend
from locator.calibration import fit_latent_calibrator
from locator.templates import l2_normalize, span_frames
from scripts.test_audio_to_audio_l1 import _seg_vec, compute_upgraded_metrics


def _encode_audio_block(
    backend: StepAudio2Backend, audio_1d: torch.Tensor
) -> torch.Tensor:
    """Encode a 16kHz mono waveform into ``[<audio_start>, encoded_frames, <audio_end>]`` embeds."""
    device = backend.device
    audio = audio_1d.to(device)
    if audio.numel() < 8_000:
        pad = 8_000 - audio.numel()
        audio = torch.nn.functional.pad(audio, (pad // 2, pad - pad // 2))
    mel = log_mel_spectrogram(audio).to(device)
    slots = audio_token_count(mel.shape[1])
    encoder_dtype = next(backend.model.encoder.parameters()).dtype
    adapter_dtype = next(backend.model.adapter.parameters()).dtype
    with torch.inference_mode():
        encoded, enc_len = backend.model.encoder(
            mel.unsqueeze(0).to(dtype=encoder_dtype),
            torch.tensor([mel.shape[1] - 2], dtype=torch.int32, device=device),
        )
        encoded = backend.model.adapter(encoded.to(adapter_dtype))
        upper = (int(enc_len[0].item()) - 1) // 2 + 1
        length = min(encoded.shape[1], max(upper, 1), max(slots, 1))
        emb_dtype = backend.model.model.embed_tokens.weight.dtype
        start_emb = backend.model.model.embed_tokens(
            torch.tensor([[AUDIO_START]], dtype=torch.long, device=device)
        )
        end_emb = backend.model.model.embed_tokens(
            torch.tensor([[AUDIO_END]], dtype=torch.long, device=device)
        )
        block = torch.cat(
            [start_emb, encoded[:, :length, :].to(emb_dtype), end_emb], dim=1
        )
        return block  # [1, length + 2, D]


def _crop_enroll_tensor(wav_path: str, t0: float, t1: float, pad_s: float = 0.05) -> torch.Tensor:
    audio = load_audio(wav_path, max_duration=30.0)
    s0 = max(0, int(round((t0 - pad_s) * SAMPLE_RATE)))
    s1 = min(audio.numel(), int(round((t1 + pad_s) * SAMPLE_RATE)))
    return audio[s0:s1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier-dir", required=True)
    parser.add_argument("--feat-dir", required=True)
    parser.add_argument("--step-audio2-dir", required=True)
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

    # Stage-1: 1-part mean vs 3-part temporal pyramid
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

    report: dict[str, Any] = {
        "stage1_mean": compute_upgraded_metrics(all_rows, s1_mean_maps, keywords, s1_mean_open),
        "stage1_pyramid3": compute_upgraded_metrics(all_rows, s1_pyr_maps, keywords, s1_pyr_open),
    }
    print("Stage-1 Mean:    ", json.dumps(report["stage1_mean"]), flush=True)
    print("Stage-1 Pyramid3:", json.dumps(report["stage1_pyramid3"]), flush=True)

    backend = StepAudio2Backend(args.step_audio2_dir, device=args.device)
    tok = backend.tokenizer
    shi_id = int(tok.encode("是", add_special_tokens=False)[0])
    fou_id = int(tok.encode("否", add_special_tokens=False)[0])
    print(f"Answer token ids: 是={shi_id}, 否={fou_id}", flush=True)

    # Pre-encode all 40 keyword enrollment audio blocks ONCE!
    t_enc = time.perf_counter()
    enroll_blocks: dict[str, torch.Tensor] = {}
    by_kw_takes: dict[str, list[dict]] = defaultdict(list)
    for r in templates:
        by_kw_takes[r["word"]].append(r)
    for kw, takes in by_kw_takes.items():
        takes_sorted = sorted(takes, key=lambda x: float(x["kw_t1"]) - float(x["kw_t0"]))
        chosen = takes_sorted[len(takes_sorted) // 2]
        wav_t = _crop_enroll_tensor(chosen["wav"], float(chosen["kw_t0"]), float(chosen["kw_t1"]))
        enroll_blocks[kw] = _encode_audio_block(backend, wav_t)
    print(f"Pre-encoded {len(enroll_blocks)} enrollment audio blocks in {time.perf_counter()-t_enc:.2f}s", flush=True)

    def _embed_text(text: str) -> torch.Tensor:
        ids = tok.encode(text, add_special_tokens=False)
        return backend.model.model.embed_tokens(
            torch.tensor([ids], dtype=torch.long, device=args.device)
        )

    head_emb = _embed_text("<|BOT|>human\n第一段完整语音：")
    mid_emb = _embed_text("\n请仔细比对发音。下面第二段是目标词参考语音：")
    tail_pure_emb = _embed_text("\n第二段语音中的词是否在第一段语音中出现过？请只回答是或否。<|EOT|><|BOT|>assistant\n")

    def _score_logodds(last_h: torch.Tensor) -> float:
        pad_n = max(40, 1)
        padded = last_h.reshape(1, -1).expand(pad_n, -1)
        logits = torch.matmul(backend.model.lm_head.weight, padded.transpose(0, 1)).float()[:, 0]
        logp = torch.log_softmax(logits, dim=0)
        return float((logp[shi_id] - logp[fou_id]).item())

    k = int(args.top_k)

    for mode_name, include_text in (("step2_pairwise_pure_a2a", False), ("step2_pairwise_a2a_plus_text", True)):
        t0 = time.perf_counter()
        raw_maps, raw_open = [], []
        fused_maps: dict[float, list[dict[str, float]]] = {0.5: [], 1.0: [], 1.5: []}
        fused_open: dict[float, list[float]] = {0.5: [], 1.0: [], 1.5: []}

        with torch.inference_mode():
            for row, s1_map in zip(all_rows, s1_pyr_maps):
                ranked = sorted(s1_map.items(), key=lambda kv: -kv[1])
                top_words = [w for w, _ in ranked[:k]]
                top_s1 = np.array([v for _, v in ranked[:k]], dtype=np.float64)

                # Step A: Prefill Query Audio 1 ONCE into Prefix KV-Cache!
                q_wav = load_audio(row["wav"], max_duration=30.0)
                q_block = _encode_audio_block(backend, q_wav)
                prefix_emb = torch.cat([head_emb, q_block, mid_emb], dim=1)
                _, base_cache = backend._scorer._forward_hidden(
                    inputs_embeds=prefix_emb,
                    attention_mask=torch.ones(1, prefix_emb.shape[1], dtype=torch.long, device=args.device),
                    use_cache=True,
                )
                past_len = int(base_cache.get_seq_length())

                # Step B: Score each Top-K candidate (Audio 2) reusing Query KV-Cache!
                a2a_scores = np.empty(k, dtype=np.float64)
                for ci, w in enumerate(top_words):
                    c_block = enroll_blocks[w]
                    if include_text:
                        tail_e = _embed_text(f"\n第二段语音（参考词“{w}”）是否在第一段语音中出现过？请只回答是或否。<|EOT|><|BOT|>assistant\n")
                    else:
                        tail_e = tail_pure_emb
                    cand_emb = torch.cat([c_block, tail_e], dim=1)
                    cloned = backend._scorer._clone_cache(base_cache)
                    mask = torch.ones(1, past_len + cand_emb.shape[1], dtype=torch.long, device=args.device)
                    hidden, _ = backend._scorer._forward_hidden(
                        inputs_embeds=cand_emb,
                        attention_mask=mask,
                        past_key_values=cloned,
                        use_cache=False,
                    )
                    a2a_scores[ci] = _score_logodds(hidden[:, -1, :])

                rmap = {w: float(s1_map[w] - 100.0) for w in words}
                for w, v in zip(top_words, a2a_scores):
                    rmap[w] = float(v)
                raw_maps.append(rmap)
                raw_open.append(float(np.max(a2a_scores)))

                for lam in (0.5, 1.0, 1.5):
                    f_sc = fuse(top_s1, a2a_scores, lambda_lm=lam)
                    fmap = {w: float(s1_map[w] - 100.0) for w in words}
                    for w, v in zip(top_words, f_sc):
                        fmap[w] = float(v)
                    fused_maps[lam].append(fmap)
                    bi = int(np.argmax(f_sc))
                    fused_open[lam].append(float(top_s1[bi] + 0.15 * (a2a_scores[bi] - float(np.median(a2a_scores)))))

        elapsed = time.perf_counter() - t0
        report[mode_name] = {
            "median_s_per_utt": round(elapsed / len(all_rows), 4),
            "raw_a2a": compute_upgraded_metrics(all_rows, raw_maps, keywords, raw_open),
            "fused_by_lam": {
                str(lam): compute_upgraded_metrics(all_rows, fused_maps[lam], keywords, fused_open[lam])
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
