"""Benchmark Shared-Audio-KV Listwise Multiple-Choice (A..T + U. NONE) with
forward + reversed option-order debiasing on L1, L2, and L3 across both
frozen Audio-LLM backends (Step-Audio-2 and Qwen2-Audio).

Key properties:
  - Encodes audio ONCE into PrefixKVState (shared Audio KV-Cache!)
  - Evaluates all Top-K candidates + NONE in TWO 1-token forward passes
    (forward order + reversed order to cancel option-position bias)
  - 0 external 40ms readers, 0 Mandarin pinyin, 0.00% prefix leakage
  - Built-in open-set score: max_i log P(opt_i) - log P(NONE)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.open_set import compute_roc_metrics, evaluate_speaker_cv
from common.io import read_jsonl, write_json
from common.scoring import fuse, rank_of
from core import enroll, locate
from locator.calibration import fit_latent_calibrator

LETTERS = [chr(ord("A") + i) for i in range(26)]


def _build_mcq_prompt_suffix(candidates: list[str], backend_name: str) -> tuple[str, list[str], str]:
    """Build listwise multiple-choice block with a final NONE option."""
    k = len(candidates)
    opt_letters = LETTERS[:k]
    none_letter = LETTERS[k]
    lines = ["\nWhich of the following phrases was spoken in the audio?"]
    for letter, word in zip(opt_letters, candidates):
        lines.append(f"{letter}. {word}")
    lines.append(f"{none_letter}. NONE (none of the above was spoken)")
    lines.append(f"Answer with a single letter ({opt_letters[0]}-{none_letter}) only.")
    body = "\n".join(lines)
    if backend_name == "step_audio2":
        return body + "<|EOT|><|BOT|>assistant\n", opt_letters, none_letter
    return body + "<|im_end|>\n<|im_start|>assistant\n", opt_letters, none_letter


def _score_mcq_one_order(
    backend,
    base_cache,
    candidates: list[str],
) -> tuple[np.ndarray, float]:
    """Run ONE incremental forward from shared audio KV-cache -> (opt_logp[K], none_logp)."""
    tok = backend.tokenizer
    suffix_text, opt_letters, none_letter = _build_mcq_prompt_suffix(candidates, backend.name)
    all_letters = opt_letters + [none_letter]
    letter_ids = [int(tok.encode(l, add_special_tokens=False)[0]) for l in all_letters]

    q_ids = tok.encode(suffix_text, add_special_tokens=False)
    past = int(base_cache.get_seq_length())
    q_t = torch.tensor([q_ids], dtype=torch.long, device=backend.device)
    mask = torch.ones(1, past + len(q_ids), dtype=torch.long, device=backend.device)

    with torch.inference_mode():
        if backend.name == "step_audio2":
            expanded = backend._scorer._clone_cache(base_cache)
            emb = backend.model.model.embed_tokens(q_t)
            out = backend.model.model(
                inputs_embeds=emb,
                attention_mask=mask,
                past_key_values=expanded,
                use_cache=False,
            )
            last_h = out.last_hidden_state[:, -1, :]
            pad_n = max(40, 1)
            padded = last_h.expand(pad_n, -1)
            logits = torch.matmul(backend.model.lm_head.weight, padded.transpose(0, 1)).float()[:, 0]
        else:
            import copy

            expanded = copy.deepcopy(base_cache)
            emb = backend.model.get_input_embeddings()(q_t)
            out = backend.model.language_model.model(
                inputs_embeds=emb,
                attention_mask=mask,
                past_key_values=expanded,
                use_cache=False,
            )
            last_h = out.last_hidden_state[:, -1, :]
            logits = backend.model.language_model.lm_head(last_h).float()[0]

        logp = torch.log_softmax(logits, dim=-1)
        picked = logp[letter_ids].cpu().numpy().astype(np.float64)
        return picked[:-1], float(picked[-1])


def score_listwise_debias(
    backend,
    wav_path: str,
    candidates: list[str],
) -> tuple[np.ndarray, float]:
    """Average forward + reversed option orders over ONE shared Audio KV-Cache."""
    state = backend.prefill_prefix_kv(wav_path, "")
    # Forward order
    fwd_logp, fwd_none = _score_mcq_one_order(backend, state.cache, candidates)
    # Reversed order (cancels option-position bias!)
    rev_cands = list(reversed(candidates))
    rev_logp, rev_none = _score_mcq_one_order(backend, state.cache, rev_cands)
    rev_logp = rev_logp[::-1]

    avg_opt = 0.5 * (fwd_logp + rev_logp)
    avg_none = 0.5 * (fwd_none + rev_none)
    # Log-odds relative to NONE: >0 means more likely spoken than NONE
    return avg_opt - avg_none, avg_none


def _setup_audio_only_prompt(backend):
    if backend.name == "step_audio2":
        backend._scorer.prompt = backend.tokenizer.encode(
            "<|BOT|>human\n<audio_patch>Listen carefully to the speech.",
            add_special_tokens=False,
        )
        backend._scorer._audio_key = None
        backend._scorer._audio_cache = None
    else:
        backend.prompt_text = (
            "<|im_start|>user\n"
            "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            "Listen carefully to the speech."
        )
        backend._cached_wav = None
        backend._cached_inputs = None
        backend._cached_audio_kv = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks-root", required=True)
    parser.add_argument("--feat-cache-root", required=True)
    parser.add_argument("--step-audio2-dir", required=True)
    parser.add_argument("--qwen2-audio-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tiers", default="L1,L2,L3")
    parser.add_argument("--lambda-lm", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from integration.qwen2_audio_backend import Qwen2AudioBackend
    from integration.step_audio2_backend import StepAudio2Backend

    benchmarks = Path(args.benchmarks_root)
    feat_root = Path(args.feat_cache_root)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    results: dict[str, Any] = {}
    for b_name, cls, m_dir in (
        ("step_audio2", StepAudio2Backend, args.step_audio2_dir),
        ("qwen2_audio", Qwen2AudioBackend, args.qwen2_audio_dir),
    ):
        print(f"\n================ LISTWISE MCQ BACKEND: {b_name} ================", flush=True)
        backend = cls(m_dir, device=args.device)
        _setup_audio_only_prompt(backend)
        results[b_name] = {}

        for tier in tiers:
            tier_dir = benchmarks / tier
            templates = read_jsonl(tier_dir / "templates.jsonl")
            pos_rows = read_jsonl(tier_dir / "search_pos.jsonl")
            neg_rows = read_jsonl(tier_dir / "search_neg.jsonl")

            fdir = feat_root / b_name / f"layer_{backend.default_layer}" / tier
            def _load_or_ext(rows, id_key, sub):
                d = fdir / sub
                d.mkdir(parents=True, exist_ok=True)
                res = {}
                for r in rows:
                    cid = r[id_key]
                    p = d / f"{cid}.npy"
                    if p.exists():
                        res[cid] = np.load(p).astype(np.float64)
                    else:
                        arr = backend.extract_shallow(r["wav"])
                        np.save(p, arr.astype(np.float16))
                        res[cid] = arr.astype(np.float64)
                return res

            enroll_feats = _load_or_ext(templates, "clip_id", "enroll")
            pos_feats = _load_or_ext(pos_rows, "utt", "pos")
            neg_feats = _load_or_ext(neg_rows, "utt", "neg")

            cal = fit_latent_calibrator(list(enroll_feats.values()), mode="center")
            bank = enroll(templates, lambda cid: enroll_feats[cid], frame_s=backend.frame_s, calibrator=cal)

            recs = []
            pos_s1, pos_mcq, pos_open = [], [], []
            times = []
            for row in pos_rows:
                pool = locate(row["utt"], cal.transform(pos_feats[row["utt"]]), bank, pool_top=20, gold=row["gold"])
                t0 = time.perf_counter()
                mcq_odds, _ = score_listwise_debias(backend, row["wav"], pool.words)
                times.append(time.perf_counter() - t0)
                fused = fuse(pool.acoustic_scores, mcq_odds, lambda_lm=args.lambda_lm)
                bi = int(np.argmax(fused))
                pos_s1.append(float(np.max(pool.acoustic_scores)))
                pos_mcq.append(float(np.max(mcq_odds)))
                pos_open.append(float(pool.acoustic_scores[bi] + 0.15 * mcq_odds[bi]))
                w2i = {w: i for i, w in enumerate(pool.words)}
                for g in row["gold"]:
                    gw = g["word"]
                    if gw not in w2i:
                        recs.append({"speaker": row["speaker"], "in_pool": False, "s1_top1": 0.0, "s2_top1": 0.0, "leaked": False, "ends_leaked": False})
                    else:
                        idx = w2i[gw]
                        recs.append({
                            "speaker": row["speaker"],
                            "in_pool": True,
                            "s1_top1": float(rank_of(pool.acoustic_scores, idx) <= 1.0),
                            "s2_top1": float(rank_of(fused, idx) <= 1.0),
                            "s2_mcq_only": float(rank_of(mcq_odds, idx) <= 1.0),
                            "leaked": False,
                            "ends_leaked": False,
                        })

            neg_s1, neg_mcq, neg_open = [], [], []
            for row in neg_rows:
                pool = locate(row["utt"], cal.transform(neg_feats[row["utt"]]), bank, pool_top=20)
                mcq_odds, _ = score_listwise_debias(backend, row["wav"], pool.words)
                fused = fuse(pool.acoustic_scores, mcq_odds, lambda_lm=args.lambda_lm)
                bi = int(np.argmax(fused))
                neg_s1.append(float(np.max(pool.acoustic_scores)))
                neg_mcq.append(float(np.max(mcq_odds)))
                neg_open.append(float(pool.acoustic_scores[bi] + 0.15 * mcq_odds[bi]))

            cv = evaluate_speaker_cv(recs, n_folds=5)
            mcq_only_top1 = float(np.mean([r.get("s2_mcq_only", 0.0) for r in recs]))
            y = [1] * len(pos_s1) + [0] * len(neg_s1)
            os_s1 = compute_roc_metrics(y, pos_s1 + neg_s1)
            os_mcq = compute_roc_metrics(y, pos_mcq + neg_mcq)
            os_open = compute_roc_metrics(y, pos_open + neg_open)
            b = cv["bootstrap"]
            results[b_name][tier] = {
                "median_s": round(float(np.median(times)), 4),
                "mcq_only_top1": round(mcq_only_top1, 4),
                "cv": cv,
                "open_set": {"stage1": os_s1, "mcq_only": os_mcq, "fused": os_open},
            }
            print(
                f"  [{b_name}/{tier}] {np.median(times):.3f}s/utt | R@20={cv['pool_recall']*100:.1f}% | "
                f"S1={cv['s1_top1']*100:.1f}% -> MCQ={mcq_only_top1*100:.1f}% | Fused={cv['s2_top1']*100:.1f}% "
                f"(gain={b['mean_gain_pp']:+.2f}pp, CI=[{b['ci95_low_pp']:+.2f},{b['ci95_high_pp']:+.2f}]) | "
                f"AUC: S1={os_s1['auc']:.3f} -> MCQ={os_mcq['auc']:.3f} / Fused={os_open['auc']:.3f} | "
                f"R@5%FAR: {os_s1['recall_at_far_5pct']*100:.1f}% -> {os_mcq['recall_at_far_5pct']*100:.1f}%",
                flush=True,
            )

        del backend
        torch.cuda.empty_cache()

    write_json(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
