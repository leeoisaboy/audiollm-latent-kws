"""Build standardized, speaker-disjoint L1 / L2 / L3 public KWS benchmarks.

Tiers:
  - L1 (Same dialect family - Wu/Shanghainese):
      TingChen-ppmc/Shanghai_Dialect_Conversational_Speech_Corpus
  - L2 (Cross-Chinese dialect):
      TwinkStart/KeSpeech (multi-dialect test shard)
  - L3 (Cross-lingual English + Open-Set Hard/Easy Negatives):
      charsiu/libriphrase (2-word phrase KWS with hard & easy negatives)

Every tier outputs the exact same directory schema:
  - ``keywords.json``      list of target keywords
  - ``templates.jsonl``    enrollment clips: {clip_id, word, speaker, wav, kw_t0, kw_t1, duration}
  - ``search_pos.jsonl``   positive queries: {utt, speaker, wav, text, duration, gold: [{word, kw_t0, kw_t1}]}
  - ``search_neg.jsonl``   1:1 negative queries: {utt, speaker, wav, text, duration, neg_type, gold: []}
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

TARGET_SR = 16_000


def _decode_audio_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return np.asarray(audio, dtype=np.float32), sr


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_zh_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff]", "", str(text or ""))


def _build_parquet_chinese_tier(
    tier_name: str,
    parquet_path: Path,
    out_dir: Path,
    text_col: str,
    spk_fn,
    n_keywords: int = 40,
    max_pos_utts: int = 180,
    max_takes_per_kw: int = 3,
    min_utterance_chars: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(parquet_path)
    meta_rows: list[dict[str, Any]] = []
    # Pass 1: read metadata only (fast, no audio decoding yet)
    cols = [c for c in pf.schema_arrow.names if c != "audio"]
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        d = batch.to_pydict()
        n = len(d[cols[0]])
        for i in range(n):
            raw_row = {c: d[c][i] for c in cols}
            clean = _clean_zh_text(raw_row[text_col])
            if len(clean) < min_utterance_chars:
                continue
            spk = str(spk_fn(raw_row))
            meta_rows.append(
                {
                    "row_idx": len(meta_rows),
                    "orig_idx": i,
                    "speaker": spk,
                    "text": clean,
                }
            )

    # Re-index with exact global row index in parquet
    meta_rows = []
    global_idx = 0
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        d = batch.to_pydict()
        n = len(d[cols[0]])
        for i in range(n):
            raw_row = {c: d[c][i] for c in cols}
            clean = _clean_zh_text(raw_row[text_col])
            if len(clean) >= min_utterance_chars:
                meta_rows.append(
                    {
                        "global_idx": global_idx,
                        "speaker": str(spk_fn(raw_row)),
                        "text": clean,
                    }
                )
            global_idx += 1

    # Find recurring 2-3 char phrases appearing in >= 5 utts and >= 3 distinct speakers
    phrase_utts: dict[str, list[int]] = defaultdict(list)
    phrase_spks: dict[str, set[str]] = defaultdict(set)
    # Exclude overly generic single-function filler bigrams
    stopwords = {
        "我们", "你们", "他们", "这个", "那个", "什么", "怎么", "就是", "不是",
        "可以", "没有", "一个", "知道", "觉得", "现在", "时候", "因为", "所以",
        "但是", "如果", "这样", "那样", "还是", "或者", "已经", "非常", "比较",
        "一点", "一些", "一下", "一样", "的话", "这里", "那里", "自己", "其实",
        "然后", "而且", "虽然", "不过", "只是", "应该", "可能", "需要", "开始",
        "阿拉", "伊拉", "格个", "哪能", "勿是", "晓得", "伊个", "个辰", "辰光",
    }
    for m_idx, item in enumerate(meta_rows):
        text = item["text"]
        spk = item["speaker"]
        seen_in_utt: set[str] = set()
        for n_len in (2, 3):
            # Prefer keywords not at char 0 so there is real left context!
            for pos in range(2, len(text) - n_len + 1):
                gram = text[pos : pos + n_len]
                if gram in stopwords or gram in seen_in_utt:
                    continue
                if len(set(gram)) == 1:
                    continue
                seen_in_utt.add(gram)
                phrase_utts[gram].append(m_idx)
                phrase_spks[gram].add(spk)

    candidates = [
        gram
        for gram, spks in phrase_spks.items()
        if len(spks) >= 3 and 5 <= len(phrase_utts[gram]) <= 120
    ]
    # Rank by speaker diversity then frequency, preferring non-overlapping grams
    candidates.sort(
        key=lambda g: (-len(phrase_spks[g]), -len(phrase_utts[g]), g)
    )
    selected_kws: list[str] = []
    for gram in candidates:
        if any(gram in prev or prev in gram for prev in selected_kws):
            continue
        selected_kws.append(gram)
        if len(selected_kws) >= n_keywords:
            break

    # Split speakers deterministically into enroll pool (~30%) and test pool (~70%)
    all_spks = sorted({item["speaker"] for item in meta_rows})
    rng.shuffle(all_spks)
    n_enroll_spks = max(2, int(round(0.30 * len(all_spks))))
    enroll_spks = set(all_spks[:n_enroll_spks])

    enroll_selection: list[tuple[int, str]] = []  # (m_idx, word)
    used_enroll_m: set[int] = set()
    usable_kws: list[str] = []
    for kw in selected_kws:
        e_utts = [
            m_idx for m_idx in phrase_utts[kw] if meta_rows[m_idx]["speaker"] in enroll_spks
        ]
        t_utts = [
            m_idx for m_idx in phrase_utts[kw] if meta_rows[m_idx]["speaker"] not in enroll_spks
        ]
        if not e_utts or not t_utts:
            continue
        usable_kws.append(kw)
        for m_idx in e_utts[:max_takes_per_kw]:
            enroll_selection.append((m_idx, kw))
            used_enroll_m.add(m_idx)

    # Collect positive test utterances (held-out speakers, containing >=1 usable_kw at pos >= 2)
    pos_candidates: list[tuple[int, list[str]]] = []
    kw_set = set(usable_kws)
    for m_idx, item in enumerate(meta_rows):
        if item["speaker"] in enroll_spks or m_idx in used_enroll_m:
            continue
        text = item["text"]
        hits = [kw for kw in usable_kws if kw in text[2:]]
        if hits:
            pos_candidates.append((m_idx, hits))

    rng.shuffle(pos_candidates)
    # Balance across keywords so no single keyword dominates
    kw_counts: Counter[str] = Counter()
    chosen_pos: list[tuple[int, list[str]]] = []
    for m_idx, hits in pos_candidates:
        if len(chosen_pos) >= max_pos_utts:
            break
        if all(kw_counts[h] >= 12 for h in hits):
            continue
        chosen_pos.append((m_idx, hits))
        for h in hits:
            kw_counts[h] += 1

    # Collect 1:1 negative test utterances (held-out speakers, containing NONE of usable_kws)
    neg_candidates = [
        m_idx
        for m_idx, item in enumerate(meta_rows)
        if item["speaker"] not in enroll_spks
        and m_idx not in used_enroll_m
        and not any(kw in item["text"] for kw in usable_kws)
    ]
    rng.shuffle(neg_candidates)
    chosen_neg = neg_candidates[: len(chosen_pos)]

    # Gather required global_idx set and extract audio in one streaming pass
    needed_global: set[int] = set()
    for m_idx, _ in enroll_selection:
        needed_global.add(meta_rows[m_idx]["global_idx"])
    for m_idx, _ in chosen_pos:
        needed_global.add(meta_rows[m_idx]["global_idx"])
    for m_idx in chosen_neg:
        needed_global.add(meta_rows[m_idx]["global_idx"])

    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    global_to_wav: dict[int, tuple[str, float]] = {}

    g_idx = 0
    for batch in pf.iter_batches(batch_size=256, columns=["audio"]):
        audios = batch.to_pydict()["audio"]
        for item in audios:
            if g_idx in needed_global:
                raw_bytes = item["bytes"]
                wav_arr, sr = _decode_audio_bytes(raw_bytes)
                dur = float(len(wav_arr) / sr)
                wav_path = wav_dir / f"{tier_name}_{g_idx:06d}.wav"
                if not wav_path.exists():
                    sf.write(wav_path, wav_arr, sr)
                global_to_wav[g_idx] = (str(wav_path), dur)
            g_idx += 1

    # Build manifests
    templates_rows: list[dict[str, Any]] = []
    for take_i, (m_idx, kw) in enumerate(enroll_selection):
        item = meta_rows[m_idx]
        wav_path, dur = global_to_wav[item["global_idx"]]
        text = item["text"]
        pos = text.find(kw, 2)
        if pos < 0:
            pos = text.find(kw)
        t0 = round((pos / len(text)) * dur, 4)
        t1 = round(((pos + len(kw)) / len(text)) * dur, 4)
        templates_rows.append(
            {
                "clip_id": f"{tier_name}_e{take_i:04d}_{item['global_idx']:06d}",
                "word": kw,
                "speaker": item["speaker"],
                "wav": wav_path,
                "text": text,
                "kw_t0": t0,
                "kw_t1": max(t0 + 0.16, t1),
                "duration": round(dur, 4),
            }
        )

    pos_rows: list[dict[str, Any]] = []
    for m_idx, hits in chosen_pos:
        item = meta_rows[m_idx]
        wav_path, dur = global_to_wav[item["global_idx"]]
        text = item["text"]
        gold = []
        for kw in hits:
            pos = text.find(kw, 2)
            if pos < 0:
                continue
            t0 = round((pos / len(text)) * dur, 4)
            t1 = round(((pos + len(kw)) / len(text)) * dur, 4)
            gold.append({"word": kw, "kw_t0": t0, "kw_t1": max(t0 + 0.16, t1)})
        if not gold:
            continue
        pos_rows.append(
            {
                "utt": f"{tier_name}_p_{item['global_idx']:06d}",
                "speaker": item["speaker"],
                "wav": wav_path,
                "text": text,
                "duration": round(dur, 4),
                "gold": gold,
            }
        )

    neg_rows: list[dict[str, Any]] = []
    for m_idx in chosen_neg:
        item = meta_rows[m_idx]
        wav_path, dur = global_to_wav[item["global_idx"]]
        neg_rows.append(
            {
                "utt": f"{tier_name}_n_{item['global_idx']:06d}",
                "speaker": item["speaker"],
                "wav": wav_path,
                "text": item["text"],
                "duration": round(dur, 4),
                "neg_type": "real_negative_utt",
                "gold": [],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "keywords.json").write_text(
        json.dumps(sorted(usable_kws), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(out_dir / "templates.jsonl", templates_rows)
    _write_jsonl(out_dir / "search_pos.jsonl", pos_rows)
    _write_jsonl(out_dir / "search_neg.jsonl", neg_rows)

    return {
        "tier": tier_name,
        "n_keywords": len(usable_kws),
        "n_templates": len(templates_rows),
        "n_pos": len(pos_rows),
        "n_neg": len(neg_rows),
        "n_test_speakers": len({r["speaker"] for r in pos_rows}),
    }


def _build_libriphrase_tier(
    zip_path: Path,
    csv_path: Path,
    out_dir: Path,
    n_keywords: int = 40,
    max_pos_utts: int = 180,
    max_takes_per_kw: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Build L3 English open-set KWS benchmark with left context + hard/easy negatives."""
    rng = np.random.default_rng(seed)

    # Index positive and negative pairs from 2-word LibriPhrase CSV
    phrase_anchors: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    phrase_pos_comps: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    spk_clips: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    hard_negs: list[tuple[str, str, str, float]] = []  # (path, spk, text, dur)
    easy_negs: list[tuple[str, str, str, float]] = []

    seen_anchor: set[tuple[str, str]] = set()
    seen_comp: set[tuple[str, str]] = set()
    seen_spk_clip: set[str] = set()

    with open(csv_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            a_path = row["anchor"]
            a_spk = row["anchor_spk"]
            a_txt = row["anchor_text"].strip().lower()
            a_dur = float(row["anchor_dur"])
            c_path = row["comparison"]
            c_spk = row["comparison_spk"]
            c_txt = row["comparison_text"].strip().lower()
            c_dur = float(row["comparison_dur"])
            rtype = row["type"]

            if (a_txt, a_path) not in seen_anchor:
                seen_anchor.add((a_txt, a_path))
                phrase_anchors[a_txt].append((a_path, a_spk, a_dur))
            if c_path not in seen_spk_clip:
                seen_spk_clip.add(c_path)
                spk_clips[c_spk].append((c_path, c_txt, c_dur))
            if rtype == "diffspk_positive":
                if (a_txt, c_path) not in seen_comp:
                    seen_comp.add((a_txt, c_path))
                    phrase_pos_comps[a_txt].append((c_path, c_spk, c_dur))
            elif rtype == "diffspk_hardneg":
                hard_negs.append((c_path, c_spk, c_txt, c_dur))
            elif rtype == "diffspk_easyneg":
                easy_negs.append((c_path, c_spk, c_txt, c_dur))

    # Pick n_keywords with rich speaker diversity
    kw_cands = [
        p
        for p, comps in phrase_pos_comps.items()
        if len(phrase_anchors[p]) >= 2 and len(comps) >= 4
    ]
    kw_cands.sort(key=lambda p: (-len(phrase_pos_comps[p]), p))
    usable_kws = sorted(kw_cands[:n_keywords])
    kw_set = set(usable_kws)

    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(zip_path, "r")
    silence = np.zeros(int(0.12 * TARGET_SR), dtype=np.float32)

    def _read_zip_wav(member: str) -> np.ndarray:
        raw = zf.read(member)
        arr, _ = _decode_audio_bytes(raw)
        return arr

    # Build enrollment templates (isolated clips from anchor speakers)
    templates_rows: list[dict[str, Any]] = []
    enroll_spks_by_kw: dict[str, set[str]] = defaultdict(set)
    for kw_idx, kw in enumerate(usable_kws):
        anchors = phrase_anchors[kw][:max_takes_per_kw]
        for take_i, (a_path, a_spk, a_dur) in enumerate(anchors):
            enroll_spks_by_kw[kw].add(a_spk)
            arr = _read_zip_wav(a_path)
            dur = float(len(arr) / TARGET_SR)
            cid = f"L3_e{kw_idx:03d}_{take_i}"
            out_wav = wav_dir / f"{cid}.wav"
            sf.write(out_wav, arr, TARGET_SR)
            templates_rows.append(
                {
                    "clip_id": cid,
                    "word": kw,
                    "speaker": a_spk,
                    "wav": str(out_wav),
                    "text": kw,
                    "kw_t0": 0.0,
                    "kw_t1": round(dur, 4),
                    "duration": round(dur, 4),
                }
            )

    # Helper to pick a non-target left-context clip from the SAME speaker
    def _pick_context(spk: str, exclude_txt: str) -> tuple[str, str]:
        pool = [
            (p, t)
            for p, t, _ in spk_clips.get(spk, [])
            if t != exclude_txt and t not in kw_set and exclude_txt.split()[0] not in t.split()
        ]
        if pool:
            return pool[int(rng.integers(0, len(pool)))]
        # Fallback: pick from any speaker
        for other_spk, items in spk_clips.items():
            for p, t, _ in items:
                if t != exclude_txt and t not in kw_set:
                    return p, t
        raise RuntimeError("no context clip available")

    # Build positive test utterances with real spoken left context
    pos_pool: list[tuple[str, str, str]] = []  # (kw, c_path, c_spk)
    for kw in usable_kws:
        for c_path, c_spk, _ in phrase_pos_comps[kw]:
            if c_spk not in enroll_spks_by_kw[kw]:
                pos_pool.append((kw, c_path, c_spk))
    rng.shuffle(pos_pool)

    pos_rows: list[dict[str, Any]] = []
    kw_counts: Counter[str] = Counter()
    for kw, c_path, c_spk in pos_pool:
        if len(pos_rows) >= max_pos_utts:
            break
        if kw_counts[kw] >= 6:
            continue
        ctx_path, ctx_txt = _pick_context(c_spk, kw)
        ctx_arr = _read_zip_wav(ctx_path)
        tgt_arr = _read_zip_wav(c_path)
        full_arr = np.concatenate([ctx_arr, silence, tgt_arr])
        kw_t0 = float((len(ctx_arr) + len(silence)) / TARGET_SR)
        kw_t1 = float(len(full_arr) / TARGET_SR)
        utt_id = f"L3_p_{len(pos_rows):04d}"
        out_wav = wav_dir / f"{utt_id}.wav"
        sf.write(out_wav, full_arr, TARGET_SR)
        pos_rows.append(
            {
                "utt": utt_id,
                "speaker": c_spk,
                "wav": str(out_wav),
                "text": f"{ctx_txt} {kw}",
                "duration": round(kw_t1, 4),
                "gold": [{"word": kw, "kw_t0": round(kw_t0, 4), "kw_t1": round(kw_t1, 4)}],
            }
        )
        kw_counts[kw] += 1

    # Build 1:1 negative test utterances (50% hardneg + 50% easyneg)
    n_neg = len(pos_rows)
    n_hard = n_neg // 2
    n_easy = n_neg - n_hard
    rng.shuffle(hard_negs)
    rng.shuffle(easy_negs)

    neg_rows: list[dict[str, Any]] = []
    used_neg_paths: set[str] = set()
    for neg_type, source, quota in (
        ("libriphrase_hardneg", hard_negs, n_hard),
        ("libriphrase_easyneg", easy_negs, n_easy),
    ):
        added = 0
        for c_path, c_spk, c_txt, _ in source:
            if added >= quota:
                break
            if c_path in used_neg_paths or c_txt in kw_set:
                continue
            used_neg_paths.add(c_path)
            ctx_path, ctx_txt = _pick_context(c_spk, c_txt)
            ctx_arr = _read_zip_wav(ctx_path)
            neg_arr = _read_zip_wav(c_path)
            full_arr = np.concatenate([ctx_arr, silence, neg_arr])
            dur = float(len(full_arr) / TARGET_SR)
            utt_id = f"L3_n_{len(neg_rows):04d}"
            out_wav = wav_dir / f"{utt_id}.wav"
            sf.write(out_wav, full_arr, TARGET_SR)
            neg_rows.append(
                {
                    "utt": utt_id,
                    "speaker": c_spk,
                    "wav": str(out_wav),
                    "text": f"{ctx_txt} {c_txt}",
                    "duration": round(dur, 4),
                    "neg_type": neg_type,
                    "gold": [],
                }
            )
            added += 1

    zf.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "keywords.json").write_text(
        json.dumps(usable_kws, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(out_dir / "templates.jsonl", templates_rows)
    _write_jsonl(out_dir / "search_pos.jsonl", pos_rows)
    _write_jsonl(out_dir / "search_neg.jsonl", neg_rows)

    return {
        "tier": "L3",
        "n_keywords": len(usable_kws),
        "n_templates": len(templates_rows),
        "n_pos": len(pos_rows),
        "n_neg": len(neg_rows),
        "n_test_speakers": len({r["speaker"] for r in pos_rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare standardized L1/L2/L3 public KWS benchmarks")
    parser.add_argument("--cache-root", required=True, help="root containing l1_shanghai, l2_kespeech, l3_libriphrase")
    parser.add_argument("--output-root", required=True, help="destination directory for L1, L2, L3 manifests & wavs")
    parser.add_argument("--n-keywords", type=int, default=40)
    parser.add_argument("--max-pos", type=int, default=180)
    args = parser.parse_args()

    cache = Path(args.cache_root)
    out = Path(args.output_root)
    summary = {}

    summary["L1"] = _build_parquet_chinese_tier(
        tier_name="L1",
        parquet_path=cache / "l1_shanghai" / "data" / "train-00000-of-00001-3b3c64c2b15eef11.parquet",
        out_dir=out / "L1",
        text_col="transcription",
        spk_fn=lambda r: r["speaker_id"],
        n_keywords=args.n_keywords,
        max_pos_utts=args.max_pos,
    )
    summary["L2"] = _build_parquet_chinese_tier(
        tier_name="L2",
        parquet_path=cache / "l2_kespeech" / "data" / "test-00000-of-00008.parquet",
        out_dir=out / "L2",
        text_col="Text",
        spk_fn=lambda r: str(r["ID"]).split("_")[0],
        n_keywords=args.n_keywords,
        max_pos_utts=args.max_pos,
    )
    summary["L3"] = _build_libriphrase_tier(
        zip_path=cache / "l3_libriphrase" / "LibriPhrase_evalset.zip",
        csv_path=cache / "l3_libriphrase" / "libriphrase_diffspk_all_2word.csv",
        out_dir=out / "L3",
        n_keywords=args.n_keywords,
        max_pos_utts=args.max_pos,
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
