"""Language-agnostic 5-step pure-function core pipeline for Latent-KWS.

Core steps:
  1. ``enroll``          - Build L2-normalized word centroids + auto window size from enrollment median duration
  2. ``locate``          - Sliding-window cosine search -> Top-K candidate pool
  3. ``safe_prefix_cut`` - Pure acoustic backoff (no external 40ms reader or pinyin required)
  4. ``kv_rerank``       - Prefix KV-Cache teacher forcing + per-pool z-score fusion
  5. ``threshold``       - Open-set detection decision (positive trigger vs negative reject)

Zero external language/pinyin dependencies. Works identically across Chinese
dialects (Wu, Mandarin, Cantonese, etc.), English, and arbitrary Unicode scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from common.scoring import fuse, zpool
from integration.backend import AudioLLMBackend
from locator.calibration import LatentCalibrator
from locator.search import fixed_span, score_utterance
from locator.templates import LatentTemplateBank, l2_normalize, span_frames


@dataclass
class AutoTemplateBank(LatentTemplateBank):
    """Template bank extended with auto-derived window size and frame rate."""

    frame_s: float = 0.08
    auto_window_frames: int = 8
    median_duration_s: float = 0.64


@dataclass
class CandidatePool:
    """Output of Stage-1 ``locate``."""

    utterance: str
    frames: int
    frame_s: float
    window: int
    words: list[str]
    acoustic_scores: np.ndarray
    starts: np.ndarray
    ends: list[int]
    gold: list[tuple[str, int, int]] | None = None

    def to_row(self, include_gold: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {
            "utt": self.utterance,
            "t": self.frames,
            "frame_s": self.frame_s,
            "window": self.window,
            "words": self.words,
            "score": [float(v) for v in self.acoustic_scores],
            "ps": [int(v) for v in self.starts],
            "pe": [int(v) for v in self.ends],
            "ntop": len(self.words),
        }
        if include_gold and self.gold is not None:
            row["gold"] = [
                {"word": w, "ts": s, "te": e} for w, s, e in self.gold
            ]
        return row


@dataclass
class PrefixCut:
    """Output of Stage-2 ``safe_prefix_cut`` for one candidate."""

    word: str
    start_frame: int
    cut_frame: int
    cut_time_s: float
    prefix_text: str
    mode: str


@dataclass
class RerankedPool:
    """Output of Stage-2 ``kv_rerank``."""

    utterance: str
    words: list[str]
    acoustic_scores: np.ndarray
    lm_scores: np.ndarray
    fused_scores: np.ndarray
    open_scores: np.ndarray
    order: np.ndarray
    prefixes: list[str]
    cut_times_s: list[float]
    lambda_lm: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def top_word(self) -> str:
        return self.words[int(self.order[0])] if len(self.order) else ""

    @property
    def top_fused_score(self) -> float:
        return float(self.fused_scores[int(self.order[0])]) if len(self.order) else float("-inf")

    @property
    def top_open_score(self) -> float:
        return float(self.open_scores[int(self.order[0])]) if len(self.order) else float("-inf")


@dataclass
class DetectionResult:
    """Output of Stage-3 ``threshold`` for open-set triggering."""

    utterance: str
    triggered: bool
    word: str | None
    open_score: float
    fused_score: float
    threshold: float


# ---------------------------------------------------------------------------
# Step 1: enroll
# ---------------------------------------------------------------------------

def compute_auto_window(
    durations_s: Sequence[float],
    frame_s: float,
    min_frames: int = 2,
    max_frames: int = 32,
) -> tuple[int, float]:
    """Derive sliding-window frame width from median effective enrollment duration."""
    valid = [float(d) for d in durations_s if float(d) > 0]
    if not valid:
        return max(min_frames, int(round(0.64 / frame_s))), 0.64
    median_s = float(np.median(valid))
    frames = int(round(median_s / frame_s))
    frames = max(min_frames, min(max_frames, frames))
    return frames, median_s


def enroll(
    enroll_items: Sequence[Mapping[str, Any]],
    feature_getter: Callable[[str], np.ndarray],
    frame_s: float = 0.08,
    calibrator: LatentCalibrator | None = None,
    only: set[str] | None = None,
) -> AutoTemplateBank:
    """Build L2-normalized word centroids and auto-compute window frames from median duration.

    Each item in ``enroll_items`` must provide ``word``, ``clip_id``, and
    optionally ``kw_t0`` / ``kw_t1`` (defaults to full clip span when omitted).
    """
    take_means: dict[str, list[np.ndarray]] = {}
    durations_by_word: dict[str, list[float]] = {}
    all_durations: list[float] = []

    for row in enroll_items:
        word = str(row["word"])
        if only is not None and word not in only:
            continue
        clip_id = str(row["clip_id"])
        try:
            features = np.asarray(feature_getter(clip_id), dtype=np.float64)
        except (FileNotFoundError, OSError, KeyError):
            continue
        if features.ndim != 2 or len(features) == 0:
            continue
        if calibrator is not None:
            features = calibrator.transform(features)
        if "kw_t0" in row and "kw_t1" in row and row["kw_t0"] is not None and row["kw_t1"] is not None:
            t0 = float(row["kw_t0"])
            t1 = float(row["kw_t1"])
            start, end = span_frames(t0, t1, frame_s, len(features))
            dur = max(frame_s, t1 - t0)
        else:
            start, end = 0, len(features)
            dur = len(features) * frame_s
        if end <= start:
            continue
        take_means.setdefault(word, []).append(
            l2_normalize(features[start:end].mean(axis=0))
        )
        durations_by_word.setdefault(word, []).append(dur)
        all_durations.append(dur)

    words = sorted(take_means)
    if not words:
        raise ValueError("no usable enrollment features")
    vectors = l2_normalize(
        np.stack([np.mean(np.stack(take_means[w]), axis=0) for w in words])
    )
    auto_window, median_s = compute_auto_window(all_durations, frame_s=frame_s)
    return AutoTemplateBank(
        words=words,
        vectors=vectors,
        durations={w: float(np.median(vs)) for w, vs in durations_by_word.items()},
        takes={w: len(vs) for w, vs in take_means.items()},
        frame_s=frame_s,
        auto_window_frames=auto_window,
        median_duration_s=median_s,
    )


# ---------------------------------------------------------------------------
# Step 2: locate
# ---------------------------------------------------------------------------

def locate(
    utterance: str,
    features: np.ndarray,
    bank: LatentTemplateBank,
    pool_top: int = 20,
    window: int | None = None,
    frame_s: float | None = None,
    gold: Sequence[Mapping[str, Any]] | None = None,
) -> CandidatePool:
    """Fixed-window cosine scan returning the Top-K candidate pool.

    When ``window`` is ``None``, uses ``bank.auto_window_frames`` (if ``bank`` is
    an ``AutoTemplateBank``), removing hardcoded frame-width assumptions.
    """
    features = np.asarray(features, dtype=np.float64)
    resolved_frame_s = (
        frame_s
        if frame_s is not None
        else getattr(bank, "frame_s", 0.08)
    )
    resolved_window = (
        int(window)
        if window is not None
        else int(getattr(bank, "auto_window_frames", 8))
    )
    effective_window = max(1, min(resolved_window, len(features))) if len(features) > 0 else resolved_window
    scores, starts = score_utterance(features, bank, window=effective_window)
    k = min(int(pool_top), len(bank.words))
    order = np.argsort(-scores, kind="stable")[:k]
    ends = [
        fixed_span(int(starts[idx]), effective_window, len(features))[1]
        for idx in order
    ]
    gold_rows = None
    if gold is not None:
        gold_rows = []
        for item in gold:
            ts = int(np.floor(float(item["kw_t0"]) / resolved_frame_s))
            te = int(np.ceil(float(item["kw_t1"]) / resolved_frame_s))
            gold_rows.append((str(item["word"]), ts, te))
    return CandidatePool(
        utterance=utterance,
        frames=len(features),
        frame_s=resolved_frame_s,
        window=effective_window,
        words=[bank.words[idx] for idx in order],
        acoustic_scores=scores[order],
        starts=starts[order],
        ends=ends,
        gold=gold_rows,
    )


# ---------------------------------------------------------------------------
# Step 3: safe_prefix_cut
# ---------------------------------------------------------------------------

def _reading_backoff_text(
    chars: Sequence[tuple[int, str]],
    cut_time_s: float,
    read_frame_s: float,
) -> str:
    """Language-agnostic timestamp cut on a token reading sequence (no CJK filter)."""
    if not chars or cut_time_s <= 0.0:
        return ""
    target = cut_time_s / read_frame_s
    kept = [symbol for frame, symbol in chars if frame < target]
    if not kept:
        return ""
    # Preserve natural spacing for Latin/space-separated tokens, join directly for CJK
    if any(" " in s for s in kept):
        return "".join(kept).rstrip()
    # If tokens are whole English words (multi-char alpha), join with spaces
    if all(s.isascii() and len(s) > 1 for s in kept):
        return " ".join(kept) + " "
    return "".join(kept)


def safe_prefix_cut(
    pool: CandidatePool,
    backoff_frames: int = 2,
    mode: str = "acoustic_backoff",
    prefix_decoder: Callable[[float], str] | None = None,
    reading: Sequence[tuple[int, str]] | None = None,
    read_frame_s: float = 0.04,
    quantize_s: float = 0.16,
    shared_cut: bool = True,
) -> list[PrefixCut]:
    """Compute leak-safe left-context prefixes for every candidate in ``pool``.

    Modes:
      - ``acoustic_backoff`` (default, reader-free):
        Shifts acoustic start left by ``backoff_frames`` (e.g., 1-2 frames =
        80-160 ms safety buffer). When ``shared_cut=True`` (default), all
        candidates share the Top-1 acoustic cut point so the Audio-LLM decodes
        at most ONE prefix and reuses ONE Prefix KV-Cache across all Top-K
        candidates.
      - ``empty``:
        Zero text prefix (pure audio-conditioned scoring, 0.0% prefix leak).
    """
    cuts: list[PrefixCut] = []
    decode_cache: dict[float, str] = {}
    anchor_start = int(pool.starts[0]) if len(pool.starts) else 0

    for word, start_frame in zip(pool.words, pool.starts):
        start_int = anchor_start if (shared_cut and mode == "acoustic_backoff") else int(start_frame)
        cut_frame = max(0, start_int - int(backoff_frames))
        cut_time_s = round(cut_frame * pool.frame_s, 4)

        if mode == "empty" or cut_frame <= 0:
            prefix_text = ""
        elif mode == "acoustic_backoff":
            if prefix_decoder is not None:
                q_time = (
                    round(cut_time_s / quantize_s) * quantize_s
                    if quantize_s > 0
                    else cut_time_s
                )
                q_time = round(max(0.0, q_time), 4)
                if q_time not in decode_cache:
                    decode_cache[q_time] = prefix_decoder(q_time) if q_time > 0 else ""
                prefix_text = decode_cache[q_time]
            elif reading is not None:
                prefix_text = _reading_backoff_text(reading, cut_time_s, read_frame_s)
            else:
                prefix_text = ""
        else:
            raise ValueError(f"unsupported prefix cut mode: {mode}")

        cuts.append(
            PrefixCut(
                word=word,
                start_frame=start_int,
                cut_frame=cut_frame,
                cut_time_s=cut_time_s,
                prefix_text=prefix_text,
                mode=mode,
            )
        )
    return cuts


# ---------------------------------------------------------------------------
# Step 4: kv_rerank
# ---------------------------------------------------------------------------

def kv_rerank(
    pool: CandidatePool,
    wav_path: str,
    backend: AudioLLMBackend,
    prefix_cuts: Sequence[PrefixCut] | None = None,
    lambda_lm: float = 4.0,
    backoff_frames: int = 2,
    open_beta: float = 0.25,
    mode: str = "acoustic_backoff",
    use_self_prefix: bool = True,
) -> RerankedPool:
    """Rerank ``pool`` using the minimal 3-method backend + per-pool z-score fusion.

    Within-pool ranking uses ``zpool(acoustic) + lambda_lm * zpool(lm)``.
    Open-set detection uses ``acoustic_raw + open_beta * lm_raw`` (absolute
    acoustic + absolute LM log-prob evidence, preserving negative-utterance
    rejectability).
    """
    if prefix_cuts is None:
        decoder = (
            (lambda t: backend.transcribe_prefix(wav_path, t))
            if (use_self_prefix and mode == "acoustic_backoff")
            else None
        )
        prefix_cuts = safe_prefix_cut(
            pool,
            backoff_frames=backoff_frames,
            mode=mode,
            prefix_decoder=decoder,
        )

    grouped: dict[str, list[int]] = {}
    for idx, cut in enumerate(prefix_cuts):
        grouped.setdefault(cut.prefix_text, []).append(idx)

    lm_scores = np.empty(len(pool.words), dtype=np.float64)
    for prefix_text, indices in grouped.items():
        state = backend.prefill_prefix_kv(wav_path, prefix_text)
        suffixes = [pool.words[i] for i in indices]
        lm_scores[indices] = backend.score_suffixes(state, suffixes)

    fused = fuse(pool.acoustic_scores, lm_scores, lambda_lm=lambda_lm)
    # Open-set detection score: raw acoustic + within-pool LM margin (cancels utterance-level Yes-bias)
    lm_med = float(np.median(lm_scores)) if len(lm_scores) else 0.0
    lm_margin = lm_scores - lm_med
    open_scores = np.asarray(pool.acoustic_scores, dtype=np.float64) + float(open_beta) * lm_margin
    order = np.argsort(-fused, kind="stable")

    return RerankedPool(
        utterance=pool.utterance,
        words=list(pool.words),
        acoustic_scores=np.asarray(pool.acoustic_scores, dtype=np.float64),
        lm_scores=lm_scores,
        fused_scores=fused,
        open_scores=open_scores,
        order=order,
        prefixes=[c.prefix_text for c in prefix_cuts],
        cut_times_s=[c.cut_time_s for c in prefix_cuts],
        lambda_lm=float(lambda_lm),
        meta={
            "backend": backend.name,
            "unique_prefixes": len(grouped),
            "mode": mode,
            "backoff_frames": backoff_frames,
            "zpool_acoustic": zpool(pool.acoustic_scores).tolist(),
            "zpool_lm": zpool(lm_scores).tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Step 5: threshold
# ---------------------------------------------------------------------------

def threshold(
    reranked: RerankedPool,
    threshold_value: float,
    score_type: str = "open",
) -> DetectionResult:
    """Apply an open-set decision threshold to ``RerankedPool``."""
    if len(reranked.order) == 0:
        return DetectionResult(
            utterance=reranked.utterance,
            triggered=False,
            word=None,
            open_score=float("-inf"),
            fused_score=float("-inf"),
            threshold=float(threshold_value),
        )
    best_idx = int(reranked.order[0])
    open_val = float(reranked.open_scores[best_idx])
    fused_val = float(reranked.fused_scores[best_idx])
    decision_val = open_val if score_type == "open" else fused_val
    triggered = bool(decision_val >= float(threshold_value))
    return DetectionResult(
        utterance=reranked.utterance,
        triggered=triggered,
        word=reranked.words[best_idx] if triggered else None,
        open_score=open_val,
        fused_score=fused_val,
        threshold=float(threshold_value),
    )
