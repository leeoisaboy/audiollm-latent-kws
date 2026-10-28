"""Command line entry point for the production two-stage pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.audit import audit_prefix_pollution
from analysis.evaluation import occurrence_metrics, pool_recall, score_records
from common.config import DEFAULT_CALIBRATION, PRIMARY_K
from common.io import read_jsonl, write_json, write_jsonl
from locator import (
    DirectoryLatentStore,
    build_pool,
    build_templates,
    fit_latent_calibrator,
)
from reranker.align import load_reading
from reranker.scoring import load_scorer, rerank_pool


def fit_calibrator(args: argparse.Namespace, rows):
    if args.calibration == "none":
        return None
    clip_ids = []
    if args.calibration_manifest:
        clip_ids = [str(row["clip_id"]) for row in read_jsonl(args.calibration_manifest)]
    else:
        clip_ids = [str(row["clip_id"]) for row in rows]
    calibration_dir = Path(args.calibration_dir or args.enroll_dir)
    arrays = []
    for clip_id in dict.fromkeys(clip_ids):
        try:
            arrays.append(np.load(calibration_dir / f"{clip_id}.npy"))
        except (FileNotFoundError, OSError):
            continue
    if not arrays:
        raise ValueError("no calibration features available")
    return fit_latent_calibrator(arrays, mode=args.calibration)


def command_locate(args: argparse.Namespace) -> int:
    template_rows = read_jsonl(args.templates)
    store = DirectoryLatentStore(args.enroll_dir, expected_width=args.latent_width)
    calibrator = fit_calibrator(args, template_rows)
    bank = build_templates(template_rows, store, calibrator=calibrator)
    output_rows = []
    for row in read_jsonl(args.search_manifest):
        features = np.load(Path(args.search_dir) / f"{row['utt']}.npy")
        if calibrator is not None:
            features = calibrator.transform(features)
        pool = build_pool(
            row["utt"],
            features,
            bank,
            pool_top=args.pool_top,
            window=args.window,
            gold=row.get("gold"),
        )
        output_rows.append(pool.to_row(include_gold=args.include_gold))
    write_jsonl(args.output, output_rows)
    return 0


def command_rerank(args: argparse.Namespace) -> int:
    scorer = load_scorer(
        args.scorer,
        model=args.model,
        device=args.device,
        adapter=args.adapter,
        kv_cache=args.kv_cache,
    )
    readings, read_frame_s = load_reading(args.reading)
    output = []
    for row in read_jsonl(args.pool):
        result = rerank_pool(
            row,
            readings.get(row["utt"], []),
            read_frame_s,
            scorer,
            str(Path(args.wav_dir) / f"{row['utt']}.wav"),
            phonetic=args.phonetic_snap,
            max_phoneme_distance=args.max_phoneme_distance,
        )
        output.append(result.to_row())
    write_jsonl(args.output, output)
    return 0


def command_audit(args: argparse.Namespace) -> int:
    readings, frame_s = load_reading(args.reading)
    pool_rows = read_jsonl(args.pool)
    lm_rows = {row["utt"]: row for row in read_jsonl(args.lm)}
    report = audit_prefix_pollution(
        pool_rows,
        lm_rows,
        readings,
        frame_s,
        null_shift=args.null_shift,
        tolerance=args.snap_tol,
        phonetic=args.phonetic_snap,
        max_phoneme_distance=args.max_phoneme_distance,
    )
    write_json(args.output, report)
    return 0


def command_fuse(args: argparse.Namespace) -> int:
    pool_rows = read_jsonl(args.pool)
    lm_rows = {row["utt"]: row for row in read_jsonl(args.lm)}
    records = score_records(
        pool_rows,
        lm_rows,
        "C",
        args.lambda_lm,
        args.pool_top,
    )
    write_json(args.output, occurrence_metrics(records))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.pool)
    write_json(
        args.output,
        {"n": len(rows), "pool_recall": pool_recall(rows, args.pool_top)},
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training-free Audio-LLM latent keyword search"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    locate = subparsers.add_parser("locate", help="build latent cosine candidate pools")
    locate.add_argument("--templates", required=True)
    locate.add_argument("--enroll-dir", required=True)
    locate.add_argument("--search-manifest", required=True)
    locate.add_argument("--search-dir", required=True)
    locate.add_argument("--latent-width", type=int, default=3584)
    locate.add_argument("--pool-top", type=int, default=20)
    locate.add_argument(
        "--window",
        type=int,
        default=PRIMARY_K,
        help="fixed acoustic window in 80 ms frames",
    )
    locate.add_argument("--include-gold", action="store_true")
    locate.add_argument(
        "--calibration",
        choices=("none", "center"),
        default=DEFAULT_CALIBRATION,
    )
    locate.add_argument("--calibration-dir", default="")
    locate.add_argument("--calibration-manifest", default="")
    locate.add_argument("--output", required=True)
    locate.set_defaults(handler=command_locate)

    evaluate = subparsers.add_parser(
        "evaluate", help="summarize Stage-1 pool recall"
    )
    evaluate.add_argument(
        "--pool",
        required=True,
        help="pool JSONL with gold when --include-gold was used",
    )
    evaluate.add_argument("--pool-top", type=int, default=20)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=command_evaluate)

    rerank = subparsers.add_parser(
        "rerank", help="teacher-force a frozen Audio-LLM scorer"
    )
    rerank.add_argument("--pool", required=True)
    rerank.add_argument("--reading", required=True)
    rerank.add_argument("--wav-dir", required=True)
    rerank.add_argument(
        "--scorer",
        required=True,
        help="Python module:factory returning an AudioLMScorer",
    )
    rerank.add_argument("--model", default="")
    rerank.add_argument("--adapter", default="")
    rerank.add_argument("--device", default="cuda:0")
    rerank.add_argument("--phonetic-snap", action="store_true")
    rerank.add_argument("--max-phoneme-distance", type=int, default=1)
    rerank.add_argument("--no-kv-cache", dest="kv_cache", action="store_false")
    rerank.set_defaults(kv_cache=True)
    rerank.add_argument("--output", required=True)
    rerank.set_defaults(handler=command_rerank)

    audit = subparsers.add_parser(
        "audit", help="measure prefix pollution without changing ranking"
    )
    audit.add_argument("--pool", required=True)
    audit.add_argument("--lm", required=True)
    audit.add_argument("--reading", required=True)
    audit.add_argument("--null-shift", type=int, default=137)
    audit.add_argument("--snap-tol", type=int, default=6)
    audit.add_argument("--phonetic-snap", action="store_true")
    audit.add_argument("--max-phoneme-distance", type=int, default=1)
    audit.add_argument("--output", required=True)
    audit.set_defaults(handler=command_audit)

    fuse = subparsers.add_parser("fuse", help="apply pool z-score fusion and summarize")
    fuse.add_argument("--pool", required=True)
    fuse.add_argument("--lm", required=True)
    fuse.add_argument("--lambda-lm", type=float, default=4.0)
    fuse.add_argument("--pool-top", type=int, default=20)
    fuse.add_argument("--output", required=True)
    fuse.set_defaults(handler=command_fuse)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
