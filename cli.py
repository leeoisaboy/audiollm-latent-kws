from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.audit import audit_prefix_pollution
from analysis.evaluation import occurrence_metrics, score_records
from analysis.layer_probe import compare_layer_fusions, compare_layers, stage1_metrics
from analysis.shape import shape_fusion_delta
from common.io import read_jsonl, write_json, write_jsonl
from locator import DirectoryLatentStore, build_pool, build_templates, fit_latent_calibrator
from reranker.align import load_reading
from reranker.scoring import load_scorer, rerank_pool


def fit_calibrator(args: argparse.Namespace, rows):
    if args.whitening == "none":
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
    return fit_latent_calibrator(
        arrays,
        mode=args.whitening,
        rank=args.whitening_rank,
    )


def command_locate(args: argparse.Namespace) -> int:
    template_rows = read_jsonl(args.templates)
    store = DirectoryLatentStore(args.enroll_dir, expected_width=args.latent_width)
    calibrator = fit_calibrator(args, template_rows)
    bank = build_templates(
        template_rows,
        store,
        prototypes=args.prototypes,
        calibrator=calibrator,
    )
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
            adaptive_window=args.adaptive_window,
            multi_scale=args.multi_scale,
            use_prototypes=args.use_prototypes,
            use_grow_span=args.grow_span,
            grow_delta=args.grow_delta,
            nms_candidates=args.nms_candidates,
            nms_radius=args.nms_radius,
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
    pool_rows = read_jsonl(args.pool)
    output = []
    for row in pool_rows:
        gold = [
            (item["word"], float(item["ts"]), float(item["te"]))
            for item in row.get("gold", [])
        ]
        result = rerank_pool(
            row,
            readings.get(row["utt"], []),
            read_frame_s,
            scorer,
            str(Path(args.wav_dir) / f"{row['utt']}.wav"),
            variants=args.variants.split(","),
            null_chars=readings.get(shifted_id(pool_rows, row["utt"], args.null_shift), []),
            gold=gold if args.allow_oracle else None,
            transcript=row.get("text", "") if args.allow_oracle else "",
            phonetic=args.phonetic_snap,
            max_phoneme_distance=args.max_phoneme_distance,
            alternate_starts=args.alternate_starts,
            text_only=args.pmi,
        )
        output.append(result.to_row())
    write_jsonl(args.output, output)
    return 0


def shifted_id(rows, utterance: str, shift: int) -> str:
    ids = [row["utt"] for row in rows]
    return ids[(ids.index(utterance) + shift) % len(ids)]


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
        args.variant,
        args.lambda_lm,
        args.pool_top,
        calibration=args.calibration,
        gamma=args.gamma,
    )
    write_json(args.output, occurrence_metrics(records))
    return 0


def command_shape(args: argparse.Namespace) -> int:
    shape_rows = read_jsonl(args.shape)
    pool_rows = {row["utt"]: row for row in read_jsonl(args.pool)}
    acoustic, language_model, shape, gold = [], [], [], []
    candidates = []
    for row in shape_rows:
        pool = pool_rows.get(row["utt"])
        if pool is None:
            continue
        acoustic.append(pool["score"])
        candidates.append(pool["words"])
        language_model.append(row.get("lm_C", []))
        shape.append(row.get("d1", []))
        gold.append([[item["word"] for item in pool.get("gold", [])]])
    write_json(
        args.output,
        shape_fusion_delta(
            acoustic,
            language_model,
            shape,
            gold,
            candidates,
            args.beta,
            gating=args.gating,
            entropy_threshold=args.entropy_threshold,
            lambda_lm=args.lambda_lm,
        ),
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    write_json(args.output, stage1_metrics(read_jsonl(args.pool), pool_top=args.pool_top))
    return 0


def command_compare_layers(args: argparse.Namespace) -> int:
    layers = tuple(int(value.strip()) for value in args.layers.split(",") if value.strip())
    report = compare_layers(
        layers,
        args.features_root,
        args.templates,
        args.search_manifest,
        latent_width=args.latent_width,
        pool_top=args.pool_top,
        layer_pattern=args.layer_pattern,
    )
    write_json(args.output, report)
    print(
        "\n".join(
            f"layer {layer}: pool_recall={report['layers'][layer]['stage1']['pool_recall']:.6f}, "
            f"inter/intra={report['layers'][layer]['template_discriminability']['inter_over_intra']:.6f}"
            for layer in report["ranking_by_pool_recall_then_discriminability"]
        )
    )
    return 0


def command_compare_fusions(args: argparse.Namespace) -> int:
    report = compare_layer_fusions(
        args.fusions,
        args.features_root,
        args.templates,
        args.search_manifest,
        latent_width=args.latent_width,
        pool_top=args.pool_top,
        layer_pattern=args.layer_pattern,
    )
    write_json(args.output, report)
    for name in report["ranking_by_pool_recall_then_discriminability"]:
        view = report["views"][name]
        print(
            f"{name}: pool_recall={view['stage1']['pool_recall']:.6f}, "
            f"inter/intra={view['template_discriminability']['inter_over_intra']:.6f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training-free Audio-LLM latent keyword search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    locate = subparsers.add_parser("locate", help="build latent cosine candidate pools")
    locate.add_argument("--templates", required=True)
    locate.add_argument("--enroll-dir", required=True)
    locate.add_argument("--search-manifest", required=True)
    locate.add_argument("--search-dir", required=True)
    locate.add_argument("--latent-width", type=int, default=3584)
    locate.add_argument("--pool-top", type=int, default=20)
    locate.add_argument("--window", type=int, default=5, help="fixed acoustic window in 80 ms frames")
    locate.add_argument("--include-gold", action="store_true")
    locate.add_argument("--adaptive-window", action="store_true")
    locate.add_argument("--multi-scale", type=int, default=0)
    locate.add_argument("--prototypes", type=int, default=1)
    locate.add_argument("--no-prototypes", dest="use_prototypes", action="store_false")
    locate.set_defaults(use_prototypes=True)
    locate.add_argument("--grow-span", action="store_true")
    locate.add_argument("--grow-delta", type=float, default=0.05)
    locate.add_argument("--nms-candidates", type=int, default=1)
    locate.add_argument("--nms-radius", type=int, default=2)
    locate.add_argument("--whitening", choices=("none", "center", "abtt", "whiten"), default="none")
    locate.add_argument("--whitening-rank", type=int, default=32)
    locate.add_argument("--calibration-dir", default="")
    locate.add_argument("--calibration-manifest", default="")
    locate.add_argument("--output", required=True)
    locate.set_defaults(handler=command_locate)

    evaluate = subparsers.add_parser("evaluate", help="summarize Stage-1 pool recall and ranks")
    evaluate.add_argument("--pool", required=True, help="pool JSONL with gold when --include-gold was used")
    evaluate.add_argument("--pool-top", type=int, default=20)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=command_evaluate)

    layer_probe = subparsers.add_parser("compare-layers", help="compare layer-wise discriminability and Stage-1 recall")
    layer_probe.add_argument("--features-root", required=True, help="root containing layer_<n>/{enroll,search}")
    layer_probe.add_argument("--templates", required=True)
    layer_probe.add_argument("--search-manifest", required=True)
    layer_probe.add_argument("--layers", default="0,1,2,3,4,8")
    layer_probe.add_argument("--layer-pattern", default="layer_{layer}")
    layer_probe.add_argument("--latent-width", type=int, default=3584)
    layer_probe.add_argument("--pool-top", type=int, default=20)
    layer_probe.add_argument("--output", required=True)
    layer_probe.set_defaults(handler=command_compare_layers)

    fusion_probe = subparsers.add_parser(
        "compare-fusions", help="compare weighted or concatenated layer views for Stage 1"
    )
    fusion_probe.add_argument("--features-root", required=True)
    fusion_probe.add_argument("--templates", required=True)
    fusion_probe.add_argument("--search-manifest", required=True)
    fusion_probe.add_argument(
        "--fusions",
        default="0:0.5,2:0.5;0:0.7,2:0.3;0:0.3,2:0.7;concat:0,2",
    )
    fusion_probe.add_argument("--layer-pattern", default="layer_{layer}")
    fusion_probe.add_argument("--latent-width", type=int, default=3584)
    fusion_probe.add_argument("--pool-top", type=int, default=20)
    fusion_probe.add_argument("--output", required=True)
    fusion_probe.set_defaults(handler=command_compare_fusions)

    rerank = subparsers.add_parser("rerank", help="teacher-force a frozen Audio-LLM scorer")
    rerank.add_argument("--pool", required=True)
    rerank.add_argument("--reading", required=True)
    rerank.add_argument("--wav-dir", required=True)
    rerank.add_argument("--scorer", required=True, help="Python module:factory returning an AudioLMScorer")
    rerank.add_argument("--model", default="")
    rerank.add_argument("--adapter", default="")
    rerank.add_argument("--device", default="cuda:0")
    rerank.add_argument("--variants", default="A,P,X,S,T,C,O")
    rerank.add_argument("--null-shift", type=int, default=137)
    rerank.add_argument("--allow-oracle", action="store_true", help="evaluation-only O/T contexts")
    rerank.add_argument("--phonetic-snap", action="store_true")
    rerank.add_argument("--max-phoneme-distance", type=int, default=1)
    rerank.add_argument("--alternate-starts", action="store_true")
    rerank.add_argument("--pmi", action="store_true", help="also compute text-only scores")
    rerank.add_argument("--no-kv-cache", dest="kv_cache", action="store_false")
    rerank.set_defaults(kv_cache=True)
    rerank.add_argument("--output", required=True)
    rerank.set_defaults(handler=command_rerank)

    audit = subparsers.add_parser("audit", help="measure prefix pollution without changing ranking")
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
    fuse.add_argument("--variant", default="C")
    fuse.add_argument("--lambda-lm", type=float, default=4.0)
    fuse.add_argument("--calibration", choices=("none", "pmi"), default="none")
    fuse.add_argument("--gamma", type=float, default=1.0)
    fuse.add_argument("--pool-top", type=int, default=20)
    fuse.add_argument("--output", required=True)
    fuse.set_defaults(handler=command_fuse)

    shape = subparsers.add_parser("analyze-shape", help="audit the negative temporal-shape result")
    shape.add_argument("--shape", required=True)
    shape.add_argument("--pool", required=True)
    shape.add_argument("--beta", type=float, default=1.0)
    shape.add_argument("--gating", action="store_true")
    shape.add_argument("--entropy-threshold", type=float, default=0.65)
    shape.add_argument("--lambda-lm", type=float, default=4.0)
    shape.add_argument("--output", required=True)
    shape.set_defaults(handler=command_shape)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
