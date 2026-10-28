from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.layer_probe import (
    compare_layer_fusions,
    compare_layers,
    stage1_metrics,
    template_discriminability,
)
from cli import main
from common.io import write_jsonl


class LayerProbeTests(unittest.TestCase):
    def test_compare_layers_ranks_separable_features_first(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        words = ["开灯", "关窗", "拉帘"]
        centers = np.eye(3, 8) * 6.0
        template_rows = []
        search_rows = []
        rng = np.random.default_rng(17)
        for word_index, word in enumerate(words):
            for take in range(2):
                clip_id = f"clip-{word_index}-{take}"
                template_rows.append({"clip_id": clip_id, "word": word, "kw_t0": 0.0, "kw_t1": 0.24})
                for layer in (3, 7):
                    frames = rng.normal(scale=0.05, size=(4, 8))
                    frames[1:3] += centers[word_index]
                    destination = root / f"layer_{layer}" / "enroll"
                    destination.mkdir(parents=True, exist_ok=True)
                    np.save(destination / f"{clip_id}.npy", frames)
        for utterance_index in range(6):
            word_index = utterance_index % 3
            search_rows.append(
                {
                    "utt": f"utt-{utterance_index}",
                    "gold": [{"word": words[word_index], "kw_t0": 0.24, "kw_t1": 0.64}],
                }
            )
            for layer in (3, 7):
                frames = rng.normal(scale=0.05, size=(9, 8))
                if layer == 3:
                    frames[3:8] += centers[word_index]
                else:
                    frames[3:8] += centers[0]
                destination = root / f"layer_{layer}" / "search"
                destination.mkdir(parents=True, exist_ok=True)
                np.save(destination / f"utt-{utterance_index}.npy", frames)
        templates = root / "templates.jsonl"
        manifest = root / "manifest.jsonl"
        write_jsonl(templates, template_rows)
        write_jsonl(manifest, search_rows)
        report = compare_layers((3, 7), root, templates, manifest, latent_width=8, pool_top=1)
        self.assertEqual(report["best_layer"], "3")
        self.assertEqual(report["layers"]["3"]["stage1"]["pool_recall"], 1.0)
        self.assertLess(report["layers"]["7"]["stage1"]["pool_recall"], 1.0)
        self.assertLess(
            report["layers"]["3"]["template_discriminability"]["inter_template_cosine"],
            report["layers"]["3"]["template_discriminability"]["intra_template_cosine"],
        )

    def test_template_metrics_and_stage1(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        rows = []
        for index in range(2):
            frames = np.zeros((4, 3), dtype=np.float64)
            frames[:, index] = 4.0
            np.save(root / f"clip-{index}.npy", frames)
            rows.append({"clip_id": f"clip-{index}", "word": f"w{index}", "kw_t0": 0.0, "kw_t1": 0.24})
        metrics = template_discriminability(rows, root, latent_width=3)
        self.assertEqual(metrics["words"], 2)
        self.assertAlmostEqual(metrics["inter_template_cosine"], 0.0)
        stage = stage1_metrics(
            [{"words": ["w0", "w1"], "gold": [{"word": "w0"}]}],
            pool_top=2,
        )
        self.assertEqual(stage["pool_recall"], 1.0)

    def test_cli_evaluate_and_compare_layers(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        pool = root / "pool.jsonl"
        output = root / "metrics.json"
        write_jsonl(pool, [{"utt": "utt", "words": ["开灯", "关窗"], "gold": [{"word": "关窗"}]}])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main(["evaluate", "--pool", str(pool), "--pool-top", "2", "--output", str(output)])
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["pool_recall"], 1.0)

    def test_compare_weighted_and_concatenated_layer_views(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        words = ["开灯", "关窗"]
        template_rows = []
        search_rows = []
        rng = np.random.default_rng(29)
        for index, word in enumerate(words):
            center = np.zeros(8)
            center[index] = 5.0
            for take in range(3):
                clip_id = f"clip-{index}-{take}"
                template_rows.append(
                    {"clip_id": clip_id, "word": word, "kw_t0": 0.0, "kw_t1": 0.32}
                )
                for layer in (0, 2):
                    frames = rng.normal(scale=0.03, size=(8, 8))
                    frames[2:6] += center
                    destination = root / f"layer_{layer}" / "enroll"
                    destination.mkdir(parents=True, exist_ok=True)
                    np.save(destination / f"{clip_id}.npy", frames)
        for index, word in enumerate(words):
            search_rows.append(
                {
                    "utt": f"utt-{index}",
                    "gold": [{"word": word, "kw_t0": 0.16, "kw_t1": 0.48}],
                }
            )
            center = np.zeros(8)
            center[index] = 5.0
            for layer in (0, 2):
                frames = rng.normal(scale=0.03, size=(12, 8))
                frames[3:7] += center
                destination = root / f"layer_{layer}" / "search"
                destination.mkdir(parents=True, exist_ok=True)
                np.save(destination / f"utt-{index}.npy", frames)
        templates = root / "templates.jsonl"
        manifest = root / "manifest.jsonl"
        write_jsonl(templates, template_rows)
        write_jsonl(manifest, search_rows)
        report = compare_layer_fusions(
            "0:0.5,2:0.5;concat:0,2",
            root,
            templates,
            manifest,
            latent_width=8,
            pool_top=1,
        )
        self.assertIn("0:0.5,2:0.5", report["views"])
        self.assertIn("concat:0,2", report["views"])
        self.assertEqual(report["views"]["concat:0,2"]["view_width"], 16)
        self.assertIn(report["best_view"], {"0:0.5,2:0.5", "concat:0,2"})
        for view in report["views"].values():
            self.assertEqual(view["stage1"]["pool_recall"], 1.0)
