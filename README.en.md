<p align="center">
  <a href="https://github.com/leeoisaboy/audiollm-latent-kws">
    <img src="docs/assets/title_banner.en.png" alt="AudioLLM-Latent-KWS" />
  </a>
</p>

<p align="center">
  <a href="README.md"><img alt="Chinese" src="https://img.shields.io/badge/中文-README-2f81f7?style=flat-square"></a>
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-555555?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-b8860b?style=flat-square">
</p>

# AudioLLM-Latent-KWS

**Add a new dialect command with a few enrollment clips, without retraining the audio model.**

This repository studies a practical question: do the frozen representations of an audio-language model already contain enough acoustic information to find a newly enrolled word? The system uses a two-stage, training-free pipeline. Stage 1 builds a shortlist with mean acoustic templates. Stage 2 puts each candidate back into the frozen model with its left text context and rescores it.

The project is a research baseline. It reports two findings that are easy to miss: a **mismatch between two time grids**, and a **temporal-shape experiment that did not improve the final score**.

## Main results

The real-corpus experiment contains 2031 Wu utterances, 3244 target-word occurrences, and 293 words.

| Question | Result | Plain reading |
|---|---:|---|
| Is the correct word in the top 20 candidates? | **94.7%** | 3071 of 3244 occurrences reach the shortlist |
| Is it first with acoustic ranking only? | **47.4%** | 1538 of 3244 occurrences |
| After default context reranking? | **54.2%** | 1757 of 3244 occurrences |
| With optional homophone-tolerant snapping at the recommended weight? | **55.0%** | 1783 of 3244 occurrences |

The recommended configuration is as follows.

```text
layer 2 of the frozen audio model
  + 8-frame (about 640 ms) mean template
  + enrollment mean centering
  + top-20 shortlist
  + character-grid snap on the 40 ms reading timeline
  + acoustic score + 4.0 x language-model score
  + optional Mandarin-pinyin fuzzy first-character snap
  + temporal shape disabled
```

The `4.0` multiplies the language-model term; the acoustic term remains at 1. The measured curve is flat from 4.0 to 5.5. Weight 5.5 has the highest occurrence-only score (55.1%), while 4.0 gives 55.0% and a higher word-level score (64.5% versus 64.1%). The difference is 0.1 point, inside sampling noise, so the repository keeps the more conservative and balanced 4.0.

## Two time grids can misalign

Audio features use 80 ms frames. The external text reader uses a 40 ms character grid. A naive nearest-frame cut often leaves the target word's first character in the left context, effectively leaking part of the answer.

On 3244 real occurrences:

| Left-context cut | Target first character appears in prefix |
|---|---:|
| Naive nearest-frame hard cut | **35.2%** |
| Snap to the character grid first | **3.4%** |

Only the context boundary changes. The model, shortlist, and fusion weights stay fixed. See [reports/prefix_audit_real.json](reports/prefix_audit_real.json).

## How the settings were checked

Configuration selection used nested five-fold cross-validation. The Stage-1 comparison selected `k=8 + mean centering` in all five speaker-grouped folds and four of five utterance-grouped folds.

| Comparison | Change in occurrence top-1 | 95% interval | Grouping |
|---|---:|---:|---|
| 8-frame + centering vs. old 5-frame | **+3.70 points** | [+2.80, +4.61] | speaker |
| Same comparison | **+3.51 points** | [+2.62, +4.42] | utterance |
| Rerank weight 4.0 vs. inherited 1.5 | **+1.05 points** | [+0.47, +1.64] | paired bootstrap |
| Pinyin snap at 4.0 vs. exact snap at 4.0 | **+0.80 points** | [+0.37, +1.26] | paired bootstrap |

The speaker grouping comes from the corpus's `rec_sXX` naming rule, not an official speaker release. It should be rechecked with official metadata or a second corpus.

## Variants outside the default path

These branches were tested on the real corpus and kept for reproducibility. They are not enabled by default.

| Variant | Observation | Decision |
|---|---|---|
| Duration-adaptive windows and peak growing | top-1 46.5%, pool recall 93.4% | not default |
| Multi-prototype / k-means templates | about 45.0%, pool recall 91.7% | research only |
| Principal-component removal / PCA whitening | 39.5% / 43.2%, clear recall loss | experiment only |
| Layer-score fusion | lower pool recall than layer 2 | not default |
| Two starts per candidate | +0.68 points at weight 4.0 | below pinyin snap |
| Text-prior calibration | -0.96 at weight 1.5; -3.45 at weight 4.0 for gamma 1 | kept as a result record |
| Temporal-shape gate | +0.17 / +0.03 points, both intervals cross zero | disabled |

The shape result is especially important. A first-difference signal can rank candidates in isolation, but adding it with a fixed linear weight to the acoustic and language-model scores does not produce a stable gain. A signal can work on its own and still fail to improve the full pipeline.

## Cache speedup

The audio prefix is identical for all candidates in one utterance. Reusing its KV cache is useful, and the measured speedup is **1.30x to 2.14x** for the tested audio lengths. The cache changes scores by at most about 0.04 log-probability units, and the top-1 result is identical on 199 of 200 utterances. See [reports/cache_benchmark.json](reports/cache_benchmark.json).

## Running the project

Synthetic fixtures only verify the interfaces. They are not accuracy evidence.

```bash
git clone https://github.com/leeoisaboy/audiollm-latent-kws.git
cd audiollm-latent-kws
python -m venv .venv
python -m pip install -e ".[dev]"
python scripts/make_fixtures.py --output data/synthetic
```

```bash
python cli.py locate \
  --templates data/synthetic/templates.jsonl \
  --enroll-dir data/synthetic/enroll \
  --search-manifest data/synthetic/search_manifest.jsonl \
  --search-dir data/synthetic/search \
  --latent-width 16 \
  --window 8 \
  --calibration center \
  --pool-top 20 \
  --include-gold \
  --output reports/synthetic/pool.jsonl

python cli.py evaluate --pool reports/synthetic/pool.jsonl --output reports/synthetic/stage1.json

python cli.py rerank \
  --pool reports/synthetic/pool.jsonl \
  --reading data/synthetic/readings.jsonl \
  --wav-dir data/synthetic \
  --scorer examples.dummy_scorer:create_scorer \
  --output reports/synthetic/lm.jsonl

python cli.py fuse \
  --pool reports/synthetic/pool.jsonl \
  --lm reports/synthetic/lm.jsonl \
  --lambda-lm 4.0 \
  --output reports/synthetic/metrics.json

python cli.py audit \
  --pool reports/synthetic/pool.jsonl \
  --lm reports/synthetic/lm.jsonl \
  --reading data/synthetic/readings.jsonl \
  --output reports/synthetic/prefix_audit.json
```

Install `.[phonetic]` and add `--phonetic-snap` to enable the optional pinyin branch.

Real feature extraction needs private audio and a local frozen checkpoint:

```bash
python integration/step_audio2.py \
  --model /path/to/Step-Audio-2-mini \
  --manifest /private/path/all_clips.jsonl \
  --enroll-manifest /private/path/enroll.jsonl \
  --search-manifest /private/path/search.jsonl \
  --output-root features \
  --layers 0,1,2,3,4,8 \
  --device cuda:0 \
  --quantization 4bit \
  --dtype float16
```

Weights, restricted audio, speaker identities, and recognizer output are not distributed.

## Code structure

The `v3` layout separates the final runtime path from the experiment record. The first part is enough for normal use; the second part supports result checks and reruns. Each `__init__.py` only exports the public interface of its package and is not listed separately below.

### Directory tree

```text
audiollm-latent-kws-v3/
├── cli.py                         command-line entry for five subcommands
├── pyproject.toml                 dependencies, packaging, command declaration
├── README.md / README.en.md       Chinese and English project pages
├── LICENSE                        Apache-2.0 license
├── SECURITY.md                    privacy and security notes
├── CONTRIBUTING.md                contribution notes
├── .github/workflows/ci.yml       tests on Python 3.11 and 3.12
├── common/                        shared runtime code
│   ├── config.py                  fixed parameters and time grids
│   ├── io.py                      JSON and JSONL input/output
│   └── scoring.py                 normalization, fusion, and ranks
├── locator/                       Stage 1, acoustic template search
│   ├── templates.py               build one mean template per word
│   ├── calibration.py             remove the enrollment-wide offset
│   └── search.py                  8-frame window, cosine score, top-20
├── reranker/                      Stage 2, left-context rescoring
│   ├── align.py                   40 ms character-grid snap, optional pinyin
│   └── scoring.py                 candidate scoring and rerank flow
├── analysis/                      evaluation and error checks
│   ├── evaluation.py              pool recall, top-1, word-level metrics
│   └── audit.py                   first-character leakage audit
├── integration/                   frozen-model integration
│   ├── step_audio2.py             feature export, scoring, cache reuse
│   └── low_memory_safetensors.py  low-memory weight loading
├── examples/
│   └── dummy_scorer.py            fixed fake scores for CLI smoke tests
├── experiments/                   variants outside the default path
│   ├── layer_probe.py             layer and layer-fusion comparison
│   ├── shape.py                   differencing, resampling, DTW, entropy gate
│   ├── text_prior.py              text-only calibration scorer
│   └── legacy/                    old search, alignment, templates, calibration
│       ├── search.py              window growth, multiscale search, NMS
│       ├── align.py               old left-context variants
│       ├── templates.py           k-means sub-templates and old template store
│       ├── calibration.py         component removal and PCA whitening
│       └── layers.py              multi-layer configuration and loading
├── scripts/                       real-corpus experiments and reproduction
│   ├── make_fixtures.py           generate synthetic fixtures
│   ├── stage1_cv.py               grouped Stage-1 cross-validation
│   ├── stage1_real_ablation.py    window, centering, whitening, template comparison
│   ├── stage1_grow_real.py        peak-neighborhood window growth
│   ├── stage1_layer_fusion_real.py    layer-score fusion
│   ├── stage2_real_ablation.py    pinyin, two starts, text prior, leakage
│   ├── lambda_sweep.py            language-model weight sweep and intervals
│   ├── lambda_cv.py               weight selection and held-out validation
│   ├── shape_gate_real.py         temporal-shape weighting on real data
│   ├── shape_gate_cv.py           temporal-shape cross-validation
│   ├── shape_dtw_real.py          DTW and subsequence DTW
│   └── benchmark_kv_cache.py      latency, memory, and ranking parity
├── tests/                         default-path unit tests
│   ├── test_locator.py            templates, centering, windows, pool search
│   ├── test_alignment.py          grid snap, pinyin matching, prefixes
│   ├── test_scoring.py            normalization, ranks, fusion, leakage
│   ├── test_step_audio2_guard.py  half-precision linear guard
│   └── research/                  experiment-branch tests
│       ├── test_layer_probe.py    layer comparison and fusion
│       ├── test_shape.py          shape operators and DTW
│       └── test_shape_gate.py     shape gating and baseline scores
├── data/                          schema examples and synthetic inputs
├── docs/                          architecture, models, reproduction, results
└── reports/                       JSON evidence and report index
```

### Final runtime path

| File | Purpose |
|---|---|
| `cli.py` | Provides `locate`, `evaluate`, `rerank`, `fuse`, and `audit`; loads arguments, calls each stage, and writes results. |
| `common/config.py` | Holds acoustic and text frame lengths, the feature layer, window length, pool size, and language-model weight. |
| `common/io.py` | Reads and writes JSON and JSONL. |
| `common/scoring.py` | Normalizes pool scores, fuses acoustic and language-model terms, and computes ranks. |
| `locator/templates.py` | Builds word templates by averaging frames inside each take, then averaging takes for the same word; also records duration and take count. |
| `locator/calibration.py` | Estimates and removes a global enrollment mean. The final path uses mean centering only. |
| `locator/search.py` | Runs an 8-frame mean window over an utterance, scores cosine similarity against templates, keeps the best start per word, and returns the top 20 words. |
| `reranker/align.py` | Reads the 40 ms character grid, snaps a candidate start to a character boundary, and optionally matches a nearby pronunciation through Mandarin pinyin. |
| `reranker/scoring.py` | Defines the scorer interface, walks the candidate pool, calls the model, and collects rerank results. |
| `analysis/evaluation.py` | Computes top-20 pool recall, occurrence top-1, and word-level metrics. |
| `analysis/audit.py` | Compares nearest-frame cuts with character-grid cuts and counts first-character leakage in the prefix. |
| `integration/step_audio2.py` | Loads the frozen Step-Audio-2, exports layer features, implements the rerank scorer, and reuses the audio-prefix cache. |
| `integration/low_memory_safetensors.py` | Loads safetensors weights lazily or with lower memory use. |

### Experiment modules

| File | Purpose |
|---|---|
| `experiments/layer_probe.py` | Compares layers and layer-fusion variants by separation, top-1, and pool recall. |
| `experiments/shape.py` | Resampling, first differences, shape templates, cosine distance, DTW, subsequence DTW, and entropy gating. |
| `experiments/text_prior.py` | Text-only calibration scorer kept for the rejected text-prior experiment. |
| `experiments/legacy/align.py` | Old context-splitting variants, including no-prefix, prefix, and cross-word branches. |
| `experiments/legacy/calibration.py` | Old component-removal and PCA-whitening calibrations. |
| `experiments/legacy/layers.py` | Old multi-layer fusion settings and feature loading. |
| `experiments/legacy/search.py` | Old search with window growth, multiscale scoring, alternate starts, and non-maximum suppression. |
| `experiments/legacy/templates.py` | Old template construction with k-means sub-templates and a directory feature store. |

### Reproduction scripts

| File | Purpose |
|---|---|
| `scripts/make_fixtures.py` | Generates synthetic features, enrollment rows, search rows, and readings to verify the CLI wiring. |
| `scripts/stage1_cv.py` | Grouped cross-validation for Stage-1 window length and centering. |
| `scripts/stage1_real_ablation.py` | Stage-1 real-corpus comparisons over windows, centering, whitening, templates, and layers. |
| `scripts/stage1_grow_real.py` | Window growth around score peaks. |
| `scripts/stage1_layer_fusion_real.py` | Layer-score fusion. |
| `scripts/stage2_real_ablation.py` | Stage-2 pinyin matching, two starts, text-prior calibration, and prefix leakage. |
| `scripts/lambda_sweep.py` | Language-model weight sweep with paired intervals. |
| `scripts/lambda_cv.py` | Weight selection on training folds and validation on held-out data. |
| `scripts/shape_gate_real.py` | Temporal-shape weighting on real data. |
| `scripts/shape_gate_cv.py` | Temporal-shape cross-validation and weight selection. |
| `scripts/shape_dtw_real.py` | Dynamic time warping and subsequence dynamic time warping. |
| `scripts/benchmark_kv_cache.py` | Latency, peak memory, and ranking parity for cached and full-sequence scoring. |

### Tests and support files

| File | Purpose |
|---|---|
| `examples/dummy_scorer.py` | Deterministic fake scorer for CLI and fusion smoke tests; it is not accuracy evidence. |
| `tests/test_locator.py` | Checks mean templates, centering, window shapes, and pool localization. |
| `tests/test_alignment.py` | Checks grid snapping, pinyin matching, and prefix construction. |
| `tests/test_scoring.py` | Checks normalization, ties, score fusion, and leakage counting. |
| `tests/test_step_audio2_guard.py` | Checks the half-precision linear guard and skips when `torch` is absent. |
| `tests/research/test_layer_probe.py` | Checks layer comparison and fusion results. |
| `tests/research/test_shape.py` | Checks shape operators, DTW, and entropy gating. |
| `tests/research/test_shape_gate.py` | Checks shape weighting and baseline scores. |
| `pyproject.toml` | Declares the package, version, dependencies, optional extras, and command entry point. |
| `data/` | Provides schema examples and synthetic fixtures without private audio. |
| `docs/` | Architecture, model, reproduction, and full-result documentation. |
| `reports/` | JSON experiment results; `reports/README.md` maps reports to conclusions. |
| `.github/workflows/ci.yml` | Installs the package and runs tests on Python 3.11 and 3.12. |

## Tests

```bash
python -m compileall -q .
python -m pytest -q
```

Current local result: 22 tests pass, with one torch-dependent model-loading guard skipped when the runtime is absent.

## Scope and limits

- One private 2031-utterance Wu corpus only.
- Speaker groups use a filename rule, not official speaker IDs.
- Pinyin snapping uses a Mandarin approximation and should not be read as a Wu phonology result.
- Synthetic fixtures prove wiring only.
- False alarms, far-field noise, power, and real-time end-to-end latency are not evaluated.
- KV-cache speedup is 1.30-2.14x.
- Text-prior calibration, principal-component removal, PCA whitening, multi-template models, existing layer fusion, and fixed-weight shape fusion do not enter the default path.

The code is Apache-2.0. Models, audio, and derived features retain their original licenses.
