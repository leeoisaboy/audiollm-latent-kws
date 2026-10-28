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

The project is a research baseline, not a production wake-word system. Its main evidence is not only a score increase, but also two measurable findings: a **mismatch between two time grids**, and a **negative result for temporal-shape fusion** that many papers would hide.

## Results first

The real-corpus experiment contains 2031 Wu utterances, 3244 target-word occurrences, and 293 words.

| Question | Result | Plain reading |
|---|---:|---|
| Is the correct word in the top 20 candidates? | **94.7%** | 3071 of 3244 occurrences reach the shortlist |
| Acoustic ranking alone: is it first? | **47.4%** | 1538 of 3244 occurrences |
| After default context reranking? | **54.2%** | 1757 of 3244 occurrences |
| With optional homophone-tolerant snapping at the recommended weight? | **55.0%** | 1783 of 3244 occurrences |

The recommended configuration is:

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

## The alignment problem

Audio features use 80 ms frames. The external text reader uses a 40 ms character grid. A naive nearest-frame cut often leaves the target word's first character in the left context, effectively leaking part of the answer.

On 3244 real occurrences:

| Left-context cut | Target first character appears in prefix |
|---|---:|
| Naive nearest-frame hard cut | **35.2%** |
| Snap to the character grid first | **3.4%** |

Only the context boundary changes. The model, shortlist, and fusion weights stay fixed. See [reports/prefix_audit_real.json](reports/prefix_audit_real.json).

## Validation

Configuration selection used nested five-fold cross-validation. The Stage-1 comparison selected `k=8 + mean centering` in all five speaker-grouped folds and four of five utterance-grouped folds.

| Comparison | Change in occurrence top-1 | 95% interval | Grouping |
|---|---:|---:|---|
| 8-frame + centering vs. old 5-frame | **+3.70 points** | [+2.80, +4.61] | speaker |
| Same comparison | **+3.51 points** | [+2.62, +4.42] | utterance |
| Rerank weight 4.0 vs. inherited 1.5 | **+1.05 points** | [+0.47, +1.64] | paired bootstrap |
| Pinyin snap at 4.0 vs. exact snap at 4.0 | **+0.80 points** | [+0.37, +1.26] | paired bootstrap |

The speaker grouping comes from the corpus's `rec_sXX` naming rule, not an official speaker release. It should be rechecked with official metadata or a second corpus.

## Rejected variants

These branches were tested on the real corpus and kept for reproducibility, not as default features.

| Variant | Observation | Decision |
|---|---|---|
| Duration-adaptive windows and peak growing | top-1 46.5%, pool recall 93.4% | not default |
| Multi-prototype / k-means templates | about 45.0%, pool recall 91.7% | research only |
| All-but-the-Top / PCA whitening | 39.5% / 43.2%, clear recall loss | research only |
| Layer-score fusion | lower pool recall than layer 2 | not default |
| Two starts per candidate | +0.68 points at weight 4.0 | below pinyin snap |
| PMI text-prior calibration | -0.96 at weight 1.5; -3.45 at weight 4.0 for gamma 1 | negative result |
| Temporal-shape gate | +0.17 / +0.03 points, both intervals cross zero | disabled |

The shape result is especially important. A first-difference signal can rank candidates in isolation, but adding it with a fixed linear weight to the deployable acoustic+language-model score does not produce a stable gain. Signal existence is not the same as end-to-end utility.

## Engineering result

The audio prefix is identical for all candidates in one utterance. Reusing its KV cache is useful, but the measured gain is modest: **1.30x to 2.14x** for the tested audio lengths. The cache changes scores by at most about 0.04 log-probability units, and the top-1 result is identical on 199 of 200 utterances.

The project does **not** claim 5-10x speedup or an 80% memory reduction. See [reports/cache_benchmark.json](reports/cache_benchmark.json).

## Quick start

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

## Repository layout

The `v3` layout separates the final path from the experiment history.

| Path | Purpose |
|---|---|
| `locator/` | final Stage-1 mean templates, centering, fixed 8-frame search, top-20 pools |
| `reranker/` | final Stage-2 grid snapping, optional pinyin matching, scoring |
| `analysis/` | pool evaluation and prefix-leakage audit |
| `integration/` | Step-Audio-2 feature export, KV cache, low-memory loading |
| `experiments/` | frozen old branches, shape operators, layer probes, rejected variants |
| `scripts/` | real-corpus ablations, cross-validation, weight sweep, cache benchmark |
| `tests/` | production tests; `tests/research/` keeps research-branch tests |

## Tests

```bash
python -m compileall -q .
python -m pytest -q
```

Current local result: 22 tests pass, with one torch-dependent model-loading guard skipped when the runtime is absent.

## Limits

- One private 2031-utterance Wu corpus only.
- Speaker groups use a filename rule, not official speaker IDs.
- Pinyin snapping is a Mandarin approximation, not a Wu phonology model.
- Synthetic fixtures prove wiring only.
- False alarms, far-field noise, power, and real-time end-to-end latency are not evaluated.
- KV-cache speedup is 1.30-2.14x, not 5-10x.
- PMI, ABTT, PCA whitening, multi-prototype templates, existing layer fusion, and fixed-weight shape fusion do not enter the default path.

The code is Apache-2.0. Models, audio, and derived features retain their original licenses.
