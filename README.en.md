<p align="center">
  <a href="https://github.com/leeoisaboy/audiollm-latent-kws">
    <img src="docs/assets/title_banner.en.png" alt="AudioLLM-Latent-KWS: training-free open-vocabulary dialect keyword search" />
  </a>
</p>

<p align="center">
  <a href="README.md"><img alt="Chinese" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README-d73a49?style=flat-square"></a>
  <img alt="English" src="https://img.shields.io/badge/English-current-2f81f7?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-555555?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-b8860b?style=flat-square">
</p>

# AudioLLM-Latent-KWS

**Training-free open-vocabulary dialect keyword search with frozen Audio-LLM latents.**

Smart-home and assisted-living devices keep meeting new dialect action commands. Collecting labels and fine-tuning a model for every new command puts the cost on the long tail. This project studies a more basic question. Does a frozen audio language model already contain enough acoustic and semantic information for a new word to be retrieved from a few enrollment clips.

The repository implements a reproducible two-stage pipeline. Stage 1 matches mean enrollment templates in an early Audio-LLM latent space and returns a Top-20 candidate pool. Stage 2 puts each candidate back into the frozen model with an aligned language context and reranks the pool by teacher forcing. All model parameters remain frozen. Static character timestamps are consumed as external input. The code keeps both the positive engineering result and the negative findings that constrain future work.

## Why this project exists

Dialect action words are sparse and unevenly distributed. A household may use only a handful of commands such as open the light, close the window, draw the curtain, or set the temperature. Pronunciations also vary across regions and speakers. Training a dedicated model for each home, dialect, and command does not scale.

Open-vocabulary keyword search offers another route. A device records a few examples of a target word and turns them into templates. At search time it locates those words directly in continuous speech. If early Audio-LLM latents are stable across speakers, adding a new word can avoid training. That hypothesis needs measurement.

Stage 2 exposes a second problem. The acoustic grid is about 80 ms per frame, while the external character reading grid is about 40 ms per frame. A direct conversion between the grids selects the wrong context boundary. The candidate first character then appears in the language-model prefix, so the model is asked to predict a token it has just seen. The error systematically lowers the score of a correctly localized candidate and is difficult to see from aggregate accuracy alone.

The project asks three questions.

1. Which frozen Audio-LLM layer gives the best cross-speaker keyword retrieval. An early acoustic layer should separate different words while keeping multiple takes of the same word close together.
2. How should the heterogeneous time grids be aligned. How large is prefix pollution under naive truncation, and can a training-free grid snap remove it.
3. Does the acoustic temporal shape of a keyword complement the language-model probability. A shape score may rank candidates on its own and still fail after fusion with the LM score.

## Key findings

| Finding | Main evidence | Consequence |
| --- | --- | --- |
| Early acoustic retrieval is strong | `k=8` plus enrollment mean centering reaches 0.9467 Top-20 pool recall | A frozen latent space can support high-recall coarse localization |
| Grid alignment is a major error source | Prefix first-character pollution is 35.23% for a naive cut and 3.39% for grid snapping | Alignment matters more than most parameter tuning |
| Temporal shape is diagnostic, not deployable | DTW is positive on an LM-only diagnostic baseline and negative on the deployable fusion baseline | Shape tie-breaking stays disabled |
| Low-VRAM extraction is feasible | NF4 4-bit loading exported six layers on an RTX 4060 Laptop | The prototype can be reproduced on a small GPU |

The third row is easy to misread. Temporal shape carries real information, and its standalone ranking is far above chance. The gain disappears after the full acoustic plus LM fusion, and speaker-disjoint cross-validation shows a stable loss. The project keeps this negative result because it directly constrains later multimodal fusion designs.

## Method

<p align="center">
  <img src="docs/assets/architecture_poster.png" alt="Two-stage pipeline: acoustic latent localization, grid-snapped cut points, and frozen language-model reranking" />
</p>
<p align="center"><sub>The mascot marks the candidates first; the language model checks the prefix second. Localization handles recall, and reranking orders the pool.</sub></p>

```mermaid
flowchart LR
    A[Dialect audio] --> B[Frozen Audio-LLM]
    B --> C["llm2 states<br/>3584-d, 80 ms"]
    D[Enrollment clips] --> E["Mean word templates<br/>L2 normalized"]
    E --> F["Mean sliding window<br/>k=8, cosine"]
    C --> F
    F --> G["Top-20 candidate pool<br/>recall 0.9467"]
    H["Static character reading<br/>40 ms grid"] --> I[Grid-snapped boundary]
    G --> I
    I --> J["Frozen Audio-LLM<br/>teacher forcing"]
    J --> K["Fusion score<br/>Z_acoustic + 1.5 Z_LM"]
    K --> L[Keyword and onset]
```

### Stage 1, acoustic latent localization

The frozen Audio-LLM produces audio-range hidden states at decoder block 2. Each frame has 3584 dimensions and advances by about 80 ms. For a word, each enrollment take is averaged over its keyword interval. The take means are averaged into a centroid and L2 normalized to form the template $q_w$.

Search computes cosine similarity for a mean window of length $k$.

$$
c_{i,w}=\frac{\bar h_{i,i+k-1}}{\lVert \bar h_{i,i+k-1}\rVert_2}\cdot q_w
$$

The word score is the maximum window response, and the corresponding window start is the coarse onset. The validated real-corpus setting uses `k=8` with enrollment mean centering. The best 20 words enter Stage 2. Pool recall is the hard ceiling for any later single-candidate decision.

### Stage 2, prefix-aligned language-model reranking

Each candidate is placed after the instruction, audio, and a left context. The frozen model returns the mean token log-probability.

$$
S_{\mathrm{LM}}(w)=\frac{1}{|\mathrm{tok}(w)|}\sum_t \log P\bigl(\mathrm{tok}(w)_t \mid \text{instruction},\text{audio},\text{prefix}_w,\mathrm{tok}(w)_{<t}\bigr)
$$

The default C arm finds the nearest occurrence of the candidate first character on the 40 ms reading grid and snaps the acoustic window start to that point. The prefix therefore ends before the first character. Acoustic and LM scores are normalized inside each candidate pool and fused with a fixed weight of 1.5.

$$
\mathrm{Score}=Z_{\mathrm{acoustic}}+1.5\,Z_{\mathrm{LM}}
$$

## Experimental protocol

The main experiment uses Wu dialect data derived from WenetSpeech-Wu-Bench. It contains 2031 utterances, 3244 keyword occurrences, and 293 words. Features come from a frozen Step-Audio-2-mini checkpoint at `llm2`, with 3584 dimensions per frame and an 80 ms frame shift. The candidate pool is fixed at Top 20. Stage-2 fusion uses a fixed weight of 1.5.

Results report occurrence Top-1 and per-word Top-1. Occurrence Top-1 weights every keyword occurrence equally. Per-word Top-1 averages within a word first, then across words. Cross-validation groups by utterance or by the `rec_sXX` session field. Window and calibration choices are made on training folds only. The `rec_sXX` grouping is a heuristic speaker proxy, not an official speaker annotation.

Synthetic fixtures verify code paths and never serve as accuracy evidence. The raw records for all reported numbers are in `docs/REAL_CORPUS_RESULTS.md` and `docs/layer_sweep.md`.

## Comparisons and results

### Stage 1, window, calibration, and template structure

| Configuration | Occurrence Top-1 | Word Top-1 | Top-20 recall |
| --- | ---: | ---: | ---: |
| `k=5` mean template, recomputed historical baseline | 0.4371 | 0.4821 | 0.9251 |
| `k=8` mean template | 0.4732 | 0.5523 | 0.9343 |
| `k=8` plus mean centering | **0.4741** | **0.5451** | **0.9467** |
| `k=8` plus ABTT | 0.3952 | 0.4506 | 0.8634 |
| `k=8` plus PCA whitening | 0.4319 | 0.4908 | 0.9097 |
| Duration-adaptive window | 0.4658 | 0.5412 | 0.9340 |
| 2-means multi-prototype and multi-scale | 0.4498 | 0.5142 | 0.9168 |

`k=8` plus mean centering is the only configuration in this set that gives a stable pool-recall gain. The speaker-disjoint nested CV gain is **3.699 percentage points** for occurrence Top-1, with a 95% CI of **[+2.796, +4.607]**. All five folds select this configuration. Utterance-grouped five-fold CV gives 3.514 points, with a 95% CI of [+2.624, +4.419].

ABTT, PCA whitening, duration adaptation, multi-prototype templates, and multi-scale search do not exceed this setting. The engineering priority shifts away from template complexity and toward the time grid and context alignment.

### Layer selection, small probe and full-split follow-up

The layer probe ran on an RTX 4060 Laptop with NF4 4-bit loading, double quantization, and bf16 compute. It contains 24 keyword occurrences and 64 multi-speaker enrollment templates. A layer number denotes a decoder block output, so layer 0 is the output of the first block.

| Layer | Top-20 recall | Inter-template cosine | Intra-template cosine | Inter/intra |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000000 | 0.446985 | 0.925754 | **0.482834** |
| 1 | 0.958333 | 0.513000 | 0.933310 | 0.549657 |
| 2 | 0.958333 | 0.511176 | 0.934279 | 0.547134 |
| 3 | 0.958333 | 0.528191 | 0.936630 | 0.563927 |
| 4 | 0.958333 | 0.549397 | 0.939740 | 0.584627 |
| 8 | 1.000000 | 0.602525 | 0.945296 | 0.637394 |

Layer 0 has full recall and the lowest inter/intra ratio on the small probe. Layer 8 also has full recall but the weakest template separation. Layer 2 loses one of the 24 occurrences and ranks behind layer 0 on both separability measures.

The full split later compared available late-fusion views.

| Layer view | Top-20 recall |
| --- | ---: |
| `llm1` | 0.9337 |
| `llm2` | **0.9467** |
| `llm1:0.5 + llm2:0.5` | 0.9414 |
| `llm1:0.3 + llm2:0.7` | 0.9436 |
| `llm2:0.7 + llm6:0.3` | 0.9396 |
| `llm1:0.5 + llm2:0.3 + llm6:0.2` | 0.9371 |

No layer-fusion view improves on `llm2`, which remains the default. The small-probe advantage of layer 0 has not yet been tested as a full-split layer choice. Changing the default requires a full extraction and a controlled evaluation on the same candidate pool. Current evidence supports keeping `llm2`.

### Seven-arm ablation, time-grid alignment

All seven arms share the same candidate pool, frozen model, and fusion weight. Only the left context changes. The table comes from an earlier public working point with a 0.9266 Top-20 pool recall. It is used for attribution of the alignment effect and should not be numerically mixed with the current 0.9467 baseline.

<p align="center">
  <img src="docs/assets/grid_alignment_spotlight.png" alt="Naive versus grid-snapped cuts across 80 ms acoustic frames and 40 ms character readings" />
</p>
<p align="center"><sub>Both clocks describe the same boundary. Snapping moves the cut point only; the frozen model and fusion weight stay unchanged.</sub></p>

| Arm | Left-context rule | Word Top-1 |
| --- | --- | ---: |
| A | No left context | 0.2233 |
| P | Cut the 40 ms reading at the 80 ms window start | 0.3591 |
| X | Use another utterance reading with the same cut point | 0.3597 |
| S | Use the acoustic Top-1 snap point for the whole utterance | 0.5457 |
| T | Truncate the gold transcript to the same length | 0.5763 |
| C | Snap to the candidate first-character grid position | **0.6172** |
| O | Use the gold sentence cut point | 0.6444 |

Arm P is almost identical to the wrong-sentence control X. The loss comes from misalignment, and arbitrary context does not repair it. Arm C exceeds X by **25.8 percentage points**. Arms T and O require transcripts or gold timestamps and are diagnostic ceilings only.

The pollution audit gives a direct explanation.

| Pollution measure | P, naive cut | C, grid snap |
| --- | ---: | ---: |
| Prefix contains the candidate first character | 0.3523 | 0.0339 |
| Prefix ends on the candidate first character | 0.3419 | 0.0173 |

About one third of P candidates see their own first character in the prefix. The causal model is then asked to predict a token that has already appeared, and a correctly localized candidate is penalized. Grid snapping reduces this form of pollution to about 3%.

### Stage 2, replication and extension branches

All 2031 utterances use the same candidate pool and fusion weight. Historical C and recomputed C differ by about half a point because of the runner. The project reports that gap separately instead of treating it as a gain. The KV-cache path has a separate value-parity audit.

| Branch | Occurrence Top-1 | Word Top-1 | Change vs recomputed C |
| --- | ---: | ---: | ---: |
| Historical public C | 0.5370 | 0.6266 | +0.59 point |
| Recomputed C with KV cache | 0.5311 | 0.6182 | 0 |
| C plus phonetic soft snap | **0.5358** | **0.6286** | **+0.462 point** |
| C plus Top-2 NMS starts | 0.5351 | 0.6268 | +0.401 point |
| C plus PMI, `gamma=0.25` | 0.5342 | 0.6218 | +0.309 point |
| C plus PMI, `gamma=1.00` | 0.5216 | 0.6026 | -0.956 point |

Phonetic soft snapping gains **0.462 point** over recomputed C, with a 95% CI of **[+0.061, +0.897]**. Against historical public C it is -0.123 point, with a 95% CI of [-0.582, +0.367], which is within noise. The branch remains optional, and C stays the default.

The 95% CI for Top-2 NMS is [-0.091, +0.828], so the gain is not stable. Fixed `gamma=1` PMI loses 0.956 point, with a 95% CI of [-1.743, -0.186]. PMI is not in the default path.

### Temporal shape, diagnostic signal and fusion failure

Shape arms convert candidate-window dynamics into a similarity score. Used alone, first-difference `d1` reaches 0.2132 occurrence Top-1, linear resampling reaches 0.4069, and subsequence DTW reaches 0.3633. DTW is not the strongest standalone arm, but it carries complementary information when the language model is uncertain.

| Base score | Shape signal | Grouping | Occurrence Top-1 change | 95% CI |
| --- | --- | --- | ---: | --- |
| LM-only | DTW gate | speaker 5-fold | **+0.832 point** | [+0.264, +1.449] |
| `acoustic + 1.5 LM` | DTW gate only | utterance 5-fold | -0.100 point | [-0.373, +0.166] |
| `acoustic + 1.5 LM` | All shape arms | utterance 5-fold | -0.333 point | [-0.605, -0.098] |
| `acoustic + 1.5 LM` | All shape arms | speaker 5-fold | **-0.399 point** | [-0.741, -0.068] |

DTW has a stable gain on the LM-only diagnostic baseline. After the acoustic term returns, the shape signal gives no net gain, and speaker-disjoint five-fold CV shows a stable loss. The result indicates that local acoustic dynamics and global semantic probability occupy different distributions. A simple linear combination cannot cash in the standalone ranking ability.

Shape tie-breaking is therefore an audit operator and remains disabled by default. Any later attempt to use temporal information needs a new conditioning mechanism and a separate held-out evaluation.

### KV cache and low-cost inference

The cached path encodes the audio prefix once and decodes candidates incrementally from a shared prefix. On an H20 with 20 candidates, fp16, and five warm runs, the median speed-up is **1.30x to 2.14x**. The gain is modest because candidate strings are usually one to four tokens and the baseline already batches equal-length candidates.

In a 200-utterance parity audit, the largest score difference is 0.0391 to 0.0469 nats and the minimum Spearman correlation is 0.997. C-arm Top-1 agrees on 199 of 200 utterances, and all four reported metrics are identical. The project does not claim a 5x to 10x speed-up or an 80% memory reduction.

Feature export can write layers `0,1,2,3,4,8` in one forward pass. A 5 GB class GPU can run base feature extraction with NF4 4-bit loading, double quantization, and fp16 compute. The public 4-bit runner targets the base extractor. Adapter merging uses the unquantized model path.

## What the prototype is useful for

The first contribution is a reproducible baseline. Open-vocabulary dialect KWS often hides enrollment, localization, LM reranking, and threshold calibration inside one black box. This repository separates the stages and reports pool recall, within-pool Top-1, prefix pollution, and cache parity independently. A researcher can replace one stage while keeping the other controls.

The second contribution is an alignment diagnostic. Heterogeneous time grids are common in cascaded systems. The project provides a measurable pollution criterion and a set of arms that change only the left context. The 25.8-point gap between C and X shows that context placement can decide whether reranking works.

The third contribution is a reusable negative result. PMI, Top-2 NMS, whitening, multi-prototype templates, layer fusion, and temporal shape were all tested on the real corpus. They stay out of the default path because of confidence intervals, pool-recall loss, or a drop on the deployable baseline. Later work can avoid those directions or design a targeted experiment around the failure mechanism.

As an engineering prototype, the system shows a local enrollment and search path. A device needs only a few target-word recordings. Templates and latents can be produced locally, and the model does not update. It is a research baseline for dialect commands in smart-home and assisted-living devices. Production wake-word use still needs far-field robustness, false-accept measurement, real-time latency, power, rejection behavior, and independent evaluation across more dialects.

## Known limits

- The main experiment covers one 2031-utterance Wu dialect corpus and does not establish cross-dialect generalization.
- Speaker groups come from the heuristic `rec_sXX` field rather than verified official speaker IDs.
- The 4-bit layer sweep contains only 24 occurrences. It proposes candidate layers and does not replace a full evaluation.
- Phonetic snapping is significant against recomputed C but has not beaten historical public C.
- PMI, Top-2 NMS, ABTT, PCA whitening, multi-prototype templates, and layer fusion do not show a stable real-corpus net gain.
- Shape tie-breaking is negative on the deployable fusion baseline and is disabled by default.
- This repository is a research prototype, not a production wake-word system.

## Repository map

| Path | Responsibility |
| --- | --- |
| `locator/` | Stage 1 templates, sliding-window cosine search, pools, and optional calibration |
| `reranker/` | Stage 2 reading-grid snapping, left-context construction, and teacher-forcing scores |
| `analysis/` | Prefix pollution audit, layer comparison, shape operators, and non-decision tools |
| `integration/` | Step-Audio-2 feature export and low-VRAM loading adapters |
| `scripts/` | Real-corpus ablations, cross-validation, layer fusion, and cache benchmarks |
| `data/` | Public schemas, neutral examples, and deterministic synthetic fixtures |
| `docs/` | Architecture, reproduction checklist, model notes, and real-corpus results |
| `tests/` | Unit tests for localization, alignment, scoring, shape, and model-loading guards |

## Quick start

```bash
git clone https://github.com/leeoisaboy/audiollm-latent-kws.git
cd audiollm-latent-kws
python -m venv .venv
python -m pip install -e ".[dev]"
python scripts/make_fixtures.py --output data/synthetic
```

Run the complete flow on synthetic fixtures. These scores only verify that the interfaces connect.

```bash
python cli.py locate \
  --templates data/synthetic/templates.jsonl \
  --enroll-dir data/synthetic/enroll \
  --search-manifest data/synthetic/search_manifest.jsonl \
  --search-dir data/synthetic/search \
  --latent-width 16 \
  --window 8 \
  --whitening center \
  --pool-top 20 \
  --include-gold \
  --output reports/pool.jsonl

python cli.py evaluate \
  --pool reports/pool.jsonl \
  --pool-top 20 \
  --output reports/stage1.json

python cli.py rerank \
  --pool reports/pool.jsonl \
  --reading data/synthetic/readings.jsonl \
  --wav-dir data/synthetic \
  --scorer examples.dummy_scorer:create_scorer \
  --variants A,P,X,S,C \
  --null-shift 1 \
  --output reports/lm.jsonl

python cli.py fuse \
  --pool reports/pool.jsonl \
  --lm reports/lm.jsonl \
  --variant C \
  --lambda-lm 1.5 \
  --output reports/metrics.json

python cli.py audit \
  --pool reports/pool.jsonl \
  --lm reports/lm.jsonl \
  --reading data/synthetic/readings.jsonl \
  --null-shift 1 \
  --output reports/prefix_audit.json
```

Real feature extraction needs a private audio manifest and a frozen model. The following command exports six layers in one pass.

Install the optional runtime first with `python -m pip install -e ".[stepaudio2]"`.

```bash
python integration/step_audio2.py \
  --model /path/to/Step-Audio-2-mini \
  --manifest /private/path/all_clips.jsonl \
  --enroll-manifest /private/path/enroll_manifest.jsonl \
  --search-manifest /private/path/search_manifest.jsonl \
  --output-root features \
  --layers 0,1,2,3,4,8 \
  --quantization 4bit \
  --dtype float16 \
  --min-duration 0.32 \
  --max-duration 12.0
```

Compare the exported layers with the command below.

```bash
python cli.py compare-layers \
  --features-root features \
  --templates data/reference/templates.jsonl \
  --search-manifest data/dialect_kws_testset/manifest.jsonl \
  --layers 0,1,2,3,4,8 \
  --latent-width 3584 \
  --pool-top 20 \
  --output reports/layer_comparison.json
```

Reranking requires a private runner that implements `AudioLMScorer`. The public repository contains no model weights, restricted audio, speaker identities, or recognizer output.

## Data and models

Public examples use neutral household commands such as turn on the light, close the window, draw the curtain, and set the temperature. Real audio, enrollment manifests, character timestamps, derived latents, and evaluation reports belong in a private environment and remain subject to their source licenses.

The Wu resource index is [WenetSpeech-Wu](https://huggingface.co/datasets/ASLP-lab/WenetSpeech-Wu). Public model and adapter loading constraints are documented in [docs/MODELS.md](docs/MODELS.md). The repository distributes neither weights nor corpus audio.

## Reproduction and tests

```bash
python -m compileall -q .
python -m pytest -q
```

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the full checklist, [docs/REAL_CORPUS_RESULTS.md](docs/REAL_CORPUS_RESULTS.md) for real-corpus numbers, and [docs/layer_sweep.md](docs/layer_sweep.md) for the layer sweep.

The code is released under Apache-2.0. Data, models, and derived features retain their original licenses. The example files are not redistributable corpus data.

**The most useful output of this project is a frozen Audio-LLM keyword-search baseline that explains both its successes and its failures.**
