# Hidden-layer sweep

This is the 2026-10-25 local RTX 4060 Laptop probe. It compares decoder
block outputs on the WenetSpeech-Wu-Bench-derived dialect-KWS probe: 24 gold
occurrences and 64 multi-take templates. Stage 1 uses mean templates and a
Top-20 pool. The layer number is the decoder block output, not the internal
Transformers hidden-state index.

The checkpoint was loaded with NF4 4-bit quantization, double quantization,
and bf16 compute. Each audio clip was exported once for layers 0, 1, 2, 3, 4,
and 8 with a 20 s duration limit.

| Layer | Top-20 pool recall | Inter-template cosine | Intra-template cosine | Inter/intra |
|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 0.446985 | 0.925754 | 0.482834 |
| 1 | 0.958333 | 0.513000 | 0.933310 | 0.549657 |
| 2 | 0.958333 | 0.511176 | 0.934279 | 0.547134 |
| 3 | 0.958333 | 0.528191 | 0.936630 | 0.563927 |
| 4 | 0.958333 | 0.549397 | 0.939740 | 0.584627 |
| 8 | 1.000000 | 0.602525 | 0.945296 | 0.637394 |

Layer 0 is the best layer on this probe: it reaches full Top-20 recall and has
the lowest inter/intra ratio. Layer 8 also reaches full recall but has the
weakest template separability. Layer 2, the historical Stage-1 setting, loses
one of 24 occurrences and ranks behind layer 0 on both metrics. The production
default should therefore not be changed on this small probe alone; use layer 0
as the current layer-sweep winner and rerun the sweep on the full split before
switching the locator default.

## Full-split follow-up (2026-10-27)

The 2031-utterance / 3244-occurrence split was then evaluated with the same
frozen checkpoint and k=8 mean templates. Late fusion was tested at the score
level, with enrollment-only mean centering:

| View | Top-20 pool recall |
|---|---:|
| `llm1` | 0.9337 |
| `llm2` | **0.9467** |
| `llm1:0.5 + llm2:0.5` | 0.9414 |
| `llm1:0.3 + llm2:0.7` | 0.9436 |
| `llm2:0.7 + llm6:0.3` | 0.9396 |
| `llm1:0.5 + llm2:0.3 + llm6:0.2` | 0.9371 |

`llm2` remains the production default. The small-probe preference for layer 0
did not transfer to a clear full-split late-fusion win, so reports should not
claim a layer-0 improvement without re-extracting and evaluating layer 0 on the
full corpus.
