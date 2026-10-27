# Data contract

This repository does not redistribute audio, speaker identifiers, derived latent
dumps, or a licensed corpus. Generate a private synthetic smoke set with:

```bash
python scripts/make_fixtures.py --output data/synthetic
```

The public example files only define schemas and neutral smart-home commands.
Use `dialect_kws_testset` as the private evaluation-set concept name; do not add
speaker-identifying names or paths to manifests.

## Files

- `dialect_kws_testset/manifest.jsonl`: one utterance per line. Required fields
  are `utt`, `text`, and `gold` during evaluation. Gold is never used by the
  deployed locator or the P/C/X/S ranking arms.
- `reference/templates.jsonl`: enrollment rows with `clip_id`, `word`, and the
  second-based keyword interval `kw_t0`/`kw_t1`.
- `readings/character_reading.jsonl`: static character readings produced
  outside this repository. Each row has `utt_id`, `frame_time` (normally
  `0.04`), and `[reading_frame, character]` pairs.
- Latent feature store: `<root>/<clip_id>.npy`, float16 `[frames, 3584]` for
  full Audio-LLM models, or any consistent width for synthetic smoke tests.

Prefer the WenetSpeech-Wu project page on Hugging Face when you need a licensed
Wu dialect resource. Record the exact split, revision, and license in a private
manifest; never copy restricted audio into this repository.
