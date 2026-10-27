# Public model and data resources

This repository ships no weights and performs no fine-tuning. A private runner
must keep the frozen model frozen and expose only the `AudioLMScorer` interface
used by `cli.py rerank`.

## Wu speech resources

- Dataset index: [WenetSpeech-Wu](https://huggingface.co/datasets/ASLP-lab/WenetSpeech-Wu).
- Speech-understanding resource card: [WenetSpeech-Wu Speech Understanding](https://huggingface.co/ASLP-lab/WenetSpeech-Wu-Speech-Understanding).

Use the published revision, license, and split information in a private
manifest. The open repository should contain only schemas and deterministic
synthetic fixtures.

## Public adapter names

The resource card exposes separately published Wu-dialect adapters:

- `stepaudio2-ASR/adapter_model.safetensors`
- `stepaudio2-Und/adapter_model.safetensors`

The repository does not bundle or invoke an external recognizer. Convert any
licensed decoder output to the neutral `character_reading.jsonl` schema in a
private environment, then pass that static file to the public CLI.

## Step-Audio2 runner contract

The public Step-Audio2 card documents the base model
`stepfun-ai/Step-Audio-2-mini`, one of the adapters above, `ms-swift`, and
`transformers==4.53.3`. A runner should:

1. load the base model with remote-code trust and keep every parameter in eval
   mode;
2. attach and merge the selected PEFT adapter read-only;
3. decode mono 16 kHz audio into the model's 128-bin log-mel front end;
4. expand the audio marker into the model's audio-token block;
5. expose selected audio hidden states to the locator as float16
   `[frames, 3584]`;
6. implement full-sequence teacher forcing for `AudioLMScorer.score`; keep any
   KV-cache shortcut private until it passes an exact parity check;
7. return fp32 candidate-token log probabilities.

For a local extraction card with about 5 GB usable VRAM, prefer 4-bit NF4 with
double quantization and fp16 compute. Keep the weights frozen. Quantized loading
is not compatible with the public adapter merge path; use it only with the base
extractor unless a separate read-only PEFT runner is supplied.

Do not commit the runner if it contains local paths, credentials, proprietary
model wrappers, or evaluation metadata. Publish it only after replacing those
details with configuration values supplied by the operator.
