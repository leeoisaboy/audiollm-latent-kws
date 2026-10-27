# Security and privacy

Do not place credentials, hostnames, API tokens, private audio paths, speaker
identities, or licensed corpora in issues or pull requests.

This repository accepts schemas, neutral examples, and deterministic synthetic
fixtures. Keep raw audio, enrollment manifests, character readings, derived
latents, and evaluation reports in a private environment.

Before publishing a derived repository, scan the working tree for secrets and
local paths.

```bash
rg -n -i --hidden --glob '!.git' \
  'password|passwd|api[_-]?key|token|secret|/home/|/mnt/|ssh '
```

The expected result is no match. Report security concerns privately to the
repository owner instead of opening a public issue with sensitive details.
