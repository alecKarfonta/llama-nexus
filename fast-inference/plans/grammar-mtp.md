# Grammar + MTP Tool-Call Experiment

Hypothesis: lazy tool grammar (tools API) narrows valid tokens; MTP acceptance
rises on grammar-predictable positions; higher `--spec-draft-n-max` wins on
tool path (opposite of free chat).

## Run

```bash
cd fast-inference/bench && ./run_grammar_mtp.sh
```

Configs: `tools-off-mtp-off`, `tools-on-mtp-off`, `tools-on-mtp-n{2,4,6,8}`.

## Results (2026-06-08, 27B Q4_K_M MTP, GPUs 0+1, lazy tool grammar)

| Config | weather | overlay | command | MTP accept |
|--------|---------|---------|---------|------------|
| tools + MTP off (grammar only) | 27.7 | 27.3 | 27.7 | — |
| tools + MTP n2 | 54.4 | 56.1 | 54.5 | **100%** |
| tools + MTP n4 | 54.4 | 67.6 | 66.2 | **100%** |
| tools + MTP n6 | 65.8 | 64.4 | 54.1 | 90% |
| **tools + MTP n8** | **86.9** | **77.7** | **67.6** | 87.5% |

- Grammar-only tool path: ~28 tok/s (valid tool calls 100%)
- **MTP n8 + grammar: ~1.6–3.1× over grammar-only** (87 vs 28 tok/s)
- **Higher NMAX wins on tool path** (opposite of free-chat where n2 beat n4)
- Acceptance stays high (87–100%); wider drafts compensate when accept drops slightly

Recommended tool-call deploy: `--spec-type draft-mtp --spec-draft-n-max 8` with `tools` API.

## Not in this pass

- `LLAMA_LLGUIDANCE=ON` rebuild (jump-forward / 50μs masks)
- llama-nexus proxy / Deploy UI wiring
