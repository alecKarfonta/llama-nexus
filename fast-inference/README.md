# fast-inference

Compute-efficient serving stack for the 4× 3090 Ti rig, built around the
hybrid GatedDeltaNet Qwen3.6 family + MTP speculative decoding — the
"hybrid + spec-decode + 4-bit + KV quant" recipe from the architecture survey,
implemented with what actually runs on Ampere today.

## The stack

| Layer | Choice | Why |
|---|---|---|
| Model | Qwen3.6-35B-A3B (MoE, 3B active) / 27B dense | Hybrid linear-attention: tiny KV, fast long-ctx decode |
| Weights | Q4_K_XL GGUF / AWQ W4A16 | Lossless-ish 4-bit; sm_86 has no FP8 compute |
| KV cache | q8_0 (llama.cpp) / fp8_e5m2 (vLLM) | ~2× KV memory back, negligible quality cost |
| Spec decode | Built-in MTP head (`--spec-type draft-mtp`) | 1.4–2.2× decode, no draft model to train/host |
| Reuse | `--cache-reuse` / `--enable-prefix-caching` | Free prefill skip for system-prompt-heavy agents |

## Order of operations

```bash
./00_preflight.sh          # verify llama.cpp >= b9180, vLLM >= 0.19, GPUs
./01_download_models.sh    # MTP-specific GGUFs (standard GGUFs lack the head!)
./10_llamacpp_qwen36_35b_mtp.sh            # serve (single GPU default)
cd bench && ./run_matrix.sh                # baseline vs MTP, NMAX sweep
```

## Things that will bite you

1. **Standard GGUFs / most INT4 exports drop the MTP head.** Only `*-MTP-GGUF`
   repos work with `--spec-type draft-mtp`; in vLLM, a quant missing `mtp.*`
   tensors fails at load. Fallback: `SPEC=suffix` (vLLM) or ngram spec (llama.cpp).
2. **MoE benefits less from MTP than dense.** 3B active params = cheap decode
   already. Expect bigger relative wins on the 27B. High-temperature chat
   lowers draft acceptance — benchmark with YOUR sampling settings.
3. **No NVLink:** single-GPU > layer-split > TP over PCIe, in that order,
   for the 35B-A3B. Use multi-GPU only for context, not speed.
4. **MTP stability:** known intermittent crash in early llama.cpp MTP builds —
   `systemd/llama-qwen36-mtp.service` auto-restarts it.
5. **MTP under high concurrency reduces total throughput** (vLLM docs). Keep it
   for the interactive overlay path; turn off for batch pipelines.
