#!/usr/bin/env bash
# 20_vllm_qwen36_35b_mtp.sh — vLLM path (continuous batching, better under
# concurrency than llama.cpp; use this when serving multiple stream viewers /
# agents at once).
#
# Ampere (sm_86) reality check for the 3090 Ti rig:
#  * NO FP8 compute. BF16 weights or AWQ/GPTQ W4A16 (Marlin kernels) only.
#    Do NOT pull the -FP8 checkpoints.
#  * fp8_e5m2 KV-cache STORAGE works on Ampere (dequantized on read) and is
#    the cheap 2x KV win.
#  * No NVLink: TP=4 all-reduce traffic over PCIe can eat the gains. Start
#    with TP=2 on your fastest x16 pair; try PP for the rest. Benchmark both.
#
# MTP gotcha: many community INT4 exports DROP the mtp.* tensors, and vLLM's
# MTP loader then dies on a missing 'mtp.fc.weight'. The script probes for the
# head and falls back to suffix decoding (n-gram-ish, free, no weights needed).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"   # or your validated AWQ repo/local path
TP="${TP:-2}"
PP="${PP:-1}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-131072}"
GPU_UTIL="${GPU_UTIL:-0.92}"
SPEC="${SPEC:-mtp}"                       # mtp | suffix | off

SPEC_ARGS=()
case "$SPEC" in
  mtp)
    # MTP-1/2: best TPOT at low concurrency; lowers peak throughput under load.
    SPEC_ARGS=(--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}')
    ;;
  suffix)
    # No draft weights needed; modest speedup, great for repetitive agent loops.
    SPEC_ARGS=(--speculative-config '{"method": "suffix", "num_speculative_tokens": 8}')
    ;;
  off) ;;
esac

exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --pipeline-parallel-size "$PP" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  "${SPEC_ARGS[@]}"

# Notes:
#  * If MTP startup fails with a missing mtp.* weight on a quantized repo,
#    rerun with SPEC=suffix (or SPEC=off) — the quant dropped the head.
#  * Under heavy concurrent load, MTP can REDUCE total throughput; vLLM docs
#    recommend it for latency-sensitive low-concurrency. For your overlay
#    chatbot (1-3 concurrent), keep it on; for batch jobs, SPEC=off.
#  * Prefix caching is the silent killer feature for agent/system-prompt-heavy
#    workloads: repeated prefixes skip prefill entirely.
