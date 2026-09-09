#!/usr/bin/env bash
# [TIME] | INFO | SCRIPTS.DEPLOY_QWEN38_FLASH | Deploy Qwen3.8-Flash-Next (qwen4exp) via backend API
# Usage: ./scripts/deploy_qwen38_flash.sh [BACKEND_URL]
#
# Prerequisites (already wired into this repo):
#   - llama.cpp image built from PR #27742 (qwen4exp) — see Dockerfile LLAMA_CPP_REF
#   - UD-Q4_K_XL shards + mmproj-F16.gguf in the gpt_oss_models volume
set -euo pipefail

BACKEND_URL="${1:-http://127.0.0.1:8700}"
API_KEY="${API_KEY:-placeholder-api-key}"

echo "== [TIME] | INFO | SCRIPTS.DEPLOY_QWEN38_FLASH | target backend: ${BACKEND_URL}"

# 1. Backend alive?
health=$(curl -s -o /dev/null -w '%{http_code}' "${BACKEND_URL}/api/v1/service/status" -H "Authorization: Bearer ${API_KEY}")
if [ "$health" != "200" ]; then
    echo "❌ backend not reachable (HTTP ${health}) — start it first: docker compose up -d backend-api"
    exit 1
fi

# 2. Model files present in the volume?
echo "-- checking model shards in gpt_oss_models volume"
if ! docker volume inspect llama-nexus_gpt_oss_models >/dev/null 2>&1; then
    echo "❌ models volume missing"
    exit 1
fi
docker run --rm -v llama-nexus_gpt_oss_models:/m alpine sh -c '
    count=$(ls /m/UD-Q4_K_XL/*.gguf 2>/dev/null | wc -l)
    mmproj=$(ls /m/mmproj-F16.gguf 2>/dev/null | wc -l)
    echo "   shards found: ${count}/4, mmproj: ${mmproj}/1"
    [ "$count" -eq 4 ] && [ "$mmproj" -eq 1 ]
' || { echo "❌ model files incomplete — download still running?"; exit 1; }

# 3. Current model in local list?
echo "-- verifying backend sees the model"
local_models=$(curl -s "${BACKEND_URL}/v1/models/local" -H "Authorization: Bearer ${API_KEY}")
echo "$local_models" | python3 -c '
import json, sys
data = json.load(sys.stdin)
items = data.get("data", []) if isinstance(data, dict) else data
match = [m for m in items if "Qwen3.8-Flash-Next" in json.dumps(m)]
if match:
    print("   ✅ found:", match[0].get("name"), match[0].get("variant", ""))
else:
    # Not fatal — the deploy API matches on name/variant config, not this listing
    print("   ⚠️  not in /v1/models/local yet (may still be scanning)")
'

# 4. Push deploy config (thinking-mode sampling from the model card)
#
# [TIME] | INFO | SCRIPTS.DEPLOY_QWEN38_FLASH | VRAM/offload math for UD-Q4_K_XL (111.3 GB):
#   - The rig holds 103.0 GB (4x24564 MiB), so offload is forced, not a tuning choice
#   - 28.8 GB n-gram (PLE) table lives in HOST RAM via mmap — do NOT pass --no-mmap/--mlock
#     It is a row lookup, not a matmul (~90 B per token per layer), so this costs ~nothing
#   - 82.5 GB other weights (77.0 GB experts) + <1 GB KV (q8_0, 64K ctx — only 12/48 hold KV)
#   - Default: all 4 GPUs, n_cpu_moe=0, PLE the only thing off-GPU
#   - Fallback if GPU 2 is busy (moss-tts): GPUs 0,1,3 needs experts on CPU too
#   - If a GPU OOMs during load: raise n_cpu_moe by 2 per ~3 GB shortfall
#   - Do NOT raise the power limit: measured 50-129 W of 350 W under load, clocks
#     pinned 1860-1935 MHz, no throttling. This rig is launch/latency-bound.
#   - Troubleshooting (day-0 PR): garbage output -> lower gpu_layers; assert at load ->
#     env LLAMA_ATTN_ROT_DISABLE=1 in the container (see PR #27742 discussion)
# [TIME] | WARN | SCRIPTS.DEPLOY_QWEN38_FLASH | These defaults are the tuned
# optimum recorded in notes.md 2026-08-30 round 2, NOT what the server happened
# to be running on 2026-09-05 — the live config had drifted off it on five axes
# (ubatch 256 vs 1024, p_min 0.75 vs 0.9, parallel 1 vs 2, ctx 196608 vs 131072,
# tensor_split 0.92,1.08,1.08,0.92 vs 0.85,1.1,1.0,1.05). Do not "sync" these
# back to a running server without re-benchmarking first.
GPU_SET="${GPU_SET:-0,1,2,3}"
CPU_MOE="${CPU_MOE:-0}"
# 131072 = 64k per slot at PARALLEL=2. Fits only with mmproj dropped.
CTX="${CTX:-131072}"
GPU_LAYERS_CFG="${GPU_LAYERS_CFG:-999}"
# Vision tower off: dropping its 0.9 GB is what let ubatch 1024 fit on GPU 0.
# Set MMPROJ=mmproj-F16.gguf to restore it, and drop UBATCH back to 512 if you do.
MMPROJ="${MMPROJ:-}"
# JSON wants an unquoted null when off, a quoted string when on.
if [ -n "$MMPROJ" ]; then MMPROJ_JSON="\"${MMPROJ}\""; else MMPROJ_JSON="null"; fi
# --parallel 2 works on the patched MTP build (the day-0 PR #27742 indexer-cache
# assert is gone). Single-stream TPS is unchanged, but 2 streams aggregate
# 65.6-68.7 vs 59.7 queued (+10-15%, wall -20%). N divides per-slot ctx.
PARALLEL="${PARALLEL:-2}"
# [TIME] | INFO | SCRIPTS.DEPLOY_QWEN38_FLASH | --fit cannot combine with -ot, so balance manually.
# All 4 GPUs hold 103.0 GB, so the 82.5 GB of non-PLE weights fit with NO expert
# offload — only the PLE leaves the GPU. Do not weaken this default.
# Falling back to 3 GPUs (moss-tts keeping GPU 2) needs experts on CPU too, which
# is ~39 GB over PCIe and much slower. For that case pass both:
#   GPU_SET=0,1,3 TENSOR_SPLIT=1,1,1 \
#   OVERRIDE_TENSOR='per_layer_token_embd=CPU,blk[.](3|7|11|15|19|23|27|31|35|39|43|47|2|6|10|14|18|22|26|30|34|38|42|46)[.].*_exps[.]weight=CPU'
# Use [.] not \. (JSON).
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-per_layer_token_embd=CPU}"
# Hand-balanced for 4x24 GB; evens VRAM to 22.0/22.0 and frees ~2.5 GB on GPU 0,
# which is what made ubatch 1024 fit. One entry per GPU in GPU_SET.
TENSOR_SPLIT="${TENSOR_SPLIT:-0.85,1.1,1.0,1.05}"
# ubatch 1024 measured cold prefill 905 -> 996 tok/s (+10%). It only fits with
# MMPROJ empty. batch_size was never recorded in the tuning log; 2048 is the
# llama.cpp default and must stay >= ubatch.
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-1024}"
MTP_ENABLED="${MTP_ENABLED:-true}"
MTP_DRAFT_N_MAX="${MTP_DRAFT_N_MAX:-16}"
# 0.9, not 0.75. Round 1 declared 0.75 the winner but had actually been running
# 0.9; the clean A/B afterwards gave STRUCT 95.9 at 0.9 vs 90.9 at 0.75.
MTP_DRAFT_P_MIN="${MTP_DRAFT_P_MIN:-0.9}"
# [TIME] | INFO | SCRIPTS.DEPLOY_QWEN38_FLASH | MTP sidecar: trained draft head from
# dzannotti/Qwen3.8-Flash-Next-MTP-GGUF (needs the patched b10631-qwen4exp-mtp build).
# README-tuned: p-min 0.75 matters more than depth; n-max 8 + p-min 0.3 measured slower.
# Default on: MTP is what the box actually runs (74.4% draft acceptance measured).
# Set MTP_DRAFT_MODEL= (empty) to deploy without speculative decoding.
MTP_DRAFT_MODEL="${MTP_DRAFT_MODEL-Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf}"

# [TIME] | INFO | SCRIPTS.DEPLOY_QWEN38_FLASH | speculative block injected only when
# MTP_DRAFT_MODEL is set (heredoc ${:+} can't contain literal braces — build in a var)
SPEC_BLOCK=""
if [ -n "${MTP_DRAFT_MODEL}" ]; then
    SPEC_BLOCK=",
    \"speculative\": {
        \"model_draft\": \"${MTP_DRAFT_MODEL}\",
        \"gpu_layers_draft\": 999
    }"
fi

echo "-- pushing deploy config (gpus=${GPU_SET}, n_cpu_moe=${CPU_MOE}, ctx=${CTX}, parallel=${PARALLEL}, batch=${BATCH}/${UBATCH}, split=${TENSOR_SPLIT}, mmproj=${MMPROJ:-off}, mtp=${MTP_ENABLED} n_max=${MTP_DRAFT_N_MAX} p_min=${MTP_DRAFT_P_MIN}, draft=${MTP_DRAFT_MODEL:-none})"
curl -s -X PUT "${BACKEND_URL}/api/v1/service/config" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d @- <<JSON | python3 -m json.tool | head -20
{
    "model": {
        "name": "Qwen3.8-Flash-Next",
        "variant": "UD-Q4_K_XL",
        "mmproj": ${MMPROJ_JSON},
        "context_size": ${CTX},
        "gpu_layers": ${GPU_LAYERS_CFG},
        "n_cpu_moe": ${CPU_MOE},
        "override_tensor": "${OVERRIDE_TENSOR}",
        "flash_attn": "on"
    },
    "sampling": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0
    },
    "performance": {
        "threads": 40,
        "batch_size": ${BATCH},
        "ubatch_size": ${UBATCH},
        "parallel_slots": ${PARALLEL},
        "split_mode": "layer",
        "tensor_split": "${TENSOR_SPLIT}",
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0"
    },
    "execution": {"mode": "gpu", "cuda_devices": "${GPU_SET}"},
    "server": {"alias": "${ALIAS:-qwen38-flash}"},
    "mtp": {
        "enabled": ${MTP_ENABLED},
        "draft_n_max": ${MTP_DRAFT_N_MAX},
        "draft_n_min": 0,
        "draft_p_min": ${MTP_DRAFT_P_MIN}
    }${SPEC_BLOCK}
}
JSON

# 5. Restart llama.cpp service with the new config
echo "-- restarting llamacpp service"
curl -s -X POST "${BACKEND_URL}/api/v1/service/restart" -H "Authorization: Bearer ${API_KEY}" | head -c 400; echo

echo
echo "== next: watch loading with:"
echo "   docker logs -f llamacpp-api"
echo "== then smoke-test:"
echo "   curl http://127.0.0.1:8600/v1/chat/completions -H 'Authorization: Bearer ${API_KEY}' \\"
echo "        -H 'Content-Type: application/json' -d '{\"model\":\"Qwen3.8-Flash-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}'"
