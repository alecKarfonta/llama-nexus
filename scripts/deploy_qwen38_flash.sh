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
# [TIME] | INFO | SCRIPTS.DEPLOY_QWEN38_FLASH | VRAM/offload math for UD-Q4_K_XL (~111 GB):
#   - ~33 GB n-gram (PLE) table lives in HOST RAM via mmap — do NOT pass --no-mmap/--mlock
#   - ~78 GB GPU weights + <1 GB KV (q8_0, 64K ctx — only 12/48 layers hold KV)
#   - Option A (default): GPUs 0,1,3 (~71 GB free) + n_cpu_moe=4 (~6 GB experts to CPU)
#   - Option B: free GPU 2 (stop moss-tts) then use "all" with n_cpu_moe=0
#   - If a GPU OOMs during load: raise n_cpu_moe by 2 per ~3 GB shortfall
#   - Troubleshooting (day-0 PR): garbage output -> lower gpu_layers; assert at load ->
#     env LLAMA_ATTN_ROT_DISABLE=1 in the container (see PR #27742 discussion)
GPU_SET="${GPU_SET:-0,1,3}"
CPU_MOE="${CPU_MOE:-0}"
CTX="${CTX:-65536}"
GPU_LAYERS_CFG="${GPU_LAYERS_CFG:-999}"
# [TIME] | INFO | SCRIPTS.DEPLOY_QWEN38_FLASH | --fit cannot combine with -ot, so balance manually.
# Default: PLE table to host RAM + every other MoE layer's experts to CPU (24 of 48
# layers, ~30 GB) leaving ~48 GB for 3 GPUs + mmproj vision tower + KV. Use [.] not \. (JSON).
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-per_layer_token_embd=CPU,blk[.](3|7|11|15|19|23|27|31|35|39|43|47|2|6|10|14|18|22|26|30|34|38|42|46)[.].*_exps[.]weight=CPU}"
MTP_ENABLED="${MTP_ENABLED:-false}"
MTP_DRAFT_N_MAX="${MTP_DRAFT_N_MAX:-3}"
# [TIME] | INFO | SCRIPTS.DEPLOY_QWEN38_FLASH | MTP sidecar: trained draft head from
# dzannotti/Qwen3.8-Flash-Next-MTP-GGUF (needs the patched b10631-qwen4exp-mtp build).
# README-tuned: p-min 0.75 matters more than depth; n-max 8 + p-min 0.3 measured slower.
MTP_DRAFT_MODEL="${MTP_DRAFT_MODEL:-}"

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

echo "-- pushing deploy config (gpus=${GPU_SET}, n_cpu_moe=${CPU_MOE}, ctx=${CTX}, gpu_layers=${GPU_LAYERS_CFG}, mtp=${MTP_ENABLED}, draft=${MTP_DRAFT_MODEL:-none})"
curl -s -X PUT "${BACKEND_URL}/api/v1/service/config" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d @- <<JSON | python3 -m json.tool | head -20
{
    "model": {
        "name": "Qwen3.8-Flash-Next",
        "variant": "UD-Q4_K_XL",
        "mmproj": "mmproj-F16.gguf",
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
        "batch_size": 1024,
        "ubatch_size": 512,
        "parallel_slots": 1,
        "split_mode": "layer",
        "tensor_split": null,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0"
    },
    "execution": {"mode": "gpu", "cuda_devices": "${GPU_SET}"},
    "server": {"alias": "${ALIAS:-qwen38-flash}"},
    "mtp": {
        "enabled": ${MTP_ENABLED},
        "draft_n_max": ${MTP_DRAFT_N_MAX},
        "draft_n_min": 0,
        "draft_p_min": 0.75
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
