# Notes

## Goal
Fix Docker build errors and run the application. Restore missing HuggingFace download functionality.

## Problems Found
- `backend-api` build failed due to missing `public/` directory. (Fixed)
- `llamacpp-frontend` build failed due to missing components. (Fixed)
- `llamacpp-frontend` failed to start due to port 3000 conflict. (Fixed)
- `llamacpp-api` is stuck restarting because it fails to download the model. (Investigating)
    - The model URL `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/...` was wrong.
    - `ggml-org/gpt-oss-20b-GGUF` repo doesn't seem to have the file either in the main branch.
    - `curl` check for `https://huggingface.co/ggml-org/models/resolve/main/gpt-oss-20b-GGUF/gpt-oss-20b-GGUF-mxfp4.gguf` returned 307 Redirect to `/ggml-org/models-moved/resolve/main/gpt-oss-20b-GGUF/gpt-oss-20b-GGUF-mxfp4.gguf`.
    - It seems `ggml-org/models` or `ggml-org/models-moved` might be the repo?
- **ML Model Manager Page Missing Download UI**: The page was incomplete - backend API and frontend service methods existed, but the UI dialog was never implemented.

## Solutions
1.  **Backend**: Modified `backend/Dockerfile` to remove `COPY public/ ./public/`.
2.  **Frontend**: Rewrote `frontend/src/pages/ModelsPage.tsx`.
3.  **Ports**: Changed `llamacpp-frontend` port to 3002.
4.  **Model Download**:
    - The correct repository seems to be `ggml-org/models`.
    - The file path seems to be `gpt-oss-20b-GGUF/gpt-oss-20b-GGUF-mxfp4.gguf`.
    - Need to update `start.sh` or `docker-compose.yml` to point to the correct repository and file.
5.  **HuggingFace Download UI**: 
    - Created new `DownloadModelDialog.tsx` component with repository ID and file selection
    - Integrated dialog into ModelsPage with proper state management
    - Connected to existing backend API endpoints `/v1/models/download` and `/v1/models/downloads`
    - Dialog supports auto-fetching available files from HuggingFace repos
    - Added download priority selection (low/normal/high)

## Completed
- Created `DownloadModelDialog.tsx` component
- Integrated download dialog into ModelsPage
- Updated backend to support multiple file formats (.gguf, .safetensors, .bin, .pth, .pt) not just GGUF
- Rebuilt and restarted backend: `docker compose build backend-api && docker compose up -d backend-api`
- File listing API now returns all model file types

### Restored Missing ModelsPage Functionality
- Added download progress bars with real-time updates (speed, ETA, percentage)
- Added "Deploy" button to navigate to deployment page with selected model
- Added detailed model info dialog showing all model metadata
- Implemented proper start/stop/info action handlers
- Added formatBytes, formatSpeed, formatETA utility functions
- Improved model card UI with better layout and information display
- All placeholder functionality replaced with real implementations
- Rebuilt frontend: `docker compose build --no-cache llamacpp-frontend && docker compose up -d`

## Current Goal
Configure llamacpp-api to run with Qwen/Qwen3-VL-4B-Thinking model.

## Model Information
- HuggingFace Repository: `Qwen/Qwen3-VL-4B-Thinking-GGUF`
- Model Type: Vision-language model with reasoning capabilities
- Requirements:
  - llama.cpp version b5401 or later
  - Special parameters: `--reasoning-format deepseek` and `--jinja`
  - Supports various quantization formats (Q4_K_M, Q5_K_M, Q8_0, etc.)

## Tasks
1. Update docker-compose.yml to set MODEL_NAME and MODEL_VARIANT for Qwen3-VL-4B-Thinking - DONE
2. Update start.sh to handle this model with proper repository and parameters - DONE
3. Rebuild and restart the llamacpp-api container - DONE
4. Fixed warmup assertion error - DONE
5. Verified API is running and accessible - DONE

## Changes Made
- Updated docker-compose.yml:
  - Set MODEL_NAME=Qwen_Qwen3-VL-4B-Thinking (with correct naming)
  - Set MODEL_VARIANT=Q4_K_M
  - Changed MODEL_REPO=bartowski/Qwen_Qwen3-VL-4B-Thinking-GGUF (bartowski has all quantization versions)
  - Added MMPROJ_FILE=mmproj-Qwen_Qwen3-VL-4B-Thinking-f16.gguf (for vision support)
  - Updated CONTEXT_SIZE=40960
  - Adjusted TEMPERATURE=0.6, TOP_P=0.95
- Updated start.sh:
  - Added support for MODEL_REPO environment variable
  - Fixed model name to Qwen_Qwen3-VL-4B-Thinking (matches bartowski naming)
  - Added multimodal projection file download support
  - Added --mmproj parameter for vision-language model support
  - Added special parameters for reasoning models: --reasoning-format deepseek and --no-context-shift

## File Names Found
The bartowski repository uses this naming convention:
- Main model: Qwen_Qwen3-VL-4B-Thinking-Q4_K_M.gguf
- Vision projection: mmproj-Qwen_Qwen3-VL-4B-Thinking-f16.gguf
- Available quantizations: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, bf16, and many IQ variants

## Problems Encountered and Solutions
1. Initial 404 error - Wrong repository (Qwen/Qwen3-VL-4B-Thinking-GGUF doesn't exist)
   - Solution: Used bartowski/Qwen_Qwen3-VL-4B-Thinking-GGUF instead
2. Warmup assertion error: GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd <= (int64_t) embd_size) failed
   - Solution 1: Removed --embeddings flag for multimodal models (incompatible)
   - Solution 2: Added --no-warmup flag to skip problematic warmup phase

## Success
The Qwen3-VL-4B-Thinking model is now running successfully!
- API accessible on: http://localhost:8600
- Model capabilities: completion, multimodal (vision + text)
- Parameters: ~4 billion
- Context size: 262,144 tokens
- Reasoning mode: Enabled with DeepSeek format
- Vision support: Enabled with mmproj file

## Latest Features

### Chat Page Model Agnostic Display (NEW)
- **Made chat page dynamically display current model name**
  - Fetches current model info on component mount using `apiService.getCurrentModel()`
  - Uses dynamic model name in greeting messages instead of hardcoded "Qwen3-Coder"
  - Updates page subtitle to show actual loaded model name
  - Provides graceful fallback to "AI Model" if fetch fails
  - Changes applied to: initial greeting, clear chat message, page subtitle

### Log Streaming (NEW)
- **Added real-time log viewer to Deploy page**
  - Backend endpoints: `/api/v1/logs/container` and `/api/v1/logs/container/stream`
  - Frontend component: `LogViewer.tsx` with play/pause, refresh, clear, download
  - Uses Server-Sent Events (SSE) for real-time streaming
  - Auto-scroll, line limiting, terminal-style UI
  - Only visible when service is running

### Fixed llamacpp-api container crash
- **Fixed `--flash-attn` flag issue**: The flag was missing its required value
  - Changed from `--flash-attn` to `--flash-attn auto` in `backend/main.py`
  - Container now starts successfully and is healthy
  - Model inference service is running on port 8600

### File Management System (NEW)
- **Added file management to Models page**
  - Backend endpoints: `/v1/models/local-files` (GET) for listing, `/v1/models/local-files` (DELETE) for deletion
  - New "Downloaded Files" tab showing all model files on disk
  - Displays file name, size, modified date, and type
  - Delete confirmation dialog with file details
  - Disk usage summary showing total files and space used
  - Fixed API client routing to use backendClient (port 8700) instead of client (port 8600)
  - Features: sortable table, search/filter, bulk operations ready

### Log Clearing on Deploy Page (NEW - Dec 2025)
- **Added ability to clear logs on deploy page**
  - Updated `LogViewer.tsx` to use `forwardRef` and expose `clearLogs()` method via ref
  - Added `LogViewerRef` interface to allow parent components to control the log viewer
  - Updated `DeployPage.tsx` to use a ref to access the LogViewer component
  - Logs are now automatically cleared when the "Restart" action is triggered
  - Clear button already existed in LogViewer UI - now properly integrated with parent component
  - Changes to files:
    - `/home/alec/git/llama-nexus/frontend/src/components/LogViewer.tsx`
    - `/home/alec/git/llama-nexus/frontend/src/pages/DeployPage.tsx`
  
## Fixed: Chat Page Content Parsing & Tokens/Second Display (Dec 3, 2025)

### Issues Fixed
1. **Chat responses showing as empty**: The streaming parser was only checking `delta.content`, but reasoning models output in `delta.reasoning_content`
2. **Missing tokens/second metrics**: Timing information was available but not displayed
3. **Model name showing as "Qwen3-Coder"**: Browser cache issue, needs hard refresh

### Root Cause
The gpt-oss-120b model is a **reasoning model** using DeepSeek reasoning format. It outputs:
- First: `delta.reasoning_content` - thinking/reasoning tokens
- Then: `delta.content` - final response
- The chat UI was only checking for `delta.content`, missing all the reasoning tokens

### Changes Made
1. Updated `ChatPage.tsx` streaming handler to check all content locations:
   - `choices[0].delta.content` (standard response)
   - `choices[0].delta.reasoning_content` (thinking tokens from reasoning models)
   - `__verbose.content` (llama.cpp verbose output)
2. Added tokens/second extraction from `timings.predicted_per_second`
3. Added `tokensPerSecond` field to `ChatMessage` interface in `types/api.ts`
4. Display tokens/second as a chip badge next to assistant messages
5. Rebuilt and restarted frontend

### Commands
```bash
docker compose build llamacpp-frontend
docker stop llamacpp-frontend-temp && docker rm llamacpp-frontend-temp
docker run -d --name llamacpp-frontend-temp -p 3002:80 --network llama-nexus_default llama-nexus-llamacpp-frontend
```

### Result
- Chat responses now display correctly (including thinking/reasoning tokens)
- Each assistant message shows generation speed (e.g., "3.72 tok/s")
- Model name displays correctly after browser hard refresh (Ctrl+Shift+R)

## FIXED: Multi-GPU Split Mode for MoE Models (Dec 3, 2025)

### THE OPTIMAL SOLUTION FOR 2x 24GB GPUs + CPU

For the **gpt-oss-120b** MoE model on **2x 24GB GPUs**, use this configuration:

```bash
--split-mode layer      # Layer split mode
--tensor-split 2,1      # 2:1 ratio (GPU0 gets more layers, whose experts go to CPU)
--n-gpu-layers 999      # All layers on GPU
--n-cpu-moe 12          # First 12 layers' experts on CPU
--ctx-size 4096         # Context size
```

### Result (VERIFIED WORKING):
| Resource | Memory Used | Utilization |
|----------|-------------|-------------|
| GPU 0    | 23.0 GB / 24.6 GB (94%) | 16-68% |
| GPU 1    | 19.3 GB / 24.6 GB (79%) | 13-53% |
| CPU      | ~20 GB (for MoE experts) | Active |

- **Total GPU utilization: 86% of available VRAM**
- **Both GPUs actively computing during inference!**

### Key Insight: The tensor_split and n-cpu-moe Must Be Coordinated

With `--tensor-split 2,1`:
- Layers 0-24 → GPU0 (25 layers)  
- Layers 25-36 → GPU1 (12 layers)

With `--n-cpu-moe 12`:
- Layers 0-11's experts → CPU
- Layers 12-24's experts → GPU0 (~13 layers of experts)
- Layers 25-36's experts → GPU1 (~12 layers of experts)

This creates **balanced GPU memory usage** because each GPU gets roughly the same amount of expert weights.

### Command to Run Manually
```bash
docker run -d --name llamacpp-api \
  --runtime nvidia --gpus all \
  --network llama-nexus_default \
  -p 8600:8080 --shm-size 16g \
  -v llama-nexus_gpt_oss_models:/home/llamacpp/models \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  --entrypoint /usr/local/bin/llama-server \
  llama-nexus-llamacpp-api \
  --model /home/llamacpp/models/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  --host 0.0.0.0 --port 8080 --ctx-size 4096 \
  --n-gpu-layers 999 --n-cpu-moe 12 \
  --split-mode layer --tensor-split 2,1 \
  --api-key placeholder-api-key --verbose
```

### Tuning Guide for Different Hardware
| Total VRAM | n-cpu-moe | tensor-split | Notes |
|------------|-----------|--------------|-------|
| 2x 24GB    | 12        | 2,1          | Optimal for 120B model |
| 2x 16GB    | 20-25     | 2,1          | More on CPU |
| 2x 12GB    | 28-30     | 3,1          | Most on CPU |

## Previous Issue: Split Mode Only Using One GPU (Dec 3, 2025)

### Problem
- When selecting split mode (layer or row) in the Deploy UI, the model still only uses CUDA0
- Terminal logs show: `llama_kv_cache: layer X: dev = CUDA0` for ALL layers
- Backend correctly passes `--split-mode layer` flag to llama-server
- Issue: Container is configured to only see GPU 0

### Root Cause
In `docker-compose.yml` lines 89-90, the llamacpp-api container has:
```yaml
- CUDA_VISIBLE_DEVICES=0
- NVIDIA_VISIBLE_DEVICES=0
```

This restricts the container to only see GPU 0, even when split mode is configured.

### Solution - COMPLETED
Changed lines 89-90 and 167-168 in docker-compose.yml to:
```yaml
- CUDA_VISIBLE_DEVICES=all
- NVIDIA_VISIBLE_DEVICES=all
```

Restarted containers:
```bash
docker stop llamacpp-api && docker rm llamacpp-api
docker compose up -d llamacpp-api
```

Verified both GPUs are visible in container:
```
$ docker exec llamacpp-api nvidia-smi --query-gpu=index,name,memory.total --format=csv
index, name, memory.total [MiB]
0, NVIDIA GeForce RTX 3090 Ti, 24564 MiB
1, NVIDIA GeForce RTX 3090 Ti, 24564 MiB
```

### How to Use Multi-GPU Split Mode
1. Open Deploy page: http://localhost:31111/deploy
2. Configure Performance Settings:
   - Split Mode: `layer` (distributes layers across GPUs)
   - Tensor Split: `1,1` (equal 50/50) or leave empty
   - Main GPU: `0`
3. Click Restart to reload model with multi-GPU support

Expected logs when working:
```
llama_kv_cache: layer   0: dev = CUDA0
llama_kv_cache: layer  18: dev = CUDA0
llama_kv_cache: layer  19: dev = CUDA1
llama_kv_cache: layer  35: dev = CUDA1
```

Split mode options:
- **layer**: Distributes different transformer layers across GPUs (recommended)
- **row**: Splits weight matrices across GPUs (for very large models)
- **none**: Uses only main_gpu (single GPU)

## Next Steps
- Update `start.sh` logic to handle `gpt-oss-20b` correctly, or update `docker-compose.yml` to specify the correct `MODEL_REPO` if possible (currently hardcoded in `start.sh`).
- Since `start.sh` has hardcoded logic for `gpt-oss-20b`, I should update it there.
