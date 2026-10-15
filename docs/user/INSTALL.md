# Installing Llama Nexus

Llama Nexus is an all-in-one local AI platform: LLM chat, speech-to-text,
text-to-speech, embeddings, and a vector database (RAG), all managed from one
web UI, fully local — your data never leaves your machine.

## Requirements

| Requirement | Minimum |
|---|---|
| OS | Linux, Windows 10/11 (WSL2), or macOS |
| RAM | 8 GB (16 GB+ recommended) |
| Disk | 10 GB free (models need more: 3–25 GB each) |
| Docker | Docker Engine or Docker Desktop with Compose v2 |
| GPU | Optional — NVIDIA GPUs accelerate inference; CPU-only works with small models |

**GPU support:** an NVIDIA GPU with recent drivers + nvidia-container-toolkit
gives fast inference. Without one, Llama Nexus runs in CPU mode — pick small
models (Qwen3-4B Q4_K_M or similar).

## Quick install

### Linux / macOS / WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/alecKarfonta/llama-nexus/master/install.sh | bash
```

Or clone and run:

```bash
git clone https://github.com/alecKarfonta/llama-nexus.git
cd llama-nexus
bash install.sh
```

### Windows

In PowerShell:

```powershell
irm https://raw.githubusercontent.com/alecKarfonta/llama-nexus/master/install.ps1 | iex
```

Or clone and run `powershell -ExecutionPolicy Bypass -File install.ps1`.

The installer:
1. Checks Docker and Compose v2
2. Detects your GPU (NVIDIA + toolkit = GPU tier with CUDA-matched images; otherwise CPU tier)
3. Writes a `.env` with a generated API key
4. Pulls prebuilt images (no compilation)
5. Starts the stack and prints your URLs

## First run

1. Open **http://localhost:3002**
2. The welcome wizard shows your hardware and recommends a model — download it
   with one click
3. Go to **Deploy** and click **Deploy** on the recommended configuration
4. Start chatting — enable **Voice mode** in Chat for the local speech loop,
   or **Documents** to ingest files into RAG

## Platform notes

| Platform | Tier | Notes |
|---|---|---|
| Linux + NVIDIA | Full | All features incl. vLLM/NInfer engines, GPU acceleration |
| Windows (WSL2 + Docker Desktop) | Full (GPU if NVIDIA) | Install Docker Desktop with WSL2 backend; GPU passthrough works with recent NVIDIA drivers |
| macOS (Apple Silicon/Intel) | CPU | Docker runs in a Linux VM — no GPU passthrough. Use small models (≤4B Q4). Native Metal support is on the roadmap |

## Manual install (no script)

```bash
git clone https://github.com/alecKarfonta/llama-nexus.git
cd llama-nexus
cp .env.example .env          # edit: set API_KEY
docker compose -f docker-compose.yml -f docker-compose.prebuilt.yml up -d
# NVIDIA GPU hosts additionally:
docker compose -f docker-compose.yml -f docker-compose.prebuilt.yml -f docker-compose.gpu.yml up -d
```

Prebuilt image tiers (set `LLAMACPP_DOCKER_IMAGE` in `.env`):
`ghcr.io/aleckarfonta/llama-nexus-llamacpp:cpu | :cuda-86 | :cuda-89 | :cuda-120`

## Building from source (advanced)

```bash
# Detect your GPU arch (86/89/120), then:
CUDA_ARCH=$(scripts/detect-cuda-arch.sh) docker compose build
docker compose up -d
```

Source builds compile llama.cpp with CUDA (30–60+ min). Most users should use
the prebuilt images.

## Updating

```bash
cd llama-nexus
git pull
docker compose -f docker-compose.yml -f docker-compose.prebuilt.yml pull
docker compose -f docker-compose.yml -f docker-compose.prebuilt.yml up -d
```

## Uninstall

```bash
docker compose -f docker-compose.yml -f docker-compose.prebuilt.yml down -v   # -v removes models/data
```

## Troubleshooting

- **`docker compose up` fails with port conflicts** — another service owns
  3002/8700/8600. Stop it or remap the host ports in `.env`.
- **GPU not detected in containers** — install
  [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  and restart Docker.
- **Model downloads fail (401)** — the model is gated; set `HUGGINGFACE_TOKEN`
  in `.env`.
- **Everything slow** — you're on CPU with a large model. Use the starter
  (Qwen3-4B Q4_K_M) or upgrade your GPU tier.
- **Mac fans at max** — expected for CPU inference; use smaller quantizations.

More help: [FAQ](FAQ.md) · [Security notes](SECURITY.md) ·
[Main README](../../README.md)
