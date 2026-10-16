# FAQ & Troubleshooting

## Install

**Docker says "port already allocated"**
Another service owns a Llama Nexus port (3002/8700/8600/8602/6333). Stop the
other service or remap via `.env` (e.g. `QDRANT_HTTP_PORT=16333`).

**Is a GPU required?**
No. CPU-only installs run small models (Qwen3-4B Q4). GPU hosts get much
faster inference. macOS runs CPU-tier (Docker VM limitation); Apple Silicon
uses the native arm64 image.

**The installer picked the wrong image tier**
Set `LLAMACPP_DOCKER_IMAGE` in `.env` to any of
`ghcr.io/aleckarfonta/llama-nexus-llamacpp:cpu|:cuda-86|:cuda-89|:cuda-120`
and re-run `up -d`. Note: the `cpu` image is multi-arch (amd64 + native
arm64 for Apple Silicon); CUDA variants are amd64-only.

**Windows: GPU works in games but not in containers**
Install/enable the WSL2 backend in Docker Desktop and update your NVIDIA
driver (WSL-CUDA support).

**Why am I asked for a management token?**
Your operator set `MANAGEMENT_AUTH_TOKEN` in the server's `.env` — typically
because the stack is exposed on a LAN. The login dialog wants that token;
enter it once and the browser remembers it. On a default localhost install
with no token configured, the dialog never appears. See
[SECURITY.md](SECURITY.md#management-token-login-dialog).

**Where is the Neo4j password?**
In the `.env` file next to `docker-compose.yml` (`NEO4J_PASSWORD=...`). The
installers generate a random one at install time — there is no default to
look up. If you cloned the repo manually and compose fails with
`NEO4J_PASSWORD missing in .env`, just set any strong value
(e.g. `openssl rand -hex 24`).

## Models

**Download fails with 401/403**
Gated repo — put a HuggingFace token in `.env` (`HUGGINGFACE_TOKEN=`).

**"Model file not found" when deploying**
The preset's model isn't downloaded yet. Models page → download it (the
error message links the repo), then Deploy again.

**Out of memory / CUDA OOM on deploy**
Model too large for your VRAM. Pick a smaller quantization (Q4_K_M), reduce
context size, or reduce GPU layers (partial offload).

**Generation is really slow**
CPU inference with a large model. Use the starter model, smaller context,
and check the task manager isn't thermally throttling.

## Services

**Chat says the LLM isn't running**
Deploy page → check the status card → Start. First deploy takes a minute to
load weights.

**Voice mode toggle disabled**
The STT service isn't running. Deploy it from the Speech-to-Text page
(streaming variant optional).

**Vector DB chip red**
Qdrant didn't start — click the chip's page action or check
`docker logs llama-nexus-qdrant`. It normally starts with the stack.

**Embeddings fail in Documents**
Deploy the Embeddings service (Nomic Embed) first; Qdrant must be green too.

**TTS has no sound / robotic**
Check the TTS page's served voice; Piper voices vary. Ensure your browser
allows audio autoplay for localhost.

## Platform

**Where are my conversations/models stored?**
Docker volumes — see [SECURITY.md](SECURITY.md) for the table.

**Which platforms are actually tested?**
See [TEST_MATRIX.md](TEST_MATRIX.md) — Linux+NVIDIA and Linux CPU are
continuously verified; Windows/macOS are logic-reviewed pending hardware runs.

**How do I reset everything?**
`docker compose -f docker-compose.yml -f docker-compose.prebuilt.yml down -v`
then start again. The wizard re-runs on fresh browser storage.

**Can I use it with Cursor/other OpenAI-compatible apps?**
Yes — point them at `http://localhost:8600/v1` with your `API_KEY`.

**How do I see what changed / update?**
See [CHANGELOG.md](../../CHANGELOG.md); update steps in
[INSTALL.md](INSTALL.md#updating).
