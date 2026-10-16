# Choosing Models

Llama Nexus downloads models from HuggingFace (GGUF format for llama.cpp).
The built-in catalog (`Models` page, or the first-run wizard) recommends
hardware-matched models — the wizard even deploys your pick automatically —
and any HuggingFace GGUF repo works too; this guide explains the trade-offs.

## Rule of thumb

- **VRAM needed ≈ model file size + 1–2 GB** for context (KV cache)
- Quantization suffixes: `Q4_K_M` = 4-bit (recommended balance), `Q8_0` =
  8-bit (better quality, 2× size), `IQ1/IQ2/IQ3` = extreme compression for
  very large models
- CPU-only: stay at or below **4B params @ Q4** (~2.5 GB) for usable speed

## Chat LLMs

| Model | File size | Min VRAM | Good for |
|---|---|---|---|
| Qwen3-4B-Instruct Q4_K_M | ~2.5 GB | 4 GB / CPU ok | Default starter, fast chat |
| Qwen3-4B-Instruct Q8_0 | ~4.4 GB | 6 GB | Higher-quality starter on 8 GB+ GPUs |
| Llama-3.1-8B-Instruct Q4_K_M | ~4.9 GB | 6 GB | Better general quality |
| Qwen3-30B-A3B Q4_K_M (MoE) | ~18.6 GB | 20 GB | Big-model quality, small-model speed |
| Qwen3.8-Flash-Next UD-IQ1_S | ~8.6 GB | 10 GB (multi-GPU best) | High-throughput advanced workloads |

## Speech

- **STT (speech-to-text):** faster-whisper models — `distil-large-v3.5`
  (balanced, default) or `large-v3` (max accuracy). Deploy from the
  Speech-to-Text page; models download on first start.
- **Streaming STT:** NVIDIA Nemotron streaming (GPU) — live transcription for
  the voice loop.
- **TTS:** OpenAI-compatible Piper voices (alloy, echo, fable, onyx, nova,
  shimmer) — no model download needed, ships with the TTS service.

## Embeddings + RAG

- **Nomic Embed v1.5** (default, ~150 MB) — required for the Documents/RAG
  features. Deploy from the Embeddings page.
- **Qdrant vector database** — started with the stack; manage from the
  service status bar (Vector DB chip).

## Deployment presets

The Deploy page offers one-click presets:

| Preset | Target |
|---|---|
| Chat · Starter (4B) | New installs — CPU or any 4 GB+ GPU |
| Chat · Assistant / VTuber / RP / Multi-Platform | Larger curated models for GPU rigs |
| Chat · Fast / Side Bot | Second lightweight endpoint |

If a preset's model isn't downloaded yet, download it from the Models page
first (the UI will link you).

## Hardware cheatsheet

| Your hardware | Recommended setup |
|---|---|
| CPU only | Qwen3-4B Q4_K_M, ctx 8192, threads 0 (auto) |
| 6–8 GB VRAM | Qwen3-4B Q8_0 or Llama-3.1-8B Q4_K_M |
| 12–16 GB VRAM | Llama-3.1-8B Q8_0 or 14B Q4 |
| 20–24 GB VRAM | Qwen3-30B-A3B Q4_K_M (MoE) |
| Multi-GPU rig | Qwen3.8-Flash-Next / 30B+ with tensor split; vLLM engine optional |
