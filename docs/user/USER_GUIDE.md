# Using Llama Nexus

Everyday workflows after your stack is running (see [INSTALL.md](INSTALL.md)).

## The web UI

Open **http://localhost:3002**. The sidebar has two modes:

- **Simple mode** (default): Dashboard, Chat, Deploy LLM, Models, Speech
  pages, Documents, Settings.
- **Advanced tools** (sidebar footer toggle): everything — workflows,
  fine-tuning, quantization, GraphRAG workbench, benchmarks, MCP, registry.

The top status bar shows live health for LLM, Embeddings, Vector DB (Qdrant),
STT, TTS, and training/quantization workers. Click a chip to jump to its page.

## Chat with a local LLM

1. **Deploy → Deploy LLM** — pick a preset (Chat · Starter 4B for new
   installs) or a downloaded model, adjust context/sampling, click Deploy.
2. **Chat** — conversations stream token-by-token; history is saved locally.
3. Settings (gear) in Chat control model parameters per conversation.

## Voice: talk to your LLM

The full local loop — your speech → text → LLM → spoken reply:

1. Deploy **Speech-to-Text** (faster-whisper; small model for quick setup).
2. Deploy **Text-to-Speech** (Piper voices; no model download needed).
3. In **Chat**, enable the **Voice mode** toggle, then use the microphone
   button. Replies are spoken automatically in voice mode.

Individual pieces work independently too: any message has a "read aloud"
button, and STT/TTS expose OpenAI-compatible endpoints for other apps.

## Knowledge: chat with your documents (RAG)

1. Deploy **Embeddings** (Nomic Embed v1.5, one click).
2. **Documents** → upload PDFs/Office docs/text. Files are chunked and
   embedded into the local Qdrant vector DB.
3. Ask questions in **Chat** with the knowledge context enabled — answers
   cite your documents.

Optional deeper features (Advanced mode): GraphRAG workbench for knowledge
graphs, hybrid search, reasoning playground.

## The API

Everything is OpenAI-compatible for other tools:

```bash
curl http://localhost:8600/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"hello"}]}'
```

Point any OpenAI-SDK app at `http://localhost:8600/v1` with the `API_KEY`
from your `.env`. Embeddings: `/v1/embeddings` on port 8602. Transcription:
`/v1/audio/transcriptions` (STT). Speech: `/v1/audio/speech` (TTS).

## Managing the stack

- **Start/stop services**: status-bar chips or each Deploy page.
- **Logs**: each Deploy page streams container logs; or
  `docker compose logs -f <service>`.
- **Update models**: Models page → download; delete unused ones to free disk.
- **Backups**: model files live in the `gpt_oss_models` Docker volume;
  conversations in `conversation_data`. See SECURITY.md for paths.

## Tips

- First generation after deploy is slower (model warm-up).
- Smaller context = faster + less memory. 8k suits most chat.
- On CPU, expect reading-speed generation; GPUs are transformative.
- The wizard can be re-run anytime from Settings.
