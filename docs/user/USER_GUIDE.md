# Using Llama Nexus

Everyday workflows after your stack is running (see [INSTALL.md](INSTALL.md)).

## The web UI

Open **http://localhost:3002**. The sidebar has two modes:

- **Simple mode** (default): eight core pages — Dashboard, Chat, Deploy LLM,
  STT, TTS, Models, Documents, Settings.
- **Advanced tools** (sidebar footer toggle): everything — workflows,
  fine-tuning, quantization, GraphRAG workbench, benchmarks, MCP, registry.

The top status bar shows live health for LLM, Embeddings, Vector DB (Qdrant),
STT, TTS, and training/quantization workers. Click a chip to jump to its page.

**Dashboard → Workflow Runway**: a guided checklist of the llama-nexus path
(assets → deploy → chat → knowledge → finetuning → operations). Each step
shows a live status chip that flips to Complete once you've actually used the
feature.

**Management-token login dialog**: if your operator set
`MANAGEMENT_AUTH_TOKEN` in `.env`, the first management-API call fails with
401 and a login dialog appears asking for the token. Enter it once; it's
remembered in your browser and re-attached automatically. On a default
localhost install (no token configured) the dialog never appears.

## First run: the wizard deploys for you

On a fresh install the welcome wizard runs automatically: it detects your
hardware (GPU/VRAM/RAM), asks your use case, and recommends a model. Download
it and the wizard **deploys it automatically** — no separate Deploy step.
Click **Get Started** on the final step and you land straight in **Chat**
with the model live. Voice (STT/TTS) and RAG services can be enabled later
from their own pages.

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

## Settings

The Settings page manages both server-side installation settings and local
browser preferences:

- **HuggingFace token** — saved on the server (survives restarts, stored in
  the backend data volume). Needed only for gated model downloads; create a
  token at huggingface.co/settings/tokens.
- **API keys** — shows the effective inference and embedding keys (masked).
  Change them on the matching Deploy page and redeploy.
- **Storage** — where models, conversations, and caches live, with free
  space per volume.
- **Network exposure** — flip between localhost-only and LAN-wide binding.
  Requires a stack restart to take effect; the page warns you and suggests a
  management token when exposing to the LAN (see SECURITY.md).
- **Preferences** — Simple/Advanced navigation, compact layouts, and other
  browser-local toggles.

## Tips

- First generation after deploy is slower (model warm-up).
- Smaller context = faster + less memory. 8k suits most chat.
- On CPU, expect reading-speed generation; GPUs are transformative.
- The wizard can be re-run anytime from Settings.
