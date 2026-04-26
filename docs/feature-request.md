- [ ] On the deploy page refactor the paramaeter section Reference the information on the argements here https://github.com/ggml-org/llama.cpp/tree/master/tools/server


- [ ]  On the chat page show the current active model


DEFAULT PROMPT
Write a detailed plan to implement this functionality Include references to files in the codebase that need to change to support this functionality. 

-[ ] Lets plan out an integration with graph rag: https://github.com/alecKarfonta/graphrag

We should be able leverage grpahrag for more advanced document processing, chunking, entity relationship extraction, etc

This will be a major refactor so lets make all of this new functionality work in tandem with the existing Document management system in llama-nexus. We just want to be able to leverage the more advanved features of graphrag when possible.

## Current Project Handoff - 2026-04-26

### Repository State

- Current branch: `master`
- Working tree was clean before this handoff note was added.
- `master` was ahead of `origin/master` by 1 commit.
- Latest commit before this note: `c62233b chore: reorganize root directory and clean up junk files`

### Last Completed Work

The last committed work reorganized the root directory and cleaned up stray files:

- Moved documentation into `docs/`
- Moved operational scripts into `scripts/`
- Moved tests into `tests/`
- Moved sample data into `data/`
- Deleted unneeded files including `dbz_summary.txt` and `hilton-so-well-remembered.epub`
- Updated script paths in `Dockerfile` and `docker-compose.yml`

### Current Docker Compose State

The following services were running and healthy/running when checked:

- `llamacpp-backend` healthy on port `8700`
- `llamacpp-frontend` healthy on port `3002`
- `llama-nexus-qdrant` healthy on ports `6333` and `6334`
- `llama-nexus-redis` healthy on host port `6389`
- `llama-nexus-training` running

The `llamacpp-api` inference service was not listed as running.

Compose also warned that `HUGGINGFACE_TOKEN` was unset and defaulted to a blank string.

### Verification Completed

- Local commits were pushed to `origin/master`.
- `docker compose up -d --build` completed successfully after one retry.
- Backend `/health` returned HTTP `200`.
- Backend `/api/v1/service/status` returned HTTP `200`.
- Frontend root page returned HTTP `200`.
- Qdrant `/healthz` returned HTTP `200`.
- Embedding service `/health` returned HTTP `200`.

### Unfinished Or Follow-Up Items

- Set `HUGGINGFACE_TOKEN` if model download or Hugging Face access is needed.
- Historical metrics are still simulated in `frontend/src/services/api.ts`.
- Monitoring WebSocket endpoint is not implemented; the UI falls back to polling.
- Model archiving endpoint in `backend/routes/models.py` is still a stub.
- Models page start/stop handling is not fully model-specific yet.
- GraphRAG hierarchical clustering is still a TODO.
- Broader roadmap leftovers include multi-user authentication, CLI tooling, end-to-end tests, and workflow polish.

### Recent Feature Context

The project notes indicate recent feature work before cleanup focused on fine-tuning, STT/TTS, and UI polish. Fine-tuning API tests were documented as passing, but the most recent actual repository change was the cleanup/reorganization commit listed above.

