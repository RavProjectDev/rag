---
name: run-local
description: Start the RAG API or the Webhook API locally. Use when the user asks to run, start, or launch the server, webhook, or either API process locally. Handles environment setup, port selection, and process verification.
---
# Run Local — RAG API or Webhook API

This project has **two separate FastAPI processes**. Always confirm which one is needed before starting.

---

## Which process to start?

| User wants to... | Start |
|---|---|
| Query the chat endpoint, test RAG, run smoke tests | **RAG API** |
| Test CMS sync, develop webhook handlers, test HMAC | **Webhook API** |
| Both | Start in two separate terminals |

---

## RAG API

```bash
# From the repo parent directory (one level above `rag/`)
PYTHONPATH=.. uvicorn rag.app.main:app --port 8000 --reload
```

Health check: `GET http://localhost:8000/api/v1/health`

**Required env vars** (in `rag/.env`): all `Settings` fields — Gemini, Pinecone/MongoDB, Supabase, optionally Redis.

---

## Webhook API

```bash
# From the repo parent directory
PYTHONPATH=.. uvicorn rag.app.webhook:app --port 8001 --reload
```

Health check: `GET http://localhost:8001/api/v1/health`

**Required env vars**: only `SharedSettings` fields (MongoDB, embedding config) plus `WEBHOOK_SECRET` if testing HMAC verification. Does **not** need Supabase, Redis, or LLM keys.

---

## Running both simultaneously

Open two terminals from the repo parent directory:

```bash
# Terminal 1 — RAG API
PYTHONPATH=.. uvicorn rag.app.main:app --port 8000 --reload

# Terminal 2 — Webhook API
PYTHONPATH=.. uvicorn rag.app.webhook:app --port 8001 --reload
```

---

## Verifying startup

Both apps log their DB and embedding config on startup. Confirm you see lines like:

```
[PINECONE CONFIG] index=... namespace=...
```
or
```
MongoDB collection: ...
```

If startup fails, the most common causes are missing env vars or a wrong `PYTHONPATH`.

---

## Common failure modes

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError: No module named 'rag'` | Run from the **parent** directory of `rag/`, not inside it |
| `ValidationError` on startup | Missing required env var — check `app/core/config.py` for the field |
| Port already in use | Another process on that port — use `lsof -i :8000` to find it |
| Webhook app requires Supabase/Redis vars | You're accidentally using `Settings` instead of `WebhookSettings` — check `app/webhook.py` |
