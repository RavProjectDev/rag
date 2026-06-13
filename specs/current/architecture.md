# System Architecture — Current State

> Last updated: 2026-06-13
> Environment documented: LOCAL (active dev config)

---

## Purpose

Production RAG API serving natural-language questions about Rav Soloveitchik's recorded teachings.

Transcripts (`.srt` / `.txt` files sourced from Sanity CMS) are chunked, embedded into a vector
store, and retrieved at query time to ground LLM responses with cited, timestamped sources.

---

## Active configuration

| Setting | Value | Constant |
|---|---|---|
| Database backend | `PINECONE` | `app/core/constants/app.py` |
| Embedding model | `gemini-embedding-001` | `app/core/constants/llm.py` |
| LLM | `gemini-2.5-flash` | `app/core/constants/llm.py` |
| Chunking strategy | `fixed_size` | `app/core/constants/llm.py` |
| Environment | `LOCAL` | `app/core/constants/app.py` |
| Dev outputs | `True` | `app/core/constants/app.py` |

---

## Stack

| Layer | Technology | Notes |
|---|---|---|
| API framework | FastAPI 1.0.0 | Async, Pydantic v2 |
| Embeddings | Google Gemini | `gemini-embedding-001` via Gemini API |
| LLM | Gemini Flash / GPT-5.2 | Switchable via `LLM_CONFIGURATION` |
| Vector store | Pinecone | MongoDB Atlas Vector is the alternate backend |
| Auth | Supabase JWT | Enforced in `prd`/`stg`; bypassed in `local` |
| Rate limiting | Upstash Redis | Disabled when `UPSTASH_REDIS_REST_TOKEN` is absent |
| Query persistence | Supabase | RPC `submit_user_query` (async, fire-and-forget on error) |
| Metrics / exceptions | MongoDB | Written by `MetricsConnection` during timed context managers |
| CMS | Sanity | Source of transcript manifest and document metadata |

---

## Request lifecycle — chat endpoint

```
Client
  │
  ▼
POST /api/v1/chat/
  │
  ├─ verify_jwt_token (Supabase JWT — skipped in LOCAL)
  ├─ rate_limit_middleware (global, per-endpoint, Redis)
  ├─ user_rate_limit_middleware (per-user monthly cap, Redis)
  │
  ▼
pre_process_user_query()         ← cleans / normalises input
  │
  ▼
generate_embedding()             ← Gemini API, task_type=RETRIEVAL_DOCUMENT
  │
  ▼
connection.retrieve()            ← Pinecone or MongoDB vector search
  │
  ▼
generate_prompt()                ← assembles context + numbered source list
  │
  ▼
get_llm_response()               ← Gemini Flash or GPT-5.2 with JSON schema
  │
  ▼
parse JSON → extract main_text + source_numbers
  │
  ├─ submit_to_supabase() [async, non-blocking on failure]
  │
  ▼
ChatResponse { main_text, sources[], thread_id }
```

---

## Module responsibilities

| Module | Responsibility |
|---|---|
| `app/api/v1/` | HTTP routing only — no business logic |
| `app/services/embedding.py` | Embedding generation (Gemini, OpenAI, BERT, mock) |
| `app/services/llm.py` | Prompt assembly and LLM response generation |
| `app/services/auth.py` | JWT verification via Supabase |
| `app/services/prompts.py` | Prompt templates (`PromptType` enum) |
| `app/services/preprocess/` | User input cleaning; chunking strategies |
| `app/db/connections.py` | `EmbeddingConnection` and `MetricsConnection` abstractions |
| `app/db/pinecone_connection.py` | Pinecone retrieve implementation |
| `app/db/mongodb_connection.py` | MongoDB Atlas Vector retrieve implementation |
| `app/core/config.py` | `Settings` / `SharedSettings` (Pydantic settings, env-overridable) |
| `app/core/constants/` | Non-secret default values — one file per service domain |
| `app/core/lifespan.py` | Startup / shutdown: DB connections, middleware, CORS |
| `app/exceptions/` | Typed exception hierarchy (`BaseAppException` → domain exceptions) |
| `scripts/` | Transcript ingestion (`upload_manifest.py`, `sync_manifest.py`) |
| `smoke_tests/` | End-to-end smoke test runner — evaluated against [`specs/current/app-requirements.md`](app-requirements.md) |

---

## Dual vector-store backends

The system supports two interchangeable vector backends, selected at startup via `DATABASE_CONFIGURATION`:

| Backend | Constant value | Connection class |
|---|---|---|
| Pinecone | `DataBaseConfiguration.PINECONE` | `PineconeConnection` |
| MongoDB Atlas Vector | `DataBaseConfiguration.MONGO` | `MongoDBConnection` |

Backend-specific constants are conditionally imported in `app/core/constants/__init__.py`.
The `EmbeddingConnection` protocol in `app/db/connections.py` abstracts the `retrieve()` call
so routers and services are backend-agnostic.

---

## Auth modes

| Environment | Auth behaviour |
|---|---|
| `PRD` / `STG` | `verify_jwt_token` validates Supabase JWT; request rejected if invalid |
| `LOCAL` / `TEST` | JWT check bypassed; `user_id` defaults to a fixed dev string |

---

## Rate limiting

Two independent limits are applied when Redis is configured:

| Limit | Default | Reset |
|---|---|---|
| Global (per endpoint) | 1,000,000 requests per 10,000 s window | Rolling |
| Per-user monthly | 100 requests/month | Calendar month |

On timeout, `NoDocumentFoundException`, or `BaseAppException`, the per-user counter is decremented
to avoid charging the user for failed requests.
