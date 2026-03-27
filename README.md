# RAV RAG API

A production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI, purpose-built for querying and surfacing content from Rav Soloveitchik's recorded lectures and transcripts. The system ingests subtitle (`.srt`) and plain-text (`.txt`) transcript files, chunks and embeds them into a vector database, and serves a chat API that retrieves the most relevant passages before generating a cited, structured response with an LLM.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
5. [Chunking Strategies](#chunking-strategies)
6. [Prompt System](#prompt-system)
7. [Vector Database Backends](#vector-database-backends)
8. [Installation & Setup](#installation--setup)
9. [Environment Variables](#environment-variables)
10. [Running Locally](#running-locally)
11. [Docker](#docker)
12. [API Reference](#api-reference)
13. [Authentication & Rate Limiting](#authentication--rate-limiting)
14. [Monitoring & Metrics](#monitoring--metrics)
15. [Scripts (Data Ingestion)](#scripts-data-ingestion)
16. [Testing](#testing)
17. [CI/CD](#cicd)

---

## Overview

The system is composed of two main concerns:

**Ingestion** — Transcript files are downloaded (typically from a Sanity CMS CDN), preprocessed into semantically meaningful chunks using one of several configurable strategies, embedded using Google's Gemini embedding model, and stored in either MongoDB (with Atlas vector search) or Pinecone.

**Retrieval & Generation** — When a user submits a question via the chat API, the query is embedded with the same model, the top-k most similar chunks are retrieved from the vector store, and the resulting context is passed to an OpenAI GPT model along with a structured prompt. The model returns a JSON object containing a synthesised `main_text` answer and a `source_numbers` array identifying which retrieved chunks support the answer. The API maps those source numbers back to the original transcript metadata, assembles richly annotated `sources` objects, and returns the full `ChatResponse`.

---

## Architecture

```
User / Client
     │
     ▼
FastAPI Application (uvicorn)
     │
     ├── Middleware layer
     │     ├── CORS
     │     ├── Request ID & timing
     │     └── Rate limiting (global + per-user, via Upstash Redis)
     │
     ├── /api/v1/chat          Chat & document retrieval
     ├── /api/v1/health        Health probe
     ├── /api/v1/info          Build / version info
     ├── /api/v1/prompt        Prompt management
     ├── /api/v1/user          User info & rate-limit status
     ├── /api/v1/config        Available embedding/index configs
     ├── /api/v1/test          Mock endpoints (non-production)
     └── /form                 Internal form & ratings interface
           │
           ▼
     Services layer
     ├── Embedding service     Gemini (Google Vertex AI / Generative AI)
     ├── LLM service           OpenAI GPT-4 / Gemini / Mock
     ├── Preprocess service    Chunking strategies (5 modes)
     └── Auth service          JWT verification (Supabase)
           │
           ▼
     Data layer
     ├── MongoDB               Vector store + metrics + exceptions
     └── Pinecone              Alternative vector store (optional)
           │
           ▼
     External APIs
     ├── OpenAI                Chat completions
     ├── Google Gemini         Embeddings + optional LLM
     ├── Supabase              Auth (JWT) + query persistence
     └── Upstash Redis         Rate limit counters
```

---

## Directory Structure

```
rag/
├── app/
│   ├── main.py                     FastAPI app + router registration
│   ├── dependencies.py             FastAPI dependency injection helpers
│   ├── webhook.py                  Incoming webhook handler
│   │
│   ├── api/v1/
│   │   ├── chat.py                 Core chat + document retrieval endpoints
│   │   ├── config.py               Available Pinecone index/namespace configs
│   │   ├── docs.py                 Custom API docs routing
│   │   ├── form.py                 Internal form interface
│   │   ├── health.py               Health check endpoint
│   │   ├── info.py                 Build info endpoint
│   │   ├── mock.py                 Mock/test endpoints
│   │   ├── prompt.py               Prompt CRUD endpoints
│   │   ├── user.py                 User info + rate-limit status
│   │   └── webhook.py              Webhook event endpoint
│   │
│   ├── core/
│   │   ├── config.py               Pydantic settings (SharedSettings + Settings)
│   │   ├── lifespan.py             App lifespan, middleware registration, error handlers
│   │   ├── scheduler.py            APScheduler background tasks
│   │   └── webhook_config.py       Webhook routing config
│   │
│   ├── db/
│   │   ├── connections.py          Abstract base classes (EmbeddingConnection, MetricsConnection)
│   │   ├── mongodb_connection.py   MongoDB implementation (motor + Atlas vector search)
│   │   ├── pinecone_connection.py  Pinecone implementation
│   │   └── redis_connection.py     Upstash Redis (rate limiting)
│   │
│   ├── exceptions/
│   │   ├── base.py                 BaseAppException
│   │   ├── db.py                   DB/retrieval exceptions (inc. NoDocumentFoundException)
│   │   ├── embedding.py            Embedding exceptions
│   │   ├── llm.py                  LLM exceptions
│   │   └── upload.py               Upload exceptions
│   │
│   ├── middleware/
│   │   └── rate_limit.py           Global + per-user rate limit middleware helpers
│   │
│   ├── models/
│   │   └── data.py                 Core domain models (SanityData, Metadata, DocumentModel)
│   │
│   ├── schemas/
│   │   ├── data.py                 Enums + value objects (ChunkingStrategy, LLMModel, etc.)
│   │   ├── requests.py             Request schemas (ChatRequest, RetrieveDocumentsRequest)
│   │   ├── response.py             Response schemas (ChatResponse, SourceItem, ErrorResponse, etc.)
│   │   └── form.py                 Form upload schemas
│   │
│   └── services/
│       ├── auth.py                 JWT verification (Supabase)
│       ├── embedding.py            Gemini embedding generation
│       ├── llm.py                  LLM prompt building, OpenAI/Gemini calls, streaming
│       ├── prompts.py              Prompt templates (PromptType enum + PROMPTS dict)
│       ├── user_service.py         User profile helpers
│       ├── form/                   Form-specific business logic
│       ├── preprocess/
│       │   ├── constants.py        Tuning constants (token sizes, LLM models, etc.)
│       │   ├── transcripts.py      Top-level chunking orchestration
│       │   ├── user_input.py       Query preprocessing (strip question words, etc.)
│       │   └── strategies/
│       │       ├── fixed_size.py              Fixed-token chunking
│       │       ├── divided.py                 Divided (large chunk → subdivisions)
│       │       ├── sentence_aware_regex.py    Sentence-boundary-aware chunking
│       │       ├── agentic.py                 LLM-guided section detection (single call)
│       │       └── agentic_multi_call.py      LLM-guided section detection (multi-call + retries)
│       └── sync_service/           Background sync helpers
│
├── scripts/
│   ├── README.md                   Script usage documentation
│   ├── sync_manifest.py            Sync transcript manifest from CMS
│   ├── upload_manifest.py          Bulk upload + embed from manifest
│   └── lib/ingest.py               Core ingest logic used by scripts
│
├── tests/
│   ├── conftest.py
│   ├── factories.py
│   ├── unit_test/
│   └── test_db/
│
├── .github/workflows/              GitHub Actions CI/CD pipelines
│   ├── deploy-stg-gcp.yml
│   ├── deploy-prd-gcp.yml
│   ├── deploy_ec2_stg.yml
│   └── ec2_deploy_prd.yml
│
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── .pre-commit-config.yaml
```

---

## How the RAG Pipeline Works

### 1. Ingestion (offline / scripts)

```
Transcript file (.srt or .txt)
        │
        ▼
preprocess/transcripts.py
  └── selects chunking strategy from ChunkingStrategy enum
        │
        ▼
strategies/<strategy>.py
  └── splits file into Chunk objects with text + optional timestamp
        │
        ▼
services/embedding.py
  └── calls Gemini Embedding API for each chunk
        │
        ▼
db/mongodb_connection.py  OR  db/pinecone_connection.py
  └── stores { vector, text, metadata (slug, title, timestamp, …) }
```

### 2. Chat request (online / API)

```
POST /api/v1/chat/
        │
        ▼
[Auth]  verify_jwt_token  (Supabase JWT or dev bypass)
        │
        ▼
[Rate limit]  global window + per-user monthly cap (Redis)
        │
        ▼
services/embedding.py  →  embed the user question
        │
        ▼
EmbeddingConnection.retrieve()  →  top-k nearest chunks from vector DB
        │  (retried up to max_retry_attempts with exponential backoff)
        ▼
services/llm.py  generate_prompt()
  └── numbers each retrieved chunk [1], [2], … [N]
  └── selects prompt template from PromptType
  └── returns full prompt string + source_list metadata
        │
        ▼
services/llm.py  get_llm_response() / stream_llm_response()
  └── calls OpenAI (or Gemini) with optional JSON schema enforcement
        │
        ▼
api/v1/chat.py  (STRUCTURED_JSON path)
  └── json.loads(llm_response)
  └── strips inline citation brackets from main_text (regex safety net)
  └── maps source_numbers → source_list → groups by document
  └── reconstructs full document text with **bold** used quotes
  └── builds SourceItem list with timestamp ranges + cosine scores
        │
        ▼
[Optional]  submit_to_supabase()  persists query + response
        │
        ▼
ChatResponse { main_text, sources[], thread_id }
```

### Streaming path

For `type_of_request: "STREAM"` the response is Server-Sent Events (SSE):

| Event | Payload |
|-------|---------|
| First event | `transcript_data` JSON — the retrieved source metadata |
| Token events | Raw LLM token chunks as they arrive |
| Final event | `[DONE]` |

---

## Chunking Strategies

Configured via `CHUNKING_STRATEGY` env var or per-request `pinecone_namespace`. All strategies produce `Chunk` objects with `text` and an optional `timestamp`.

| Strategy | Enum value | Description |
|----------|-----------|-------------|
| **Fixed size** | `FIXED_SIZE` | Splits into windows of `TOKENS_PER_CHUNK` (default 200) tokens using `cl100k_base` encoding. Simple and fast; recommended baseline. |
| **Divided** | `DIVIDED` | Builds large main chunks of `DIVIDED_CHUNK_SIZE` (default 800) tokens then subdivides each into `DIVIDED_SUBDIVISIONS` (default 4) equal parts. Balances broad context with fine granularity. |
| **Sentence-aware regex** | `SENTENCE_FIXED_REGEX` / `SENTENCE_DIVIDED_REGEX` | Same as fixed/divided but only breaks at sentence boundaries (detected by punctuation regex). Prevents mid-sentence cuts, producing more natural chunks. |
| **Agentic (single call)** | `AGENTIC` | Sends SRT segments to an LLM (default: Gemini Flash) and asks it to identify thematic section boundaries. One API call per transcript. Sections validated against `AGENTIC_MIN_SECTION_SEGMENTS` (2) and `AGENTIC_MAX_SECTION_SEGMENTS` (30). |
| **Agentic multi-call** | `AGENTIC_MULTI_CALL` | Two-stage: first call identifies section structure, second refines boundaries. Up to `AGENTIC_MULTI_CALL_MAX_RETRIES` (5) attempts with exponential backoff starting at `AGENTIC_MULTI_CALL_RETRY_BASE_DELAY` (5 s). Best quality; slowest. |

Tuning constants live in `app/services/preprocess/constants.py`.

---

## Prompt System

Five prompt types are available, selectable per-request via `prompt_type` in `ChatRequest`:

| PromptType | Value | Description |
|-----------|-------|-------------|
| `MINIMAL` | `"minimal"` | Bare-bones — question + context only, no structural instructions. |
| `LIGHT` | `"light"` | Short, clean response with basic formatting. Default when no prompt type is specified. |
| `MODERATE` | `"moderate"` | Multi-paragraph analysis with inline quotes. |
| `COMPREHENSIVE` | `"comprehensive"` | Long-form: Introduction, themed Main Content sections with block quotes, Summary. |
| `STRUCTURED_JSON` | `"production"` | **Default for all production requests.** Forces the LLM to return a strict JSON object `{ "main_text": string, "source_numbers": [number] }`. `main_text` is citation-free prose; `source_numbers` drives the `sources` array in `ChatResponse`. |

Only `STRUCTURED_JSON` returns a populated `sources` field. All other prompt types return `sources: []`.

When `dev_outputs` is enabled in settings, every generated prompt is written to `dev_outputs/` for inspection.

---

## Vector Database Backends

Configured via `DATABASE_CONFIGURATION` env var.

### MongoDB (default)

- Uses **Motor** (async driver) against MongoDB Atlas.
- Requires an Atlas Vector Search index named by `COLLECTION_INDEX` (default `"vector_index"`) on the `MONGODB_VECTOR_COLLECTION` collection.
- The `vector` field path is configurable via `VECTOR_PATH` (default `"vector"`).
- Chunk tracking is optionally written to a separate `chunks` collection alongside vector documents.
- Metrics are stored in `METRICS_COLLECTION` (default `"metrics"`); exceptions in `EXCEPTIONS_COLLECTION` (default `"exceptions"`).

### Pinecone (optional)

- Configured when `DATABASE_CONFIGURATION=PINECONE`.
- Requires `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and optionally `PINECONE_NAMESPACE` and `PINECONE_HOST`.
- Per-request index and namespace can be overridden via `pinecone_index` and `pinecone_namespace` fields in `ChatRequest`.
- Available indexes and namespaces are discoverable via `GET /api/v1/config/available-configs`.

---

## Installation & Setup

### Prerequisites

- Python 3.13+
- MongoDB Atlas instance with Vector Search index (or Pinecone account)
- OpenAI API key
- Google Cloud project with Gemini / Vertex AI access

### Clone and install

```bash
git clone <repository-url>
cd rag

# Install production dependencies
pip install -r requirements.txt

# Install dev dependencies (includes pytest, pre-commit, etc.)
pip install -r requirements-dev.txt
```

### Set up environment

```bash
cp .env.example .env
# Edit .env and fill in the required values listed below
```

---

## Environment Variables

All variables are loaded by Pydantic Settings from `.env` (or the process environment). See `.env.example` for the full list with descriptions.

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `GEMINI_API_KEY` | Google Gemini API key (embeddings) |
| `GOOGLE_CLOUD_PROJECT_ID` | GCP project ID |
| `VERTEX_REGION` | Vertex AI region (e.g. `us-central1`) |
| `MONGODB_URI` | MongoDB connection string |
| `MONGODB_DB_NAME` | MongoDB database name |
| `MONGODB_VECTOR_COLLECTION` | Collection that holds vector embeddings |

### LLM & Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_CONFIGURATION` | `GPT_4` | LLM to use: `GPT_4`, `GEMINI_FLASH`, `MOCK` |
| `EXTERNAL_API_TIMEOUT` | `60` | Seconds before LLM/embedding calls time out |
| `RETRIEVAL_TIMEOUT_MS` | `2000` | Per-attempt vector query timeout (ms) |
| `MAX_RETRY_ATTEMPTS` | `3` | Max vector retrieval retries |
| `RETRY_DELAY_SECONDS` | `1.0` | Initial retry delay |
| `RETRY_BACKOFF_MULTIPLIER` | `2.0` | Exponential backoff multiplier |

### Embedding & Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_CONFIGURATION` | `GEMINI` | Embedding model: `GEMINI`, `MOCK` |
| `CHUNKING_STRATEGY` | `FIXED_SIZE` | Default chunking strategy (see Chunking Strategies) |
| `VECTOR_PATH` | `vector` | Field name for the vector in MongoDB documents |

### Auth

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_MODE` | `dev` | `dev` = bypass JWT validation; `prd` = enforce Supabase JWTs |
| `SUPABASE_URL` | — | Supabase project URL (required in `prd` auth mode) |
| `SUPABASE_SERVICE_ROLE_KEY` | — | Supabase service role key |
| `SUPABASE_ANON_KEY` | — | Supabase anon key |

### Rate Limiting (Redis / Upstash)

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTASH_REDIS_REST_URL` | — | Upstash Redis REST endpoint |
| `UPSTASH_REDIS_REST_TOKEN` | — | Upstash Redis token |
| `RATE_LIMIT_MAX_REQUESTS` | `10000` | Global requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `3600` | Window size in seconds |
| `USER_RATE_LIMIT_MAX_REQUESTS_PER_MONTH` | `10000` | Per-user monthly cap |

### Pinecone (optional)

| Variable | Description |
|----------|-------------|
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Default index name |
| `PINECONE_NAMESPACE` | Default namespace |
| `PINECONE_HOST` | Pinecone host URL |

### Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `STG` | `PRD`, `STG`, or `TEST` |
| `METRICS_COLLECTION` | `metrics` | MongoDB collection for request metrics |
| `EXCEPTIONS_COLLECTION` | `exceptions` | MongoDB collection for exception logs |
| `DEV_OUTPUTS` | `false` | Write generated prompts to `dev_outputs/` for debugging |

---

## Running Locally

The application package is `rag.app`, so the `rag` directory must be on `PYTHONPATH`:

```bash
# From the parent directory of rag/
PYTHONPATH=. uvicorn rag.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or from inside the `rag/` directory with `PYTHONPATH` pointing one level up:

```bash
cd rag
PYTHONPATH=.. uvicorn rag.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs are available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Docker

```bash
# Build the image
docker build -t rav-rag .

# Run (pass env vars via --env-file or -e flags)
docker run -p 8080:8080 --env-file .env rav-rag
```

The container exposes port `8080` by default (overridable via the `PORT` env var).

> **Note:** The `Dockerfile` runs `uvicorn app.main:app` (not `rag.app.main:app`). It sets `PYTHONPATH=/` so that `app` is importable directly from the container root at `/rag/app`. Make sure your deployment environment sets `PYTHONPATH` accordingly if deviating from the Docker build.

---

## API Reference

All endpoints require a `Bearer` JWT token in the `Authorization` header when `AUTH_MODE=prd`. In `dev` mode the token is not validated.

---

### Chat

#### `POST /api/v1/chat/`

The primary endpoint. Accepts a question, retrieves relevant transcript chunks, and returns a generated answer with source citations.

**Request body:**

```json
{
  "question": "What did Rav Soloveitchik say about teshuva?",
  "type_of_request": "FULL",
  "prompt_type": "production",
  "name_spaces": null,
  "pinecone_index": null,
  "pinecone_namespace": null,
  "thread_id": null,
  "submit_query": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | yes | — | The user's question (must be non-empty) |
| `type_of_request` | `"STREAM"` \| `"FULL"` | yes | — | Stream tokens via SSE or wait for full JSON |
| `prompt_type` | string | no | `"production"` | One of: `minimal`, `light`, `moderate`, `comprehensive`, `production` |
| `name_spaces` | string[] | no | null | MongoDB namespaces to restrict retrieval to |
| `pinecone_index` | string | no | env default | Pinecone index override |
| `pinecone_namespace` | string | no | env default | Pinecone namespace override |
| `thread_id` | UUID | no | null | Append to an existing Supabase conversation thread |
| `submit_query` | bool | no | `true` | Persist query + response to Supabase |

**Full response (`type_of_request: "FULL"`):**

```json
{
  "main_text": "Rav Soloveitchik taught that teshuva is not merely regret...",
  "sources": [
    {
      "slug": "on-repentance",
      "text_id": "abc123",
      "full_text": "...full transcript text with **bolded used segments**...",
      "used_quotes": [
        { "number": 3, "text": "The segment text used", "timestamp": "00:12:34" }
      ],
      "timestamp_range": "00:10:00 - 00:18:45",
      "score": 0.91
    }
  ],
  "thread_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Streaming response (`type_of_request: "STREAM"`):**

```
data: {"transcript_data": [...]}

data: "Rav Solov"
data: "eitchik taught"
...
data: [DONE]
```

---

#### `POST /api/v1/chat/documents`

Retrieve raw document chunks without LLM generation. Useful for inspecting what the vector search would surface for a given question.

**Request body:**

```json
{
  "question": "covenant and community",
  "name_spaces": null,
  "top_k": 5,
  "pinecone_index": null,
  "pinecone_namespace": null
}
```

**Response:**

```json
{
  "request_id": "uuid",
  "cleaned_question": "covenant community",
  "requested_top_k": 5,
  "documents": [...],
  "transcript_data": [...],
  "message": null
}
```

---

### Health

#### `GET /api/v1/health/`

```json
{ "status": "ok", "version": "1.0.0", "environment": "STG" }
```

---

### Configuration

#### `GET /api/v1/config/available-configs`

Returns all available Pinecone indexes and their namespaces (chunking strategies) with vector counts.

#### `GET /api/v1/config/simple-configs`

Simplified version: just index names and their namespace lists plus the current defaults.

#### `GET /api/v1/config/enhanced-configs`

Returns available embedding model + chunking strategy combinations with human-readable descriptions.

---

### User

#### `GET /api/v1/user/rate-limit`

Returns the authenticated user's current monthly request usage, remaining quota, and reset timestamp.

---

### Prompt

#### `GET /api/v1/prompt/`

Returns available prompt types and their template strings.

---

### Form

#### `GET /form/{question}`

Retrieve document chunks for a question (no LLM call). Used by the internal ratings interface.

#### `POST /form/upload_ratings`

Upload human-labelled relevance ratings for chunks returned by the form interface.

---

### Mock / Test

Endpoints under `/api/v1/test/` provide mock responses for client integration testing without hitting external APIs.

---

## Authentication & Rate Limiting

### Authentication

Controlled by `AUTH_MODE`:

- **`dev`** — All requests are accepted. No JWT is required. Safe for local development.
- **`prd`** — Every request to protected endpoints must include `Authorization: Bearer <supabase-jwt>`. The token is verified against `SUPABASE_URL` using `SUPABASE_ANON_KEY`. The `sub` claim is extracted as the user ID for rate limiting and Supabase persistence.

### Rate Limiting

Two layers, both backed by Upstash Redis. Rate limiting is silently skipped if Redis is not configured.

1. **Global** — `RATE_LIMIT_MAX_REQUESTS` requests per `RATE_LIMIT_WINDOW_SECONDS` across all users combined. Returns `429` when exceeded.
2. **Per-user monthly** — `USER_RATE_LIMIT_MAX_REQUESTS_PER_MONTH` requests per calendar month per user (resets at Eastern Time midnight on the 1st). Returns `429` with a `Retry-After` header when exceeded.

Rate limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`) are included in all chat responses.

---

## Monitoring & Metrics

- Every HTTP request is assigned a `X-Request-ID` (UUID) and its duration is logged.
- In `PRD` environment, successful requests log timing data to the `metrics` MongoDB collection via `MetricsConnection.log()`.
- Exceptions are logged to the `exceptions` MongoDB collection via `ExceptionsLogger`.
- All database operations (`retrieve`, `insert`) are wrapped in a `timed()` context manager that records operation duration.
- Set `DEV_OUTPUTS=true` during development to dump full prompt strings to `dev_outputs/` for debugging LLM behaviour.

---

## Scripts (Data Ingestion)

The `scripts/` directory contains offline tooling for the ingestion pipeline. See `scripts/README.md` for detailed usage.

| Script | Purpose |
|--------|---------|
| `sync_manifest.py` | Pull the current transcript manifest from the Sanity CMS and write it locally |
| `upload_manifest.py` | Read the manifest, download each transcript, chunk it, embed it, and upsert into the vector DB |
| `lib/ingest.py` | Shared ingestion logic used by both scripts |

Ingestion uses the same `chunking_strategy` and `embedding_configuration` settings as the API. Run these scripts to populate the vector database before serving chat requests.

---



## CI/CD

Four GitHub Actions workflows are defined in `.github/workflows/`:

| Workflow | Trigger | Target |
|---------|---------|--------|
| `deploy-stg-gcp.yml` | Push to staging branch | Google Cloud Run (staging) |
| `deploy-prd-gcp.yml` | Push to main / release | Google Cloud Run (production) |
| `deploy_ec2_stg.yml` | Push to staging branch | AWS EC2 (staging) |
| `ec2_deploy_prd.yml` | Push to main / release | AWS EC2 (production) |

Deployments build the Docker image, push it to a container registry, and update the running service. Secrets (API keys, cloud credentials) are stored as GitHub Actions secrets and injected at deploy time.

---

## Pre-commit Hooks

The repository ships with a `.pre-commit-config.yaml`. Install hooks once after cloning:

```bash
pre-commit install
```

A custom hook in `githooks/pre-commit` is also available if you prefer to manage hooks manually.
