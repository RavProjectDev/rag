# API Surface — Current State

> Last updated: 2026-06-13
> Base path: all routes under the FastAPI app in `app/main.py`

---

## Auth

All endpoints except `/api/v1/health` require a Supabase JWT passed as a Bearer token.
In `LOCAL` / `TEST` environments the check is bypassed.

```
Authorization: Bearer <supabase-jwt>
```

---

## Routers

### Chat — `/api/v1/chat`

#### `POST /api/v1/chat/`

Full RAG pipeline: embed → retrieve → prompt → LLM → response with cited sources.

**Request** (`ChatRequest`):

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | `str` | required | User's question (non-empty) |
| `name_spaces` | `list[str] \| None` | `null` | Pinecone namespaces to search |
| `prompt_type` | `PromptType` | `STRUCTURED_JSON` | Prompt template variant |
| `pinecone_index` | `str \| None` | `null` | Override active Pinecone index |
| `pinecone_namespace` | `str \| None` | `null` | Override active namespace |
| `thread_id` | `uuid.UUID \| None` | `null` | Append to existing thread |
| `submit_query` | `bool` | `true` | Persist to Supabase |

**Response** (`ChatResponse`):

```json
{
  "main_text": "string",
  "sources": [
    {
      "slug": "string",
      "text_id": "string",
      "full_text": "string (used quotes bolded with **)",
      "used_quotes": [{"number": 1, "text": "...", "timestamp": "00:01:23"}],
      "timestamp_range": "00:01:00-00:02:30",
      "score": 0.94
    }
  ],
  "thread_id": "uuid | null"
}
```

**Error responses**: `400`, `408` (timeout), `422` (validation), `500`

**Acceptance criteria**: [`specs/current/app-requirements.md`](app-requirements.md)

---

#### `POST /api/v1/chat/documents`

Retrieval only — runs embedding and vector search, returns top-k documents without calling the LLM.

**Request** (`RetrieveDocumentsRequest`):

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | `str` | required | User's question |
| `name_spaces` | `list[str] \| None` | `null` | Pinecone namespaces |
| `top_k` | `int` | `5` | Documents to return (1–50) |
| `pinecone_index` | `str \| None` | `null` | Override index |
| `pinecone_namespace` | `str \| None` | `null` | Override namespace |

**Response** (`RetrieveDocumentsResponse`):

```json
{
  "request_id": "string",
  "cleaned_question": "string",
  "requested_top_k": 5,
  "documents": [...],
  "transcript_data": [...],
  "message": "string | null"
}
```

---

### Health — `/api/v1/health`

#### `GET /api/v1/health`

No auth required. Returns `HealthResponse { status, version, environment }`.
Used by smoke test runner and load balancer health checks.

---

### Info — `/api/v1/info`

#### `GET /api/v1/info/`

Returns Sanity CMS manifest for the transcript corpus (used by ingestion scripts).

---

### Config — `/api/v1/config`

#### `GET /api/v1/config/`

Returns `ConfigInfoResponse` with active embedding model, chunking strategy, DB backend, environment, and LLM.

#### `GET /api/v1/config/available-configs`

Returns `AvailableConfigurationsResponse` listing all Pinecone indexes and their namespaces
with dimension, metric, vector counts, and namespace details.

#### `GET /api/v1/config/simple-configs`

Lightweight version of `available-configs`: just index names and namespace lists.

#### `GET /api/v1/config/enhanced-configs`

Returns `EnhancedConfigurationsResponse`: embedding models with their chunking strategy descriptions.

---

### Mock — `/api/v1/test`

Dev/test endpoints that return synthetic responses without calling the LLM or vector DB.
All endpoints in this router bypass auth in non-production environments.

---

### Prompt — `/api/v1/prompt`

Endpoints for reading and testing prompt templates (`PromptType` variants).

---

### User — `/api/v1/user`

Rate limit management endpoints. Returns `RateLimitInfoResponse` with current usage, remaining,
limit, and reset timestamp for the authenticated user.

---

### Form — `/form`

Internal form processing endpoints. Uses `FormRequest` and returns `FormFullResponse`.
Not part of the public API surface.

---

### Docs — `/`

Mounts the FastAPI auto-generated OpenAPI docs (`/docs`, `/redoc`, `/openapi.json`).

---

## Standard error envelope

All error responses use `ErrorResponse`:

```json
{
  "code": "stable_machine_readable_code",
  "message": "Human-readable description",
  "request_id": "optional-correlation-id",
  "details": {}
}
```
