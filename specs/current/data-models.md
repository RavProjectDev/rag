# Data Models — Current State

> Last updated: 2026-06-13
> Source: `app/schemas/data.py`, `app/schemas/requests.py`, `app/schemas/response.py`, `app/models/data.py`

---

## Enums (`app/schemas/data.py`)

### `DataBaseConfiguration`

Selects the active vector store backend at startup.

| Member | Wire value | Notes |
|---|---|---|
| `PINECONE` | `"pinecone"` | Active default |
| `MONGO` | `"mongo"` | MongoDB Atlas Vector |

### `Environment`

| Member | Wire value |
|---|---|
| `PRD` | `"PRD"` |
| `STG` | `"STG"` |
| `TEST` | `"TEST"` |
| `LOCAL` | `"LOCAL"` |

### `LLMModel`

| Member | Wire value | Notes |
|---|---|---|
| `GPT_4` | `"gpt-5.2-2025-12-11"` | OpenAI |
| `GEMINI_FLASH` | `"gemini-2.5-flash"` | Active default |
| `GEMINI_PRO` | `"gemini-2.5-pro"` | |
| `MOCK` | `"mock"` | Returns synthetic responses |

### `EmbeddingConfiguration`

| Member | Wire value | Notes |
|---|---|---|
| `BERT_SMALL` | `"all-MiniLM-L6-v2"` | Local model |
| `GEMINI` | `"gemini-embedding-001"` | Active default |
| `COHERE` | `"cohere"` | |
| `OPENAI` | `"openai"` | |
| `MOCK` | `"mock"` | Returns zero vectors |

### `ChunkingStrategy`

| Member | Wire value | Description |
|---|---|---|
| `FIXED_SIZE` | `"fixed_size"` | Fixed token-based chunking. **Active default** |
| `DIVIDED` | `"divided"` | Large chunks divided into sub-chunks with shared context |
| `SENTENCE_FIXED_REGEX` | `"sentence_fixed_regex"` | Sentence-aware via regex |
| `SENTENCE_DIVIDED_REGEX` | `"sentence_divided_regex"` | Sentence-aware divided via regex |
| `AGENTIC` | `"agentic"` | LLM-guided boundary detection |
| `AGENTIC_MULTI_CALL` | `"agentic_multi_call"` | Two-stage: boundary detection + per-section LLM rewrite |

### Internal-only enums (not serialised to API contracts)

- `DataSourceConfiguration`: `LOCAL` — transcript file source
- `TypeOfFormat`: `SRT`, `TXT` — input transcript format

---

## Core data models

### `Chunk` (`app/schemas/data.py`)

Represents a single processed text segment after chunking.

| Field | Type | Description |
|---|---|---|
| `full_text_id` | `uuid.UUID` | Shared by all chunks from the same source segment |
| `full_text` | `str or list[tuple]` | Plain string (TXT) or timestamped segments (SRT) |
| `text_to_embed` | `str` | The portion sent to the embedding model |
| `chunk_size` | `int` | Total size of the full text segment (words) |
| `embed_size` | `int` | Size of the embedded portion (words) |
| `time_start` | `str or None` | SRT start timestamp |
| `time_end` | `str or None` | SRT end timestamp |
| `name_space` | `str` | Source transcript / document identifier |
| `text_hash` | `str` | SHA-256 of `text_to_embed` (deduplication key) |

### `VectorEmbedding` (`app/schemas/data.py`)

Associates an embedding vector with its source chunk and CMS metadata. Written to the vector store.

| Field | Type |
|---|---|
| `vector` | `list[float]` |
| `dimension` | `int` |
| `metadata` | `Chunk` |
| `sanity_data` | `SanityData` |
| `embedding_model` | `str` |
| `chunking_strategy` | `str` |

---

## Request schemas (`app/schemas/requests.py`)

### `ChatRequest`

See `specs/current/api-surface.md` for full field table.
Key constraint: `question` must be non-empty (enforced by `question_validator`).

### `RetrieveDocumentsRequest`

`top_k` constrained to 1-50 inclusive.

### `FormRequest`

Subset of `ChatRequest` — no thread management or Supabase submission.

---

## Response schemas (`app/schemas/response.py`)

### `ChatResponse`

```
ChatResponse
├── main_text: str
├── sources: list[SourceItem]
│   └── SourceItem
│       ├── slug: str               (Sanity document slug)
│       ├── text_id: str | None
│       ├── full_text: str          (full doc text; used quotes bolded with **)
│       ├── used_quotes: list[UsedQuote]
│       │   └── UsedQuote { number, text, timestamp }
│       ├── timestamp_range: str | None
│       └── score: float | None     (cosine similarity)
└── thread_id: uuid.UUID | None
```

### `ErrorResponse`

```
ErrorResponse
├── code: str           (stable machine-readable identifier)
├── message: str
├── request_id: str | None
└── details: dict | None
```

### Other response types

| Schema | Used by |
|---|---|
| `RetrieveDocumentsResponse` | `POST /chat/documents` |
| `HealthResponse` | `GET /health` |
| `RateLimitInfoResponse` | `GET /user` rate limit endpoints |
| `ConfigInfoResponse` | `GET /config/` |
| `AvailableConfigurationsResponse` | `GET /config/available-configs` |
| `SimpleConfigurationsResponse` | `GET /config/simple-configs` |
| `EnhancedConfigurationsResponse` | `GET /config/enhanced-configs` |
| `FormFullResponse` | Form endpoints |
