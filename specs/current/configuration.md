# Configuration — Current State

> Last updated: 2026-06-13
> Source: `app/core/config.py`, `app/core/webhook_config.py`, `app/core/constants/`

---

## Philosophy

Configuration is split into two tiers:

| Tier | Location | Contains |
|---|---|---|
| **Secrets** | `.env` (gitignored) | API keys, tokens, connection strings with credentials |
| **Non-secret defaults** | `app/core/constants/` | Model names, collection names, timeouts, URLs, feature flags |

Constants are the defaults. Every constant is mirrored as an env-overridable field in `Settings`.
To change non-secret config, edit the relevant constants file — never `.env`.

---

## Settings hierarchy (`app/core/config.py`, `app/core/webhook_config.py`)

```
SharedSettings              ← secrets + defaults shared by all app instances
    ├── Settings            ← full RAG API settings (extends SharedSettings)
    └── WebhookSettings     ← Sync Webhook API settings (extends SharedSettings)
```

`get_settings()` and `get_webhook_settings()` are both `@lru_cache()`-wrapped — settings are loaded once per process.

---

## `SharedSettings` fields

### Secrets (must be in `.env`)

| Field | Env var | Description |
|---|---|---|
| `mongodb_uri` | `MONGODB_URI` | Motor connection string |
| `gemini_api_key` | `GEMINI_API_KEY` | Google Gemini API key |

### Non-secret defaults

| Field | Default constant | Description |
|---|---|---|
| `mongodb_db_name` | `"rav_dev"` | MongoDB database name |
| `mongodb_vector_collection` | `"gemini_embeddings_v3"` | Active vector collection |
| `collection_index` | `"vector_index"` | Atlas Vector index name |
| `metrics_collection` | `"metrics"` | Metrics log collection |
| `exceptions_collection` | `"exceptions"` | Exception log collection |
| `google_cloud_project_id` | `"ravlegacyproject"` | GCP project |
| `vertex_region` | `"us-central1"` | Vertex AI region |
| `google_application_credentials` | `None` | Path to service account JSON (uses ADC if absent) |
| `embedding_configuration` | `EmbeddingConfiguration.GEMINI` | Active embedding model |
| `chunking_strategy` | `ChunkingStrategy.FIXED_SIZE` | Active chunking strategy |
| `vector_path` | `"vector"` | Field name for the vector in MongoDB documents |
| `database_configuration` | `DataBaseConfiguration.PINECONE` | Active vector backend |
| `pinecone_api_key` | `None` | Required when backend is PINECONE |
| `pinecone_environment` | `None` | Pinecone environment |
| `pinecone_index_name` | `"gemini"` | Active Pinecone index |
| `pinecone_namespace` | `None` | Active Pinecone namespace |
| `pinecone_host` | `None` | Pinecone host override |
| `environment` | `Environment.LOCAL` | Runtime environment |

---

## `Settings` fields (RAG API only)

### Secrets (must be in `.env`)

| Field | Env var | Description |
|---|---|---|
| `openai_api_key` | `OPENAI_API_KEY` | OpenAI API key |
| `cohere_api_key` | `COHERE_API_KEY` | Optional |
| `supabase_service_role_key` | `SUPABASE_SERVICE_ROLE_KEY` | Required for auth + query persistence |
| `supabase_anon_key` | `SUPABASE_ANON_KEY` | Required for client-side auth |
| `upstash_redis_rest_token` | `UPSTASH_REDIS_REST_TOKEN` | Optional — rate limiting disabled if absent |

### Non-secret defaults

| Field | Default | Description |
|---|---|---|
| `llm_configuration` | `LLMModel.GEMINI_FLASH` | Active LLM model |
| `external_api_timeout` | `60` s | `asyncio.wait_for` timeout for the full pipeline |
| `retrieval_timeout_ms` | `2000` ms | MongoDB retrieval operation timeout |
| `max_retry_attempts` | `3` | Retry count for retrieval |
| `retry_delay_seconds` | `1.0` s | Initial retry delay |
| `retry_backoff_multiplier` | `2.0` | Exponential backoff multiplier |
| `supabase_url` | `"https://clvgfczatixeajovypju.supabase.co"` | Supabase project URL |
| `upstash_redis_rest_url` | `"https://loved-monster-57712.upstash.io"` | Upstash REST URL |
| `rate_limit_max_requests` | `1,000,000` | Global rate limit per window |
| `rate_limit_window_seconds` | `10,000` s | Global rate limit window |
| `user_rate_limit_max_requests_per_month` | `100` | Per-user monthly cap |
| `dev_outputs` | `True` | Write full prompts to `dev_outputs/` when `True` |

---

## `WebhookSettings` fields (Webhook API only — `app/core/webhook_config.py`)

`WebhookSettings` extends `SharedSettings`. It does **not** require OpenAI, Supabase, Redis, or rate-limit vars.

### Secrets (must be in `.env`)

None beyond those inherited from `SharedSettings`.

### Non-secret defaults

| Field | Default | Description |
|---|---|---|
| `webhook_secret` | `None` | HMAC secret for verifying Sanity webhook signatures. Defined but **not yet wired** into request handlers — signature verification is not active. |

---

## `app/core/constants/` — file map

| File | Exports |
|---|---|
| `app.py` | `DATABASE_CONFIGURATION`, `ENVIRONMENT`, `DEV_OUTPUTS` |
| `llm.py` | `LLM_CONFIGURATION`, `EMBEDDING_CONFIGURATION`, `CHUNKING_STRATEGY`, `EXTERNAL_API_TIMEOUT` |
| `mongo.py` | `MONGODB_DB_NAME`, `MONGODB_VECTOR_COLLECTION`, `COLLECTION_INDEX`, `METRICS_COLLECTION`, `EXCEPTIONS_COLLECTION`, `VECTOR_PATH`, `COLLECTIONS`, `RETRIEVAL_TIMEOUT_MS`, `MAX_RETRY_ATTEMPTS`, `RETRY_DELAY_SECONDS`, `RETRY_BACKOFF_MULTIPLIER` |
| `pinecone.py` | `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, `PINECONE_ENVIRONMENT`, `PINECONE_HOST` |
| `supabase.py` | `SUPABASE_URL` |
| `upstash.py` | `UPSTASH_REDIS_REST_URL`, `RATE_LIMIT_MAX_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`, `USER_RATE_LIMIT_MAX_REQUESTS_PER_MONTH` |
| `vertex.py` | `GOOGLE_CLOUD_PROJECT_ID`, `VERTEX_REGION` |

`__init__.py` re-exports everything and conditionally imports Pinecone constants only when
`DATABASE_CONFIGURATION == DataBaseConfiguration.PINECONE`.

---

## Switching backends

To switch to MongoDB Atlas Vector:

1. In `app/core/constants/app.py` set `DATABASE_CONFIGURATION = DataBaseConfiguration.MONGO`
2. Ensure `MONGODB_URI` is set in `.env` and the Atlas cluster has the vector index configured
3. No code changes required — `EmbeddingConnection` abstracts the backend

To switch LLM:

1. In `app/core/constants/llm.py` set `LLM_CONFIGURATION = LLMModel.GPT_4`
2. Ensure `OPENAI_API_KEY` is present in `.env`
