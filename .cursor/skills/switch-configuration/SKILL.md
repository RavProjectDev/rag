---
name: switch-configuration
description: Switch the active vector backend (Pinecone/MongoDB), LLM, embedding model, or environment. Use when changing any runtime configuration switch — all switches are pure config changes with no code edits required.
---
# Skill: Switch Configuration

Use this skill when changing the active vector backend, LLM, or embedding model.
All switches are pure config changes — no code edits required unless noted.

---

## Switch vector backend

**File**: `app/core/constants/app.py`

```python
# Pinecone (default)
DATABASE_CONFIGURATION = DataBaseConfiguration.PINECONE

# MongoDB Atlas Vector
DATABASE_CONFIGURATION = DataBaseConfiguration.MONGO
```

**Required `.env` vars per backend**:

| Backend | Required in `.env` |
|---|---|
| `PINECONE` | `PINECONE_API_KEY` |
| `MONGO` | `MONGODB_URI` |

**What happens at startup**: `app/core/constants/__init__.py` conditionally imports
backend-specific constants; `app/core/lifespan.py` instantiates the matching connection class.
No other code needs to change — `EmbeddingConnection` abstracts `retrieve()` for all routers.

---

## Switch LLM

**File**: `app/core/constants/llm.py`

```python
# Gemini Flash (default)
LLM_CONFIGURATION = LLMModel.GEMINI_FLASH

# GPT
LLM_CONFIGURATION = LLMModel.GPT_4   # or whichever GPT variant is in the enum
```

**Required `.env` vars per LLM**:

| Model | Required in `.env` |
|---|---|
| Gemini | `GEMINI_API_KEY` |
| GPT | `OPENAI_API_KEY` |

---

## Switch embedding model

**File**: `app/core/constants/llm.py`

```python
EMBEDDING_CONFIGURATION = EmbeddingConfiguration.GEMINI   # default
EMBEDDING_CONFIGURATION = EmbeddingConfiguration.OPENAI
```

> Changing the embedding model invalidates any existing vector index built with a
> different model. Re-run the ingestion pipeline after switching.

---

## Switch environment

**File**: `app/core/constants/app.py`

```python
ENVIRONMENT = Environment.LOCAL   # JWT bypassed, dev outputs enabled
ENVIRONMENT = Environment.STG     # JWT enforced
ENVIRONMENT = Environment.PRD     # JWT enforced
```

`LOCAL` and `TEST` bypass `verify_jwt_token`. All other environments enforce it.

---

## After any switch

1. Restart the server:
   ```bash
   PYTHONPATH=.. uvicorn rag.app.main:app --port 8000
   ```
2. Run smoke tests:
   ```bash
   cd smoke_tests && ./run_smoke.sh
   ```
3. All REQ-001 through REQ-006 must PASS.

---

## What NOT to do

- Do not set backend values in `.env` — they belong in `app/core/constants/`, not secrets.
- Do not hardcode environment-specific values in service code.
- Do not add `if DATABASE_CONFIGURATION == ...` checks outside of `app/core/constants/__init__.py`.
