---
name: add-configuration
description: Add a new configuration value, feature flag, timeout, model name, or external service URL to the RAG or Webhook API. Use when introducing any new tuneable setting, secret, constant, or enum value.
---
# Skill: Add a New Configuration Value

Use this skill whenever you need to introduce a new tuneable value, feature flag,
timeout, model name, or external service URL.

---

## Decision tree — where does this value live?

```
Is it a secret? (API key, token, password, URI with credentials)
├── YES → .env only (Step A)
└── NO  → Is it backend-conditional? (only valid when PINECONE or MONGO is active)
           ├── YES → backend-specific constants file + conditional import (Step C)
           └── NO  → non-secret default (Step B)
```

---

## Step A — Secret value

Secrets never appear in `app/core/constants/`. They live only in `.env`.

1. Add a field to `Settings` (or `SharedSettings` if shared with the Webhook app) in
   `app/core/config.py`:

   ```python
   class Settings(SharedSettings):
       my_api_key: str | None = None   # optional secret
       my_required_key: str            # required secret — no default
   ```

2. Document the env var name in `specs/current/configuration.md` under "Secrets".

3. Consumers call `get_settings().my_api_key` — never read `os.environ` directly.

---

## Step B — Non-secret default (most common case)

### 1. Add the constant to the right domain file in `app/core/constants/`

| Domain | File |
|---|---|
| Runtime switches, environment, feature flags | `app.py` |
| LLM model, embedding model, timeouts | `llm.py` |
| MongoDB collection names, retry settings | `mongo.py` |
| Pinecone index, namespace, host | `pinecone.py` |
| Supabase project URL | `supabase.py` |
| Upstash Redis URL, rate-limit defaults | `upstash.py` |
| GCP project, Vertex region | `vertex.py` |
| New domain with ≥ 2 constants | Create `app/core/constants/<domain>.py` |

```python
# Example: app/core/constants/llm.py
MY_NEW_TIMEOUT = 30
```

### 2. Export it from `app/core/constants/__init__.py`

```python
from rag.app.core.constants.llm import (
    ...,
    MY_NEW_TIMEOUT,   # ✅ add here
)
```

### 3. Mirror it as a `Settings` field in `app/core/config.py`

```python
import rag.app.core.constants as C

class Settings(SharedSettings):
    ...
    my_new_timeout: int = C.MY_NEW_TIMEOUT
```

Use `SharedSettings` instead of `Settings` if the Webhook app also needs this value.

### 4. Consume via `get_settings()`

```python
# ✅ correct — env-overridable at runtime
settings = get_settings()
timeout = settings.my_new_timeout

# ❌ wrong — bypasses the override path
from rag.app.core.constants import MY_NEW_TIMEOUT
```

---

## Step C — Backend-conditional constant (Pinecone / Mongo only)

Add to the backend-specific constants file (e.g. `pinecone.py`), then add the
conditional import in `app/core/constants/__init__.py`:

```python
if DATABASE_CONFIGURATION == DataBaseConfiguration.PINECONE:
    from rag.app.core.constants.pinecone import (  # noqa: F401
        ...,
        MY_PINECONE_CONSTANT,
    )
```

Consumers access it via `getattr(C, "MY_PINECONE_CONSTANT", None)` or by checking
`settings.database_configuration` before use.

---

## Step D — New enum value

If the configuration introduces a new option for an existing enum
(e.g. a new LLM model, a new embedding strategy):

1. Add the value to the enum in `app/schemas/data.py`:

   ```python
   class LLMModel(str, Enum):
       GEMINI_FLASH = "gemini-2.5-flash"
       GPT_4 = "gpt-4"
       MY_NEW_MODEL = "my-new-model"   # ✅ add here
   ```

2. Update `app/services/llm.py` (or whichever service branches on this enum)
   to handle the new value.

3. Add the required secret/key to `Settings` if the new option needs one.

---

## After adding any configuration

1. Update `specs/current/configuration.md`:
   - Add the field to the correct table (Secrets or Non-secret defaults)
   - Add the env var name and description
2. Update `app/core/constants/` file map in that doc if you created a new file.
3. Restart the server — `get_settings()` is `@lru_cache()` and only reads `.env` on first call.
4. Run smoke tests:
   ```bash
   cd smoke_tests && ./run_smoke.sh
   ```

---

## What NOT to do

- Do not put secrets in `app/core/constants/` — not even as a placeholder
- Do not hardcode values inside services or routers — always go through `get_settings()`
- Do not import from constants sub-modules directly (e.g. `from rag.app.core.constants.llm import ...`) — import from the package (`rag.app.core.constants`)
