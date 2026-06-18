---
name: add-exception
description: Add a new typed exception to the RAG exception hierarchy. Use when a new error condition needs its own type — covers choosing the right domain file, class structure, status codes, and raising/catching patterns.
---
# Skill: Add a Typed Exception

Use this skill when a new error condition needs to be represented in the exception hierarchy.
Never raise bare `Exception` or `HTTPException` from service code.

---

## Exception hierarchy

```
BaseAppException          (app/exceptions/base.py)
 ├── DataBaseException     (app/exceptions/db.py)      — 500
 │    ├── RetrievalException                            — 502
 │    ├── RetrievalTimeoutException                     — 504
 │    ├── InsertException                               — 500
 │    └── NoDocumentFoundException                      — 404
 ├── EmbeddingException    (app/exceptions/embedding.py)
 ├── LLMBaseException      (app/exceptions/llm.py)
 └── UploadException       (app/exceptions/upload.py)
```

---

## Step 1 — Choose the right domain file

| Your error is about… | File |
|---|---|
| Vector retrieval, DB writes, document not found | `app/exceptions/db.py` |
| Embedding generation or configuration | `app/exceptions/embedding.py` |
| LLM call, prompt generation, JSON parsing | `app/exceptions/llm.py` |
| Ingestion / upload pipeline | `app/exceptions/upload.py` |
| Does not fit any domain | Subclass `BaseAppException` directly in a new file |

---

## Step 2 — Add the class

```python
from rag.app.exceptions.base import BaseAppException          # or domain parent

class MyDomainException(DataBaseException):   # or appropriate parent
    status_code: int = 422                    # HTTP status returned to client
    code: str = "my_domain_error"            # machine-readable code in error envelope
    description: str = "Human-readable default message."

    def __init__(self, message: str | None = None):
        self.message = message or self.description
        super().__init__(self.message)
```

Rules:
- `status_code`, `code`, `description` are **class attributes** — not instance vars
- `code` must be unique across the entire hierarchy — grep before adding
- `description` is the fallback used when no `message` is passed
- The `__init__` signature must accept `message: str | None = None`

---

## Step 3 — No manual registration needed

`app/exceptions/__init__.py` auto-discovers every subclass of `BaseAppException` at import time
using `pkgutil` + `importlib`. Just adding the class to the correct file is enough.

---

## Step 4 — Raise it from the service layer

```python
# ✅
raise MyDomainException("Specific context about what failed")

# ✅ — use default description
raise MyDomainException()

# ❌ — never do this in a service
raise HTTPException(status_code=422, detail="...")
raise Exception("something broke")
```

---

## Step 5 — Catch it in the router (if needed)

The standard router catch-all already handles `BaseAppException`:

```python
except BaseAppException as e:
    raise HTTPException(status_code=e.status_code, detail={"code": e.code, "error": e.message})
```

Add a specific `except MyDomainException` block only if the router needs to respond
differently (e.g. return a 200 with a fallback value instead of an error).
