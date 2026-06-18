---
name: mutate-endpoint
description: Change the behaviour, request shape, or response shape of an existing RAG API route. Use when modifying an existing endpoint — covers additive vs breaking changes, schema updates, service logic, error handling, rate limiting, and spec sync.
---
# Skill: Mutate an Existing Endpoint

Use this skill when changing the behaviour, request shape, or response shape of an
existing route — without replacing it entirely.

---

## Before you touch anything

1. Read the current contract in `specs/current/api-surface.md` for the target route.
2. Identify whether the change is **additive** (new optional fields) or **breaking**
   (removing fields, changing types, changing HTTP semantics).
3. Set `canonicalChanges.apiContract: true` in `changes.yaml` if the public contract changes.

---

## Adding an optional request field

In `app/schemas/requests.py`, add with a default:

```python
class ChatRequest(BaseModel):
    question: str
    top_k: int = 5          # ✅ existing
    name_spaces: list[str] | None = None  # ✅ new optional — safe, non-breaking
```

Never remove or rename an existing required field without a versioning decision.

---

## Adding a response field

In `app/schemas/response.py`, add with a default so old clients don't break:

```python
class ChatResponse(BaseModel):
    main_text: str
    sources: list[dict]
    thread_id: uuid.UUID | None = None   # ✅ existing
    request_id: str | None = None        # ✅ new optional field
```

---

## Changing service logic

Locate the service function called by the router (never add logic directly to the router).
Make the minimal change in `app/services/`.

Rules:
- Keep the function signature backward compatible unless all callers are updated together
- Wrap new external calls in `asyncio.wait_for(..., timeout=settings.external_api_timeout)`
- Raise typed exceptions only — never bare `Exception`

---

## Changing error behaviour

If the endpoint should now return a different HTTP status or error code for an existing
condition, update the `except` block in the router:

```python
# Before
except DataBaseException as e:
    raise HTTPException(status_code=500, ...)

# After — more specific
except NoDocumentFoundException as e:
    return ChatResponse(main_text=e.message_to_ui, sources=[])
except DataBaseException as e:
    raise HTTPException(status_code=502, ...)
```

Keep the catch order specific → general (subclasses before parent classes).

---

## Rate limiting

If the mutated endpoint now has new error paths, add `redis_conn.decrement_user_rate_limit(user_id)`
in any new `except` block that represents a user-facing failure (timeout, no document, app error).

---

## After the change

1. Update `specs/current/api-surface.md` to reflect the new contract.
2. Add a new `REQ-NNN` to `specs/current/app-requirements.md` if observable behaviour changed.
3. Run smoke tests:
   ```bash
   cd smoke_tests && ./run_smoke.sh
   ```
4. All REQ-001 through REQ-006 must PASS.
