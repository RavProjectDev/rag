---
name: add-endpoint
description: Add a new HTTP route to the RAG API. Use when creating any new FastAPI endpoint — covers schema definition, service function, router file, registration in main.py, constants, and spec sync.
---
# Skill: Add a New Endpoint

Use this skill when adding any new HTTP route to the RAG API.

---

## Checklist (follow in order)

### 1. Define schemas

Add request body to `app/schemas/requests.py`:

```python
class MyRequest(BaseModel):
    question: str
    top_k: int = 10
```

Add response body to `app/schemas/response.py`:

```python
class MyResponse(BaseModel):
    result: str
    request_id: str
```

Never define Pydantic models inline inside a router file.

---

### 2. Create the service function

Add to the relevant file in `app/services/` (or create a new one):

```python
async def my_operation(question: str, settings: Settings) -> str:
    # business logic here
    ...
```

Rules:
- `async` if it calls any external API
- Raises from `app/exceptions/` only — never bare `Exception`
- Uses `asyncio.wait_for(..., timeout=settings.external_api_timeout)` for external calls

---

### 3. Create the router file

`app/api/v1/my_feature.py`:

```python
import logging
import uuid
from fastapi import APIRouter, HTTPException, Depends
from rag.app.exceptions.base import BaseAppException
from rag.app.schemas.requests import MyRequest
from rag.app.schemas.response import MyResponse
from rag.app.services.auth import verify_jwt_token
from rag.app.services.my_service import my_operation
from rag.app.core.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=MyResponse)
async def handler(
    body: MyRequest,
    user_id: str = Depends(verify_jwt_token),
) -> MyResponse:
    settings = get_settings()
    request_id = uuid.uuid4().hex
    try:
        result = await my_operation(body.question, settings)
        return MyResponse(result=result, request_id=request_id)
    except BaseAppException as e:
        raise HTTPException(status_code=e.status_code, detail={"code": e.code, "error": e.message})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "internal_server_error", "message": str(e)})
```

Rules:
- `verify_jwt_token` on every non-health endpoint — no exceptions (see `api-conventions` rule for webhook carve-out)
- Router catches exceptions and re-raises as `HTTPException` — services never raise `HTTPException`
- Path: plural nouns, kebab-case (`/documents`, `/available-configs`)

---

### 4. Register the router in `app/main.py`

```python
from rag.app.api.v1.my_feature import router as my_feature_router

app.include_router(my_feature_router, prefix="/api/v1/my-feature", tags=["my-feature"])
```

---

### 5. Add constants (if needed)

If the endpoint introduces new tuneable defaults, add them to the relevant file in
`app/core/constants/` and mirror each one in `Settings` in `app/core/config.py`.

Never hardcode values inside the router or service.

---

### 6. Update `specs/current/api-surface.md`

Add a new section describing:
- Method + path
- Auth requirement
- Request schema
- Response schema
- Error codes

This is required if `changes.yaml` has `apiContract: true`.

---

### 7. Run smoke tests

```bash
cd smoke_tests && ./run_smoke.sh
```

All REQ-001 through REQ-006 must PASS before the work is done.
