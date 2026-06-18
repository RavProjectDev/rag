---
name: smoke-test
description: Run and evaluate the smoke test suite against the RAG API. Use after any implementation to verify the system is healthy — smoke tests are mandatory before marking a spec done.
---
# Skill: Run and Evaluate Smoke Tests

Use this skill after any implementation to verify the system is healthy.
Smoke tests are **mandatory** — never mark a spec done without a passing run.

---

## Pre-requisites

The server must be running before the smoke runner starts:

```bash
# From the repo parent directory
PYTHONPATH=.. uvicorn rag.app.main:app --port 8000
```

The runner waits up to 120 seconds for the health endpoint to respond.

---

## Run the suite

```bash
cd smoke_tests && ./run_smoke.sh
```

---

## Requirements evaluated (from `specs/current/app-requirements.md`)

| ID | Trigger | What must be true |
|---|---|---|
| REQ-001 | Health endpoint | `GET /api/v1/health` returns 200 within 120 s |
| REQ-002 | Nonsense input | HTTP 200; `main_text` refuses gracefully; `sources: []` |
| REQ-003 | On-topic question | HTTP 200; `main_text` is substantive; `sources` has ≥ 1 item |
| REQ-004 | Non-empty sources | Each source has `slug`, `full_text`, `used_quotes` (≥ 1 item) |
| REQ-005 | No internal errors | No HTTP 500, no `internal_server_error`, no tracebacks |
| REQ-006 | `main_text` ends with `!` | `main_text[-1] == "!"` |

Evaluate **semantically** — do not match exact strings.

---

## Producing a verdict

For each REQ, output one of:

- **PASS** — requirement is satisfied
- **FAIL** — requirement is not satisfied (include what was observed)
- **SKIP** — requirement is not applicable to this change (include reason)

If any REQ is **FAIL**, do not mark the spec complete. Investigate and fix first.

---

## Common failure modes

| Symptom | Likely cause |
|---|---|
| REQ-001 FAIL — health never responds | Server not running, wrong port, import error at startup |
| REQ-002 FAIL — nonsense returns a real answer | Retrieval is returning low-quality matches; `NoDocumentFoundException` not being raised |
| REQ-003 FAIL — on-topic returns refusal | Vector index is empty or wrong namespace; embedding model mismatch |
| REQ-004 FAIL — missing `slug` or `used_quotes` | Schema change broke source serialisation; check `ChatResponse` and source assembly in `chat.py` |
| REQ-005 FAIL — 500 in response | Unhandled exception in service; check server logs |
| REQ-006 FAIL — `main_text` missing `!` | `postprocess_main_text` not being called; check `app/services/llm.py` |

---

## Interpreting exit codes

| Exit code | Meaning |
|---|---|
| `0` | All tests passed |
| `1` | One or more tests failed or health never became ready |
