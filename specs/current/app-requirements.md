# App Requirements — Behavioural Acceptance Criteria

> This is the canonical requirements document for the Rav Legacy RAG API.
> It is the source of truth for what "working correctly" means.
>
> After any change, an AI agent MUST run the smoke tests and evaluate
> the output against the criteria in this file before marking the migration complete.
>
> Last updated: 2026-06-13 (added REQ-006)

---

## How an agent should use this document

1. Run the smoke test suite:
   ```bash
   cd smoke_tests && ./run_smoke.sh
   ```
2. Capture the full terminal output.
3. Evaluate each requirement below against the actual output.
4. For each requirement, produce a verdict: **PASS**, **FAIL**, or **SKIP** (with reason).
5. If any requirement is FAIL, do not mark the migration complete — investigate and fix first.

Evaluate semantically, not by exact string match. Wording of responses will vary.

---

## REQ-001 — Health endpoint is reachable

**How to check**: the smoke test runner waits for `GET /api/v1/health` to return HTTP 200
before proceeding. If the service never becomes healthy within 120 seconds, the runner exits 1.

**Pass**: runner reaches the test steps (service came up healthy).
**Fail**: runner exits with "did not become healthy" error.

---

## REQ-002 — Nonsense input is gracefully refused

**Trigger**: `POST /api/v1/chat/` with a randomly generated alphanumeric string.

**HTTP status**: must be 200.

**`main_text`**: must communicate that the system cannot answer. Acceptable forms:
- Stating the topic is not in the database or transcripts
- Stating there is no relevant context or insufficient information to answer
- A polite explanation that the question is outside the scope of available teachings
- "No document found" or equivalent phrasing indicating no matching source material exists

**`main_text` must NOT**: present a substantive, factual answer about Rav Soloveitchik's teachings.

**`sources`**: must be an empty list `[]`.

---

## REQ-003 — Valid on-topic question returns a substantive answer with sources

**Trigger**: `POST /api/v1/chat/` with `"What does the Rav say about the purpose of marriage in a Jewish life?"`

**HTTP status**: must be 200.

**`main_text`**: must be a substantive response. It must:
- Be at least a few sentences long
- Clearly relate to Jewish thought, Rav Soloveitchik's teachings, marriage, or covenantal relationships
- NOT contain phrases like "no information", "not in the database", "insufficient context", "no document found", or equivalent refusals

**`sources`**: must contain at least one cited source.

---

## REQ-004 — Sources contain required fields

For any response where `sources` is non-empty, each source object must include:
- `slug` — a non-empty string (Sanity document identifier)
- `full_text` — a non-empty string
- `used_quotes` — a list with at least one item

---

## REQ-005 — No internal errors surface in responses

Neither test response body should contain:
- HTTP 500 status
- `"internal_server_error"` in the response body
- Python tracebacks or raw exception messages

---

## REQ-006 — `main_text` always ends with `!`

**Trigger**: `POST /api/v1/chat/` with any valid question (including the Test 2 question from REQ-003).

**HTTP status**: must be 200.

**`main_text`**: the final character of the string must be `!`.

**Pass**: `main_text[-1] == "!"`.
**Fail**: `main_text` does not end with `!`.

---

## Adding new requirements

When a new feature changes observable API behaviour, add a new `REQ-NNN` block here
following the same format: trigger, expected HTTP status, expected body criteria, pass/fail definition.

Keep criteria semantic — never assert exact strings, only meaning and structure.
