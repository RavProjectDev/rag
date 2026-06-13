# Spec-Driven Development — Framework Guide

This directory is the single source of truth for **what this system is**, **what it should become**,
and **how to get from one state to the other**.

It is intentionally separate from AI-agent instructions (which live in `.cursor/rules/`).

---

## Directory structure

```
specs/
├── README.md              ← this file
├── current/               ← documented current state of the system
│   ├── architecture.md    ← overall design, stack, and service map
│   ├── api-surface.md     ← all routes, request/response contracts, auth
│   ├── data-models.md     ← enums, Pydantic schemas, domain models
│   ├── configuration.md   ← constants/, Settings hierarchy, env vars
│   └── infrastructure.md  ← Docker, GCP, EC2, CI/CD
├── future/                ← proposed future states (one file per proposal)
│   └── TEMPLATE.md
└── migrations/            ← in-flight and completed state transitions
    └── TEMPLATE.md
```

---

## How to use this framework

### Reading current state

The `current/` docs describe the system **as it exists right now**. They should be accurate
at every commit. Update them when behaviour changes.

Reference them in Cursor agent sessions with `@specs/current/architecture.md` to give the
agent grounded context before asking it to implement something.

### Proposing a future state

1. Copy `future/TEMPLATE.md` → `future/<short-slug>.md`
2. Fill it in: motivation, desired end state, affected components, success criteria.
3. Link to it from any related issue, PR description, or migration file.
4. Delete or archive it once the migration is complete.

### Planning a migration

1. Copy `migrations/TEMPLATE.md` → `migrations/YYYYMMDD-<slug>.md`
2. Fill in the from/to state, ordered steps, rollback plan, and verification criteria.
3. Tick off steps as they are completed.
4. Leave completed migration files in place — they form the project's change history.

---

## Relationship to other documentation

| Location | Purpose |
|---|---|
| `specs/current/` | Architectural truth — what the system is |
| `.cursor/rules/` | AI-agent instructions — how to write code in this repo |
| `README.md` (root) | Onboarding and operational runbook |
| `smoke_tests/SPEC.md` | Semantic acceptance criteria for the chat endpoint |
| `dev_outputs/README.md` | Debug prompt dump format |
| `scripts/README.md` | Ingestion and sync script usage |

---

## Conventions

- Keep `current/` docs in sync with the codebase. Stale specs are worse than no specs.
- Prefer links over duplication — `current/api-surface.md` links to `smoke_tests/SPEC.md` rather than copying it.
- Migration files are write-once — do not edit completed steps retroactively.
