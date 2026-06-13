# Future State: [Title]

> Status: draft | approved | in-progress | superseded
> Author: [name]
> Created: YYYY-MM-DD

---

## Motivation

Why is this change needed? What problem does it solve or opportunity does it enable?

---

## Desired end state

Describe the system as it will look when this proposal is fully implemented.
Be concrete — reference specific files, modules, config values, and API contracts where possible.

---

## Affected components

- [ ] `app/core/constants/` — which file(s) and what changes
- [ ] `app/schemas/` — new or modified enums / models
- [ ] `app/services/` — new or modified service logic
- [ ] `app/api/v1/` — new or modified routes
- [ ] `app/db/` — new or modified connections
- [ ] `.env.example` — new secrets required
- [ ] `specs/current/` — which docs need updating
- [ ] `specs/current/app-requirements.md` — any new or changed behavioural requirements
- [ ] CI/CD — any workflow changes
- [ ] Infrastructure — any new external services

---

## Success criteria

How do we know this is done and working correctly?

All existing requirements in `specs/current/app-requirements.md` must still pass after this change.
If this change alters observable API behaviour, add new `REQ-NNN` entries to that file as part of the migration.

Additional criteria specific to this change:

- [ ] criterion 1
- [ ] criterion 2

---

## Migration plan

Link to `specs/migrations/YYYYMMDD-<slug>.md` once a migration file is created.
The migration file contains the mandatory smoke test verification step.

---

## Open questions

- Question 1?
- Question 2?

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| YYYY-MM-DD | | |
