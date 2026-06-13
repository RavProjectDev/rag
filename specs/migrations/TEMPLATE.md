# Migration: YYYYMMDD-[slug]

> Status: planned | in-progress | complete | rolled-back
> Author: [name]
> Started: YYYY-MM-DD
> Completed: YYYY-MM-DD

---

## From

Link to the `specs/current/` section or describe the state being migrated away from.

---

## To

Link to the `specs/future/<proposal>.md` that defines the target state.

---

## Pre-conditions

What must be true before this migration can begin?

- [ ] pre-condition 1
- [ ] pre-condition 2

---

## Steps

Check off each step as it is completed. Do not edit completed steps retroactively.

- [ ] **Step 1** — description
  - Detail / commands / files changed
- [ ] **Step 2** — description
- [ ] **Step 3** — update `specs/current/` docs to reflect the new state
- [ ] **Step 4** — run smoke tests and evaluate output (see Verification below)
- [ ] **Step 5** — all requirements in `specs/current/app-requirements.md` pass → mark complete

---

## Rollback plan

How to revert to the previous state if the migration fails partway through.

1. Step to revert change X
2. Step to revert change Y

---

## Verification

After implementing, an AI agent MUST run the smoke tests and evaluate the output
against `specs/current/app-requirements.md` before this migration can be marked complete.

**Run the tests:**

```bash
cd smoke_tests && ./run_smoke.sh
```

**Evaluate the output:**

Read the full terminal output from `run_smoke.sh` and check it against every `REQ-NNN`
requirement in `specs/current/app-requirements.md`. For each requirement produce a verdict:

- **PASS** — criteria met
- **FAIL** — criteria not met (describe what was wrong)
- **SKIP** — not applicable to this change (explain why)

Do not mark the migration complete if any requirement is FAIL.

### Requirement verdicts

| Requirement | Verdict | Notes |
|---|---|---|
| REQ-001 Health endpoint reachable | | |
| REQ-002 Nonsense input gracefully refused | | |
| REQ-003 Valid question returns substantive answer with sources | | |
| REQ-004 Sources contain required fields | | |
| REQ-005 No internal errors surface in responses | | |

---

## Notes

Anything that deviated from the plan, decisions made during execution, or lessons learned.
