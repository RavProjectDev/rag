# Spec-Driven Development — Framework Guide

This directory is the single source of truth for **what this system is**, **what it should
become**, and **how to get from one state to the other**.

AI-agent instructions live separately in `.cursor/rules/`.

---

## Directory structure

```
specs/
├── README.md              ← this file
├── index.yaml             ← registry of all features and bugs
├── current/               ← canonical system state (architecture, API, data models, etc.)
│   ├── architecture.md
│   ├── api-surface.md
│   ├── data-models.md
│   ├── configuration.md
│   └── infrastructure.md
├── features/              ← feature specs (one subdirectory per feature)
│   └── <0001_slug>/
│       ├── what.md        ← user story, problem, acceptance criteria
│       ├── how.md         ← implementation approach, changes, constraints
│       ├── test.md        ← verification cases, regression, edge cases
│       └── changes.yaml   ← scope declaration and canonical change flags
├── bugs/                  ← bug specs (same structure as features)
│   └── <0001_slug>/
│       ├── what.md
│       ├── how.md
│       ├── test.md
│       └── changes.yaml
└── tools/
    ├── spec               ← CLI to scaffold new specs
    └── templates/         ← canonical templates (feature/ and bug/)
```

---

## Core Rules

### R1 — Canonical Truth
`specs/current/` is the single source of truth. No implicit override is allowed.

### R2 — Canonical Mutation Gate
If all `canonicalChanges` flags in `changes.yaml` are `false`: `specs/current/` **must not change**.
If any are `true`: the update is allowed only with explicit validation; sync `specs/current/` after.

### R3 — Spec Completeness
A spec is **invalid** unless it contains `what.md`, `how.md`, `test.md`, and `changes.yaml`.

### R4 — Execution Flow
1. Read spec
2. Validate completeness (R3)
3. Analyse impact
4. Check `canonicalChanges`
5. Implement minimal change
6. Validate against `test.md`
7. Sync canonical (if R2 allows)

### R5 — No Scope Expansion
No changes outside `affectedAreas` or `changeScope` declared in `changes.yaml`.

---

## Creating a new spec

Use the `spec` CLI from the repo root:

```bash
specs/tools/spec create feature <slug>
specs/tools/spec create bug <slug>
```

This will:
- Assign the next sequential 4-digit ID
- Scaffold the directory with all four required files (from templates)
- Register the entry in `specs/index.yaml`

---

## Definition of Done

A spec is complete only when:

- Implementation matches `how.md`
- All test cases in `test.md` pass
- No unauthorised canonical changes were made
- `index.yaml` reflects the new entry

---

## Relationship to other documentation

| Location | Purpose |
|---|---|
| `specs/current/` | Canonical truth — what the system is right now |
| `specs/features/` | Intent — what the system should do next |
| `specs/bugs/` | Deviation — where the system behaves incorrectly |
| `specs/index.yaml` | Registry of all tracked features and bugs |
| `.cursor/rules/` | AI-agent instructions — how to write code in this repo |
| `smoke_tests/SPEC.md` | Semantic acceptance criteria for the chat endpoint |
