# Contributing

This project uses **spec-driven development**: all non-trivial changes flow through a structured specification pipeline before any code is written. This document explains how that pipeline works.

---

## Core idea

The `specs/` directory is the single source of truth for what the system is and where it is going. Code follows specs — not the other way around. Before implementing a change you write down *what* the end state should look like; after implementing it you update the specs to reflect reality.

---

## Directory layout

```
specs/
├── README.md              ← workflow overview
├── current/               ← live description of the system as it is right now
│   ├── architecture.md    ← stack, service map, request lifecycle
│   ├── api-surface.md     ← all routes, request/response contracts, auth
│   ├── data-models.md     ← enums, Pydantic schemas, domain models
│   ├── configuration.md   ← Settings hierarchy, constants, env vars
│   ├── infrastructure.md  ← Docker, GCP, EC2, CI/CD
│   └── app-requirements.md← REQ-NNN acceptance criteria for smoke tests
├── future/                ← one file per approved or in-flight proposal
│   └── TEMPLATE.md
└── migrations/            ← ordered execution plans (in-progress + complete)
    └── TEMPLATE.md
```

`current/` must be accurate at every commit. Stale specs are worse than no specs.

---

## Workflow for a new change

### 1. Write a future spec

Copy `specs/future/TEMPLATE.md` to `specs/future/<short-slug>.md` and fill in:

- **Motivation** — what problem this solves
- **Desired end state** — concrete description of how the system will look (reference specific files, modules, config values, API contracts)
- **Affected components** — checklist of files that will change
- **Success criteria** — how we know the change is done and correct

Mark the status field as `draft` initially, then `approved` when ready to execute.

### 2. Create a migration doc

Copy `specs/migrations/TEMPLATE.md` to `specs/migrations/YYYYMMDD-<slug>.md`. Fill in:

- **From** — link to the current-spec section being migrated away from
- **To** — link to the future spec
- **Steps** — ordered, checkable list of implementation tasks. The last two steps are always: update `specs/current/` and run smoke tests.
- **Rollback plan** — how to revert if something goes wrong

Do this *before* writing any code.

### 3. Implement

Work through the Steps checklist. Check off each item as you complete it. Do not edit completed steps retroactively — they form a change record.

### 4. Update `specs/current/`

After implementation, update every affected `current/` file to describe the system as it now is. The Cursor rule `spec-sync.mdc` defines exactly which files to touch:

| What changed | File to update |
|---|---|
| Stack, request lifecycle, module responsibilities | `architecture.md` |
| Any route, request/response field, or auth behaviour | `api-surface.md` |
| Any enum, schema, or response shape | `data-models.md` |
| Any constant, Settings field, or default value | `configuration.md` |
| Any environment, CI workflow, or external service | `infrastructure.md` |

If the change introduces or modifies observable API behaviour, add a new `REQ-NNN` entry to `app-requirements.md` following the existing format: `trigger → expected HTTP status → expected body criteria → pass/fail definition`.

### 5. Run smoke tests

```bash
cd smoke_tests && ./run_smoke.sh
```

Evaluate the output against every `REQ-NNN` in `specs/current/app-requirements.md` and record a verdict (`PASS` / `FAIL` / `SKIP`) in the migration doc's verdict table. Do not mark the migration complete if any requirement is `FAIL`.

### 6. Close out

- Set the migration status to `complete` and fill in the `Completed` date.
- Set the future spec status to `superseded`.
- Leave the migration file in place — completed migrations form the project's change history.

---

## Cursor AI agent workflow

The `.cursor/rules/` directory contains rules that instruct the AI agent to follow this workflow automatically:

| Rule file | When it applies |
|---|---|
| `future-execute.mdc` | When you point the agent at a `specs/future/*.md` file and ask it to implement |
| `spec-sync.mdc` | After any implementation — enforces `specs/current/` update + smoke test |
| `project-overview.mdc` | Always active — gives the agent grounded context |
| `api-conventions.mdc` | Always active — FastAPI patterns and route conventions |
| `python-standards.mdc` | Always active — code style and structure |
| `schema-conventions.mdc` | Always active — Pydantic model conventions |
| `constants-pattern.mdc` | Always active — where and how to define constants |

When starting an agent session on a significant change, reference the relevant spec files explicitly:

```
@specs/current/architecture.md @specs/future/my-proposal.md — implement this
```

---

## What belongs in specs vs. other docs

| Location | Purpose |
|---|---|
| `specs/current/` | Architectural truth — what the system is |
| `specs/future/` | Proposals — what the system should become |
| `specs/migrations/` | Execution records — how we got from one state to another |
| `.cursor/rules/` | AI agent instructions — how to write code in this repo |
| `README.md` | Onboarding and operational quick-start |
| `smoke_tests/SPEC.md` | Semantic acceptance criteria for the chat endpoint |
| `scripts/README.md` | Data ingestion script usage |

---

## Diagrams

### Cursor runtime & project structure

```mermaid
flowchart TD
    %% Define styles
    classDef cursorRuntime fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px;
    classDef rules fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
    classDef skills fill:#d5e8d4,stroke:#82b366,stroke-width:2px;
    classDef mcps fill:#f8cecc,stroke:#b85450,stroke-width:2px;
    classDef specs fill:#e1d5e7,stroke:#9673a6,stroke-width:2px;
    classDef specGroup fill:#f3e8ff,stroke:#9673a6,stroke-width:2px;

    subgraph L1 [CURSOR RUNTIME]
        subgraph R0 [Rules .cursor/rules/]
            R1["project-overview<br>alwaysApply: true<br>Scope: all sessions"]
            R2["spec-workflow<br>alwaysApply: true<br>Scope: all sessions"]
            R3["api-conventions<br>alwaysApply: false<br>Glob: app/api/**/*.py"]
            R4["python-standards<br>alwaysApply: false<br>Glob: **/*.py"]
            R5["constants-pattern<br>alwaysApply: false<br>Glob: app/core/constants/*.py"]
            R6["schema-conventions<br>alwaysApply: false<br>Glob: app/schemas/*.py"]
        end

        subgraph S0 [Skills .cursor/skills/]
            S1[add-configuration]
            S2[add-endpoint]
            S3[add-exception]
            S4[mutate-endpoint]
            S5[switch-configuration]
            S6[smoke-test]
            S7[run-local]
        end
    end

    subgraph M0 [MCPs .cursor/mcp.json]
        M1["pinecone<br>@pinecone-database/mcp<br>Auth: PINECONE_API_KEY"]
        M2["mongodb<br>mongodb-mcp-server<br>Auth: MONGODB_URI"]
        M3["rag-api<br>@modelcontextprotocol/server-fetch<br>Auth: none"]
    end

    subgraph SP0 [SPECS specs/]
        subgraph SC [specs/current/ — Canonical Truth]
            SC1[architecture.md]
            SC2[api-surface.md]
            SC3[data-models.md]
            SC4[configuration.md]
            SC5[infrastructure.md]
        end

        subgraph SF [specs/features/ & specs/bugs/ — Work Items]
            SF1["what.md<br>User story / problem"]
            SF2["how.md<br>Implementation approach"]
            SF3["test.md<br>Verification cases"]
            SF4["changes.yaml<br>Scope + canon flags"]
        end

        subgraph ST [specs/tools/ — Scaffolding]
            ST1["spec CLI<br>spec create feature &lt;slug&gt;<br>spec create bug &lt;slug&gt;"]
            ST2["templates/<br>feature/ + bug/<br>(canonical scaffolds)"]
        end

        SI["specs/index.yaml — Registry of all features & bugs"]
    end

    %% Edges
    R2 -- "gates mutations via canonicalChanges" --> SF4
    S6 -. "reads app-requirements.md" .-> SC
    M1 -.-> M3

    %% Apply Classes
    class L1 cursorRuntime;
    class R0 rules;
    class S0 skills;
    class M0 mcps;
    class SP0 specs;
    class SC,SF,ST specGroup;
```

### End-to-end spec-driven development workflow

```mermaid
flowchart TD
    %% Define styles
    classDef human fill:#f5f5f5,stroke:#555,stroke-width:2px;
    classDef cursor fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px;
    classDef rules fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
    classDef specs fill:#e1d5e7,stroke:#9673a6,stroke-width:2px;
    classDef skills fill:#d5e8d4,stroke:#82b366,stroke-width:2px;
    classDef code fill:#f8cecc,stroke:#b85450,stroke-width:2px;
    classDef git fill:#f0e6ff,stroke:#9673a6,stroke-width:2px;

    subgraph Human
        P1_1("Identify need"):::human
        P1_2["spec create feature/bug"]:::human
        P1_4["Fill what.md"]:::human
        P1_5["Fill how.md"]:::human
        P1_6["Fill test.md"]:::human
        P1_7["Fill changes.yaml"]:::human
        P2_1("Open Cursor chat"):::human
        P2_1b["Paste spec prompt"]:::human
        P6_4{"Review changes"}:::human
    end

    subgraph Cursor IDE
        P2_2["Inject rules"]:::cursor
        P2_4["Detect open files"]:::cursor
        P6_3["Present summary"]:::cursor
    end

    subgraph Rules
        P2_3["project-overview.mdc"]:::rules
        P2_5["glob-scoped rules"]:::rules
    end

    subgraph Specs
        P1_3["Scaffold 4 files"]:::specs
        P3_1["Read spec files"]:::specs
        P3_2{"All 4 present?"}:::specs
        P3_3["Check canonical flags"]:::specs
        P3_4["Read changeScope"]:::specs
        P4_4["Validate test.md"]:::specs
        P6_1{"canon flag = true?"}:::specs
        P6_2["Sync specs/current/"]:::specs
    end

    subgraph Skills
        P4_1{"Skill invoked?"}:::skills
        P4_2["Load SKILL.md"]:::skills
        P5_1["smoke-test skill"]:::skills
        P5_2{"REQ all PASS?"}:::skills
        P6_7(((DONE))):::skills
    end

    subgraph Codebase
        P3_2b["Block - ask human"]:::code
        P4_3["Implement per how.md"]:::code
        P5_3["Fix failure"]:::code
        P6_4b["Request revisions"]:::code
    end

    subgraph Git
        P6_5["git add & commit"]:::git
        P6_6["git push"]:::git
    end

    %% Phase 1
    P1_1 --> P1_2 --> P1_3
    P1_3 --> P1_4 & P1_6
    P1_4 --> P1_5
    P1_6 --> P1_7

    %% Phase 2
    P1_5 & P1_7 --> P2_1
    P2_1 --> P2_1b
    P2_1b --> P2_2
    P2_2 --> P2_3 & P2_4
    P2_4 --> P2_5 & P3_1

    %% Phase 3
    P3_1 --> P3_2
    P3_2 -- No --> P3_2b
    P3_2 -- Yes --> P3_3
    P3_3 --> P3_4

    %% Phase 4
    P3_4 --> P4_1
    P4_1 -- Yes --> P4_2
    P4_1 -- No --> P4_3
    P4_2 --> P4_3
    P4_3 --> P4_4

    %% Phase 5
    P4_4 --> P5_1
    P5_1 --> P5_2
    P5_2 -- Fail --> P5_3
    P5_3 -.-> P4_3
    P5_2 -- Pass --> P6_1

    %% Phase 6
    P6_1 -- Yes --> P6_2
    P6_1 -- No --> P6_3
    P6_2 --> P6_3
    P6_3 --> P6_4
    P6_4 -- Reject --> P6_4b
    P6_4b -.-> P4_3
    P6_4 -- Approve --> P6_5
    P6_5 --> P6_6 --> P6_7
```

---

## Conventions

- Keep `current/` in sync with the codebase. Update specs in the same commit as the code they describe.
- Prefer links over duplication across spec files.
- One future spec per logical change. Large changes should be broken into sequenced proposals.
- Migration files are write-once — do not edit completed steps.
- If a future spec is abandoned, mark it `superseded` with a note explaining why.
