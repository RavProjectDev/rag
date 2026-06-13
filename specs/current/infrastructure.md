# Infrastructure — Current State

> Last updated: 2026-06-13
> Source: `.github/workflows/`, `Dockerfile`, `smoke_tests/docker-compose.smoke.yml`

---

## Environments

| Environment | `ENVIRONMENT` value | Platform | Auth enforced |
|---|---|---|---|
| Production | `PRD` | GCP Cloud Run + EC2 | Yes |
| Staging | `STG` | GCP Cloud Run + EC2 | Yes |
| Local dev | `LOCAL` | Bare Python / uvicorn | No |
| Smoke test | `LOCAL` | Docker Compose | No |

---

## Local development

```bash
# From the repo parent directory (The Rav Legacy/AI/)
PYTHONPATH=. uvicorn rag.app.main:app --port 8000 --reload
```

Required `.env` secrets: `MONGODB_URI`, `GEMINI_API_KEY`, `OPENAI_API_KEY`.
All other config defaults to `app/core/constants/`.

---

## Docker

### Production image (`Dockerfile`)

Single-stage image. Entry point runs uvicorn without `--reload`.
Build context is the repo root; `.dockerignore` excludes `dev_outputs/`, `.env`, `tests/`, `smoke_tests/`.

### Smoke test image (`smoke_tests/docker-compose.smoke.yml`)

Builds the API image locally and runs it on port 8080 with:
- `ENVIRONMENT=LOCAL`
- Rate limiting disabled (no Redis token)

```bash
cd smoke_tests
./run_smoke.sh    # builds, waits for health, runs 2 tests, tears down
```

Evaluation criteria: [`smoke_tests/SPEC.md`](../../smoke_tests/SPEC.md)

---

## CI/CD — GitHub Actions

Four workflows in `.github/workflows/`:

| Workflow | Trigger | Target |
|---|---|---|
| `deploy-stg-gcp.yml` | Push to staging branch | GCP Cloud Run (staging) |
| `deploy-prd-gcp.yml` | Push to main / release tag | GCP Cloud Run (production) |
| `deploy_ec2_stg.yml` | Push to staging branch | EC2 (staging) |
| `ec2_deploy_prd.yml` | Push to main / release tag | EC2 (production) |

Secrets required in GitHub Actions:
- `GEMINI_API_KEY`, `OPENAI_API_KEY`, `MONGODB_URI`, `PINECONE_API_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_ANON_KEY`
- `UPSTASH_REDIS_REST_TOKEN`
- GCP service account key / EC2 SSH credentials (platform-specific)

---

## External services

| Service | URL / Identifier | Notes |
|---|---|---|
| Pinecone | Index: `gemini` | Active vector store |
| MongoDB | `MONGODB_URI` (secret) | `rav_dev` database |
| Supabase | `https://clvgfczatixeajovypju.supabase.co` | Auth + query persistence |
| Upstash Redis | `https://loved-monster-57712.upstash.io` | Rate limiting |
| GCP / Vertex | Project: `ravlegacyproject`, region: `us-central1` | Gemini embeddings |
| Sanity CMS | Via `/api/v1/info/` manifest | Transcript corpus source |

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/sync_manifest.py` | Hash-based smart sync — only re-ingests changed transcripts |
| `scripts/upload_manifest.py` | Brute-force upload with CLI overrides for model and strategy |
| `scripts/supabase_sync/rate_limit_batch_sync.py` | Batch-syncs Redis rate-limit counters to Supabase |

See `scripts/README.md` for full usage.
