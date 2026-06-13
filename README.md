# RAV RAG API

A production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI. Ingests Rav Soloveitchik lecture transcripts (`.srt` / `.txt`), embeds them into a vector database, and serves a chat API that retrieves the most relevant passages before generating a cited, structured response with an LLM.

Full architectural documentation lives in [`specs/current/`](specs/current/).

---

## Quick start

```bash
git clone <repository-url>
cd rag
pip install -r requirements.txt
cp .env.example .env   # fill in required values
PYTHONPATH=.. uvicorn rag.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: `http://localhost:8000/docs`

---

## Docker

```bash
docker build -t rav-rag .
docker run -p 8080:8080 --env-file .env rav-rag
```

The container exposes port `8080` (overridable via `PORT`). The Dockerfile sets `PYTHONPATH=/` so `app` is importable directly from `/rag/app`.

---

## Key environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `GEMINI_API_KEY` | Google Gemini API key (embeddings) |
| `GOOGLE_CLOUD_PROJECT_ID` | GCP project ID |
| `VERTEX_REGION` | Vertex AI region |
| `MONGODB_URI` | MongoDB connection string |
| `MONGODB_DB_NAME` | MongoDB database name |
| `MONGODB_VECTOR_COLLECTION` | Collection holding vector embeddings |
| `AUTH_MODE` | `dev` (no JWT) or `prd` (Supabase JWT required) |
| `LLM_CONFIGURATION` | `GPT_4`, `GEMINI_FLASH`, or `MOCK` |
| `DATABASE_CONFIGURATION` | `MONGODB` (default) or `PINECONE` |

See `.env.example` and [`specs/current/configuration.md`](specs/current/configuration.md) for the full list.

---

## Data ingestion

Populate the vector database before serving requests:

```bash
python scripts/sync_manifest.py     # pull transcript manifest from Sanity CMS
python scripts/upload_manifest.py   # chunk, embed, and upsert into vector DB
```

See [`scripts/README.md`](scripts/README.md) for detailed usage.

---

## Documentation

| Location | Contents |
|---|---|
| [`specs/current/architecture.md`](specs/current/architecture.md) | Stack, service map, request lifecycle |
| [`specs/current/api-surface.md`](specs/current/api-surface.md) | All routes, request/response contracts, auth |
| [`specs/current/data-models.md`](specs/current/data-models.md) | Enums, Pydantic schemas, domain models |
| [`specs/current/configuration.md`](specs/current/configuration.md) | Settings hierarchy, constants, env vars |
| [`specs/current/infrastructure.md`](specs/current/infrastructure.md) | Docker, GCP, EC2, CI/CD |
| [`smoke_tests/SPEC.md`](smoke_tests/SPEC.md) | Acceptance criteria for the chat endpoint |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Spec-driven development workflow |
