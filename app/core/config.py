from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

import rag.app.core.constants as C
from rag.app.schemas.data import (
    ChunkingStrategy,
    DataBaseConfiguration,
    EmbeddingConfiguration,
    Environment,
    LLMModel,
)


class SharedSettings(BaseSettings):
    """
    Settings shared by every app instance (RAG API and Webhook).
    Only true secrets are required from the environment; everything else
    defaults to the values defined in app/core/constants/.
    """

    model_config = ConfigDict(env_file=".env", extra="ignore")

    # --- Secrets (must be in .env) ---
    mongodb_uri: str
    gemini_api_key: str

    # --- Non-secret defaults (env-overridable, sourced from constants) ---
    mongodb_db_name: str = C.MONGODB_DB_NAME
    mongodb_vector_collection: str | None = C.MONGODB_VECTOR_COLLECTION
    collection_index: str = C.COLLECTION_INDEX
    metrics_collection: str = C.METRICS_COLLECTION
    exceptions_collection: str = C.EXCEPTIONS_COLLECTION

    google_cloud_project_id: str = C.GOOGLE_CLOUD_PROJECT_ID
    vertex_region: str = C.VERTEX_REGION
    google_application_credentials: str | None = None

    embedding_configuration: EmbeddingConfiguration = C.EMBEDDING_CONFIGURATION
    chunking_strategy: ChunkingStrategy = C.CHUNKING_STRATEGY
    vector_path: str = C.VECTOR_PATH
    database_configuration: DataBaseConfiguration = C.DATABASE_CONFIGURATION

    # Pinecone — populated from constants when PINECONE backend is active
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = getattr(C, "PINECONE_ENVIRONMENT", None)
    pinecone_index_name: str | None = getattr(C, "PINECONE_INDEX_NAME", None)
    pinecone_namespace: str | None = getattr(C, "PINECONE_NAMESPACE", None)
    pinecone_host: str | None = getattr(C, "PINECONE_HOST", None)

    environment: Environment = C.ENVIRONMENT


class Settings(SharedSettings):
    """
    Full settings for the RAG API (chat, retrieval, evaluation).

    Start with:
        uvicorn rag.app.main:app
    """

    # --- Secrets (must be in .env) ---
    openai_api_key: str
    cohere_api_key: str | None = None
    supabase_service_role_key: str | None = None
    supabase_anon_key: str | None = None
    upstash_redis_rest_token: str | None = None

    # --- Non-secret defaults (env-overridable, sourced from constants) ---
    llm_configuration: LLMModel = C.LLM_CONFIGURATION
    external_api_timeout: int = C.EXTERNAL_API_TIMEOUT

    retrieval_timeout_ms: int = C.RETRIEVAL_TIMEOUT_MS
    max_retry_attempts: int = C.MAX_RETRY_ATTEMPTS
    retry_delay_seconds: float = C.RETRY_DELAY_SECONDS
    retry_backoff_multiplier: float = C.RETRY_BACKOFF_MULTIPLIER

    supabase_url: str | None = C.SUPABASE_URL
    upstash_redis_rest_url: str | None = C.UPSTASH_REDIS_REST_URL
    rate_limit_max_requests: int = C.RATE_LIMIT_MAX_REQUESTS
    rate_limit_window_seconds: int = C.RATE_LIMIT_WINDOW_SECONDS
    user_rate_limit_max_requests_per_month: int = C.USER_RATE_LIMIT_MAX_REQUESTS_PER_MONTH
    dev_outputs: bool = C.DEV_OUTPUTS


@lru_cache()
def get_settings() -> Settings:
    return Settings()
