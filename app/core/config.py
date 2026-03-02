from enum import Enum
from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

from rag.app.schemas.data import (
    EmbeddingConfiguration,
    LLMModel,
    DataBaseConfiguration,
    ChunkingStrategy,
)

COLLECTIONS = ["gemini_embeddings_v2", "chunk_embeddings_gemini_embedding_001"]


class Environment(Enum):
    PRD = "PRD"
    STG = "STG"
    TEST = "TEST"


class AuthMode(Enum):
    DEV = "dev"
    PRD = "prd"


class SharedSettings(BaseSettings):
    """
    Settings shared by every app instance (RAG API and Webhook).
    Contains DB connection, embedding model, and infrastructure config.
    """

    model_config = ConfigDict(env_file=".env", extra="ignore")

    # MongoDB
    mongodb_uri: str
    mongodb_db_name: str
    mongodb_vector_collection: str | None = None
    collection_index: str = "vector_index"
    metrics_collection: str = "metrics"
    exceptions_collection: str = "exceptions"

    # Database backend
    database_configuration: DataBaseConfiguration = DataBaseConfiguration.MONGO

    # Embedding
    gemini_api_key: str
    google_cloud_project_id: str
    vertex_region: str
    google_application_credentials: str | None = None
    embedding_configuration: EmbeddingConfiguration = EmbeddingConfiguration.GEMINI
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    vector_path: str = "vector"

    # Pinecone (optional — only required when database_configuration=PINECONE)
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str | None = None
    pinecone_namespace: str | None = None
    pinecone_host: str | None = None

    # Runtime
    environment: Environment = Environment.STG


class Settings(SharedSettings):
    """
    Full settings for the RAG API (chat, retrieval, evaluation).

    Start with:
        uvicorn rag.app.main:app
    """

    # LLM
    openai_api_key: str
    cohere_api_key: str | None = None
    llm_configuration: LLMModel = LLMModel.GPT_4
    external_api_timeout: int = 60

    # Document retrieval retry
    retrieval_timeout_ms: int = 2000
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0

    # Auth (Supabase)
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    supabase_anon_key: str | None = None
    auth_mode: AuthMode = AuthMode.DEV

    # Rate limiting (Redis / Upstash)
    upstash_redis_rest_url: str | None = None
    upstash_redis_rest_token: str | None = None
    rate_limit_max_requests: int = 10000
    rate_limit_window_seconds: int = 3600
    user_rate_limit_max_requests_per_month: int = 10000

    # Debug
    dev_outputs: bool = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
