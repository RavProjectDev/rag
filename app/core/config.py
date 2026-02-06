# config.py
from enum import Enum

from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from rag.app.schemas.data import (
    EmbeddingConfiguration,
    LLMModel,
    DataBaseConfiguration,
    ChunkingStrategy,
)
from functools import lru_cache

COLLECTIONS = ["gemini_embeddings_v2", "chunk_embeddings_gemini_embedding_001"]


class Environment(Enum):
    PRD = "PRD"
    STG = "STG"
    TEST = "TEST"


class AuthMode(Enum):
    DEV = "dev"
    PRD = "prd"


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        extra="ignore",  # Ignore extra environment variables not defined in the model
    )
    
    openai_api_key: str
    cohere_api_key: str | None = None
    mongodb_uri: str
    mongodb_db_name: str
    mongodb_vector_collection: str | None = None
    collection_index: str = "vector_index"
    database_configuration: DataBaseConfiguration = DataBaseConfiguration.MONGO
    gemini_api_key: str
    google_cloud_project_id: str
    embedding_configuration: EmbeddingConfiguration = EmbeddingConfiguration.GEMINI
    llm_configuration: LLMModel = LLMModel.GPT_4
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    vector_path: str = "vector"
    vertex_region: str
    external_api_timeout: int = 60
    metrics_collection: str = "metrics"
    environment: Environment = Environment.STG
    exceptions_collection: str = "exceptions"
    # Document retrieval retry configuration
    retrieval_timeout_ms: int = 2000  # Increased timeout for document retrieval
    max_retry_attempts: int = 3  # Maximum number of retry attempts
    retry_delay_seconds: float = 1.0  # Initial delay between retries
    retry_backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    google_application_credentials: str | None = None
    supabase_url: str | None = None  # Supabase project URL for JWKS
    auth_mode: AuthMode = AuthMode.DEV  # Authentication mode: dev (no auth) or prd (requires auth)
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str | None = None
    pinecone_namespace: str | None = None
    pinecone_host: str | None = None
    dev_outputs: bool = False  # Enable writing debug outputs to dev_outputs/ directory
    upstash_redis_rest_url: str | None = None  # Upstash Redis REST URL
    upstash_redis_rest_token: str | None = None  # Upstash Redis REST Token
    # Rate limiting (Redis): max requests per window, window TTL in seconds
    rate_limit_max_requests: int = 10000  # Max requests allowed per window
    rate_limit_window_seconds: int = 3600  # Window TTL in seconds (e.g. 3600 = 1 hour, 60 = 1 minute)
    # User-based rate limiting (Redis): max requests per user per month (resets on 1st of each month ET)
    user_rate_limit_max_requests_per_month: int = 10000  # Max requests per user per month


@lru_cache()
def get_settings():
    return Settings()