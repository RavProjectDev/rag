# config.py
from enum import Enum

from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from rag.app.schemas.data import EmbeddingConfiguration, LLMModel, ChunkingStrategy, DataBaseConfiguration
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
    
    # Database configuration
    database_type: str = "MONGO"  # "MONGO" or "PINECONE"
    
    # MongoDB settings
    mongodb_uri: str | None = None
    mongodb_db_name: str | None = None
    mongodb_vector_collection: str | None = None
    collection_index: str = "vector_index"
    metrics_collection: str = "metrics"
    exceptions_collection: str = "exceptions"
    
    # Pinecone settings
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str | None = None
    pinecone_dimension: int = 784  # Default for Gemini
    pinecone_metric: str = "cosine"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    
    # API keys
    openai_api_key: str
    cohere_api_key: str | None = None
    gemini_api_key: str
    google_cloud_project_id: str
    google_application_credentials: str | None = None
    
    # Model configurations
    embedding_configuration: EmbeddingConfiguration = EmbeddingConfiguration.GEMINI_RETRIEVAL_DOCUMENT
    llm_configuration: LLMModel = LLMModel.GPT_4
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    
    # General settings
    vector_path: str = "vector"
    vertex_region: str
    external_api_timeout: int = 60
    environment: Environment = Environment.STG
    
    # Document retrieval retry configuration
    retrieval_timeout_ms: int = 2000  # Increased timeout for document retrieval
    max_retry_attempts: int = 3  # Maximum number of retry attempts
    retry_delay_seconds: float = 1.0  # Initial delay between retries
    retry_backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    
    # Authentication
    supabase_url: str | None = None  # Supabase project URL for JWKS
    auth_mode: AuthMode = AuthMode.DEV  # Authentication mode: dev (no auth) or prd (requires auth)
    dev_outputs: bool = False  # When enabled, saves LLM prompts to /dev_outputs folder


@lru_cache()
def get_settings():
    return Settings()