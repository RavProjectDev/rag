# config.py
from enum import Enum

from pydantic_settings import BaseSettings
from rag.app.schemas.data import EmbeddingConfiguration, LLMModel
from functools import lru_cache

COLLECTIONS = ["gemini_embeddings_v2", "chunk_embeddings_gemini_embedding_001"]


class Environment(Enum):
    PRD = "PRD"
    TEST = "TEST"


class Settings(BaseSettings):
    openai_api_key: str
    sbert_api_url: str
    mongodb_uri: str
    mongodb_db_name: str
    mongodb_vector_collection: str
    collection_index: str = "vector_index"
    gemini_api_key: str
    google_cloud_project_id: str
    embedding_configuration: EmbeddingConfiguration = EmbeddingConfiguration.GEMINI
    llm_configuration: LLMModel = LLMModel.GPT_4
    vector_path: str = "vector"
    vertex_region: str
    external_api_timeout: int = 60
    metrics_collection: str = "metrics"
    environment: Environment = Environment.PRD
    exceptions_collection: str = "exceptions"
    model_config = {"env_file": ".env"}


@lru_cache()
def get_settings():
    return Settings()
