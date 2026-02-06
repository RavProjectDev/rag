# Add your shared dependencies here
from fastapi import Request, Depends
from rag.app.core import config

#SETTINGS
from rag.app.schemas.data import LLMModel, EmbeddingConfiguration, ChunkingStrategy
from rag.app.db.connections import EmbeddingConnection, MetricsConnection
from rag.app.db.redis_connection import RedisConnection


def get_embedding_conn(request: Request) -> EmbeddingConnection:
    if hasattr(request.app.state, "embedding_conn"):
        return request.app.state.embedding_conn
    return request.app.state.mongo_conn


def get_metrics_conn(request: Request) -> MetricsConnection:
    return request.app.state.metrics_connection


def get_embedding_configuration() -> EmbeddingConfiguration:
    return config.get_settings().embedding_configuration


def get_llm_configuration() -> LLMModel:
    return config.get_settings().llm_configuration


def get_chunking_strategy() -> ChunkingStrategy:
    return config.get_settings().chunking_strategy


def get_settings_dependency():
    return config.get_settings()


def get_random_embedding_collection(request: Request) -> EmbeddingConnection:
    if hasattr(request.app.state, "embedding_conn"):
        return request.app.state.embedding_conn
    return request.app.state.mongo_conn


def get_redis_conn(request: Request) -> RedisConnection | None:
    """Get Redis connection from app state if available."""
    if hasattr(request.app.state, "redis_conn"):
        return request.app.state.redis_conn
    return None
