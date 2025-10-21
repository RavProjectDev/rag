# Add your shared dependencies here
from fastapi import Request, Depends
from rag.app.core import config


from rag.app.schemas.data import LLMModel, EmbeddingConfiguration
from rag.app.db.connections import EmbeddingConnection, MetricsConnection


def get_embedding_conn(request: Request) -> EmbeddingConnection:
    return request.app.state.mongo_conn


def get_metrics_conn(request: Request) -> MetricsConnection:
    return request.app.state.metrics_connection


def get_embedding_configuration() -> EmbeddingConfiguration:
    return config.get_settings().embedding_configuration


def get_llm_configuration() -> LLMModel:
    return config.get_settings().llm_configuration


def get_settings_dependency():
    return config.get_settings()


def get_random_embedding_collection(request: Request) -> EmbeddingConnection:
    return request.app.state.mongo_conn
