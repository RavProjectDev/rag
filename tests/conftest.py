# rag/tests/conftest.py

import pytest


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    from rag.app.core import config
    from rag.app.core.config import (
        Settings,
        Environment,
        AuthMode,
        EmbeddingConfiguration,
        LLMModel,
    )

    test_settings = Settings(
        openai_api_key="sk-test",
        mongodb_uri="mongodb://localhost:27017",
        mongodb_db_name="testdb",
        mongodb_vector_collection="test_vectors",
        collection_index="test_index",
        gemini_api_key="gemini_test_key",
        google_cloud_project_id="my-project",
        vector_path="vector",
        vertex_region="us-central1",
        external_api_timeout=30,
        metrics_collection="metrics",
        exceptions_collection="exceptions",
        google_application_credentials="/tmp/google-creds.json",
        environment=Environment.TEST,
        embedding_configuration=EmbeddingConfiguration.MOCK,
        llm_configuration=LLMModel.MOCK,
        auth_mode=AuthMode.DEV,  # Tests run in dev mode (no auth required)
        _env_file=None,  # Don't read from .env file during tests
    )

    # Clear the lru_cache for get_settings to ensure fresh settings
    config.get_settings.cache_clear()
    
    # Also clear cache for get_openai_client if it exists
    try:
        from rag.app.services.llm import get_openai_client
        get_openai_client.cache_clear()
    except (ImportError, AttributeError):
        pass
    
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)
    monkeypatch.setattr("rag.app.db.mongodb_connection.get_settings", lambda: test_settings)
    monkeypatch.setattr("rag.app.services.embedding.get_settings", lambda: test_settings)
    monkeypatch.setattr("rag.app.services.llm.get_settings", lambda: test_settings)
