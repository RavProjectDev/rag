# rag/tests/conftest.py

import pytest


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    from rag.app.core import config
    from rag.app.core.config import (
        Settings,
        Environment,
        EmbeddingConfiguration,
        LLMModel,
    )

    test_settings = Settings(
        openai_api_key="sk-test",
        sbert_api_url="http://localhost",
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
        environment=Environment.TEST,
        embedding_configuration=EmbeddingConfiguration.MOCK,
        llm_configuration=LLMModel.MOCK,
    )

    monkeypatch.setattr(config, "get_settings", lambda: test_settings)
