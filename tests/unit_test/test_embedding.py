import time
from unittest.mock import patch

import pytest
from google.api_core.exceptions import InvalidArgument

from rag.app.exceptions.embedding import (
    EmbeddingConfigurationException,
    EmbeddingTimeOutException,
    EmbeddingException,
    EmbeddingAPIException,
)
from rag.app.schemas.data import EmbeddingConfiguration
from rag.app.services.embedding import generate_embedding, EmbeddingServiceConfig


@pytest.mark.asyncio
async def test_generate_mock_embedding():
    text = "Hello world"
    config = EmbeddingConfiguration.MOCK
    embedding = await generate_embedding(text, config)
    assert embedding.text == text
    assert isinstance(embedding.vector, list)
    assert len(embedding.vector) == 784
    assert all(isinstance(x, float) for x in embedding.vector)


@pytest.mark.asyncio
async def test_generate_embedding_with_no_config():
    text = "Hello world"
    with pytest.raises(EmbeddingConfigurationException):
        await generate_embedding(text, configuration=None)


@pytest.mark.asyncio
async def test_generate_embedding_empty_text():
    with pytest.raises(EmbeddingException):
        await generate_embedding("", EmbeddingConfiguration.MOCK)


@pytest.mark.asyncio
@patch("rag.app.services.embedding.gemini_embedding")
async def test_generate_gemini_embedding(mock_gemini_embedding):
    text = "Hello world"
    config = EmbeddingConfiguration.GEMINI
    fake_vector = [0.1] * 784
    mock_gemini_embedding.return_value = fake_vector
    embedding = await generate_embedding(text, config)
    assert embedding.text == text
    assert embedding.vector == fake_vector
    assert len(embedding.vector) == 784


@pytest.mark.asyncio
@patch("rag.app.services.embedding._get_embedding_service_config")
@patch("rag.app.services.embedding._get_embedding_model")
async def test_gemini_embedding_timeout(mock_get_model, mock_get_config):
    config = EmbeddingServiceConfig(project_id="test", region="us-central1", timeout=1)
    mock_get_config.return_value = config

    class SlowModel:
        def get_embeddings(self, *args, **kwargs):
            time.sleep(2)  # Exceeds 1-second timeout
            return [{"values": [0.0] * 784}]

    mock_get_model.return_value = SlowModel()

    with pytest.raises(EmbeddingTimeOutException):
        await generate_embedding("test text", EmbeddingConfiguration.GEMINI)


@pytest.mark.asyncio
async def test_invalid_embedding_configuration():
    text = "Hello world"
    with pytest.raises(EmbeddingConfigurationException):
        await generate_embedding(text, configuration="INVALID")
