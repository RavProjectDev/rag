import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from rag.app.exceptions.embedding import (
    EmbeddingException,
    EmbeddingAPIException,
)
from rag.app.schemas.data import EmbeddingConfiguration
from rag.app.services.embedding import generate_embedding


@pytest.mark.asyncio
@patch("rag.app.services.embedding.cohere_embedding")
async def test_generate_cohere_embedding(mock_cohere_embedding):
    """Test Cohere embedding generation"""
    text = "Hello world"
    config = EmbeddingConfiguration.COHERE
    fake_vector = [0.1] * 1024  # Cohere embed-multilingual-v3.0 dimension
    mock_cohere_embedding.return_value = fake_vector
    
    embedding = await generate_embedding(text, config, task_type="RETRIEVAL_QUERY")
    
    assert embedding.text == text
    assert embedding.vector == fake_vector
    assert len(embedding.vector) == 1024
    mock_cohere_embedding.assert_called_once_with(text, task_type="RETRIEVAL_QUERY")


@pytest.mark.asyncio
@patch("rag.app.services.embedding.openai_embedding")
async def test_generate_openai_embedding(mock_openai_embedding):
    """Test OpenAI embedding generation"""
    text = "Hello world"
    config = EmbeddingConfiguration.OPENAI
    fake_vector = [0.1] * 3072  # OpenAI text-embedding-3-large dimension
    mock_openai_embedding.return_value = fake_vector
    
    embedding = await generate_embedding(text, config, task_type="RETRIEVAL_DOCUMENT")
    
    assert embedding.text == text
    assert embedding.vector == fake_vector
    assert len(embedding.vector) == 3072
    mock_openai_embedding.assert_called_once_with(text, task_type="RETRIEVAL_DOCUMENT")


@pytest.mark.asyncio
@patch("rag.app.services.embedding.get_settings")
async def test_cohere_embedding_no_api_key(mock_get_settings):
    """Test Cohere embedding fails without API key"""
    from rag.app.services.embedding import cohere_embedding
    
    mock_settings = MagicMock()
    mock_settings.cohere_api_key = None
    mock_get_settings.return_value = mock_settings
    
    with pytest.raises(EmbeddingException, match="Cohere API key not configured"):
        await cohere_embedding("test text")


@pytest.mark.asyncio
@patch("rag.app.services.embedding.get_settings")
async def test_openai_embedding_no_api_key(mock_get_settings):
    """Test OpenAI embedding fails without API key"""
    from rag.app.services.embedding import openai_embedding
    
    mock_settings = MagicMock()
    mock_settings.openai_api_key = None
    mock_get_settings.return_value = mock_settings
    
    with pytest.raises(EmbeddingException, match="OpenAI API key not configured"):
        await openai_embedding("test text")


@pytest.mark.asyncio
async def test_all_embedding_configurations_exist():
    """Test that all embedding configurations are properly defined"""
    configs = [
        EmbeddingConfiguration.GEMINI,
        EmbeddingConfiguration.COHERE,
        EmbeddingConfiguration.OPENAI,
        EmbeddingConfiguration.MOCK,
    ]
    
    assert EmbeddingConfiguration.GEMINI.value == "gemini-embedding-001"
    assert EmbeddingConfiguration.COHERE.value == "cohere"
    assert EmbeddingConfiguration.OPENAI.value == "openai"
    assert EmbeddingConfiguration.MOCK.value == "mock"

