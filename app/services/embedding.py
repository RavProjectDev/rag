import concurrent.futures
import logging
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import vertexai
from google.api_core.exceptions import (
    GoogleAPIError,
    InvalidArgument,
    DeadlineExceeded,
    PermissionDenied,
    Unauthenticated,
)
from google.auth import default
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from rag.app.core.config import get_settings
from rag.app.exceptions.embedding import *
from rag.app.schemas.data import EmbeddingConfiguration, Embedding

# Constants
DEFAULT_EMBEDDING_DIMENSIONALITY = 784
DEFAULT_TIMEOUT_SECONDS = 10
MOCK_EMBEDDING_DIMENSIONALITY = 784


@dataclass
class EmbeddingServiceConfig:
    """Configuration for the embedding service."""

    project_id: str
    region: str
    model_name: str = "gemini-embedding-001"
    default_task: str = "RETRIEVAL_DOCUMENT"
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONALITY


@lru_cache(maxsize=1)
def _get_embedding_service_config() -> EmbeddingServiceConfig:
    """Get cached embedding service configuration."""
    settings = get_settings()
    return EmbeddingServiceConfig(
        project_id=settings.google_cloud_project_id, region=settings.vertex_region
    )


def _initialize_vertexai() -> None:
    """Initialize Vertex AI with project configuration."""
    try:
        config = _get_embedding_service_config()
        credentials, _ = default()
        vertexai.init(
            project=config.project_id, location=config.region, credentials=credentials
        )
    except Exception as e:
        raise EmbeddingException


def _get_embedding_model() -> TextEmbeddingModel:
    """Get the embedding model instance."""
    config = _get_embedding_service_config()
    return TextEmbeddingModel.from_pretrained(config.model_name)


async def gemini_embedding(text_data: str) -> List[float]:
    """
    Generate embeddings using Gemini model.

    Args:
        text_data: Text to generate embeddings for

    Returns:
        List of float values representing the embedding

    Raises:
        EmbeddingTimeOutException: If the request times out
        EmbeddingAPIException: For API-related errors
        EmbeddingException: For other unexpected errors
    """
    _initialize_vertexai()
    model = _get_embedding_model()
    config = _get_embedding_service_config()

    text_input = TextEmbeddingInput(text=text_data, task_type=config.default_task)

    def call_model():
        return model.get_embeddings(
            [text_input], output_dimensionality=config.output_dimensionality
        )

    # Use a single shared executor for better resource management
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call_model)
        try:
            vector = future.result(timeout=config.timeout)
            return vector[0].values
        except concurrent.futures.TimeoutError:
            raise EmbeddingTimeOutException(
                f"Gemini embedding timed out after {config.timeout} seconds."
            )
        except (
            InvalidArgument,
            PermissionDenied,
            Unauthenticated,
            DeadlineExceeded,
            GoogleAPIError,
        ) as e:
            raise EmbeddingAPIException(f"Gemini API error: {str(e)}")
        except Exception as e:
            raise EmbeddingException(f"Unexpected error during embedding: {str(e)}")


def generate_mock_embedding(text: str) -> List[float]:
    """
    Generate deterministic mock embeddings for testing.

    Uses hash of the text to seed the random generator for reproducibility.
    """
    seed = hash(text) % (2**32)
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(MOCK_EMBEDDING_DIMENSIONALITY)]


async def generate_embedding(
    text: str,
    configuration: EmbeddingConfiguration,
) -> Embedding:
    """
    Generate embeddings based on the specified configuration.

    Args:
        text: Text to generate embeddings for
        configuration: Embedding configuration to use

    Returns:
        Embedding object containing text and vector

    Raises:
        EmbeddingConfigurationException: For invalid configurations
        EmbeddingException: For other embedding-related errors
    """
    if not isinstance(configuration, EmbeddingConfiguration):
        raise EmbeddingConfigurationException(
            "Invalid configuration for embedding service"
        )
    if not text:
        raise EmbeddingException("Text cannot be empty for embedding generation")
    if not configuration:
        raise EmbeddingConfigurationException(
            "Configuration cannot be empty for embedding generation"
        )
    try:
        if configuration == EmbeddingConfiguration.GEMINI:
            vector = await gemini_embedding(text)
        elif configuration == EmbeddingConfiguration.MOCK:
            vector = generate_mock_embedding(text)
        else:
            raise EmbeddingConfigurationException(
                f"Unsupported embedding configuration: {configuration.name}"
            )
        return Embedding(text=text, vector=vector)
    except Exception as e:
        logging.error(f"Error in generate_embedding: {e}")
        raise
