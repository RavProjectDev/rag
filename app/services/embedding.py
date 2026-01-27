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
import cohere
from openai import AsyncOpenAI

from rag.app.core.config import get_settings, Environment
from rag.app.exceptions.embedding import *
from rag.app.schemas.data import EmbeddingConfiguration, Embedding

# Constants
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


@lru_cache(maxsize=1)
def _get_embedding_service_config() -> EmbeddingServiceConfig:
    """Get cached embedding service configuration."""
    settings = get_settings()
    return EmbeddingServiceConfig(
        project_id=settings.google_cloud_project_id, region=settings.vertex_region
    )


def _initialize_vertexai() -> None:
    """Initialize Vertex AI with project configuration."""
    settings = get_settings()

    # Skip external initialization in test environments to allow unit tests to run without GCP
    if getattr(settings, "environment", None) == Environment.TEST:
        return

    try:
        config = _get_embedding_service_config()
        credentials, _ = default()
        vertexai.init(
            project=config.project_id, location=config.region, credentials=credentials
        )
    except Exception as e:
        raise EmbeddingException(f"Failed to initialize Vertex AI: {e}")


def _get_embedding_model() -> TextEmbeddingModel:
    """Get the embedding model instance."""
    config = _get_embedding_service_config()
    return TextEmbeddingModel.from_pretrained(config.model_name)


async def gemini_embedding(text_data: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Generate embeddings using Gemini model.

    Args:
        text_data: Text to generate embeddings for
        task_type: The task type for Gemini embeddings. Options:
                  - "RETRIEVAL_QUERY": For user queries/questions
                  - "RETRIEVAL_DOCUMENT": For documents being indexed

    Returns:
        List of float values representing the embedding

    Raises:
        EmbeddingTimeOutException: If the request times out
        EmbeddingAPIException: For API-related errors
        EmbeddingException: For other unexpected errors
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[EMBEDDING] Starting Gemini embedding generation, text_length={len(text_data)}, task_type={task_type}")
    
    try:
        _initialize_vertexai()
        logger.info(f"[EMBEDDING] Vertex AI initialized")
    except Exception as e:
        logger.error(f"[EMBEDDING ERROR] Failed to initialize Vertex AI: {e}", exc_info=True)
        raise
        
    model = _get_embedding_model()
    config = _get_embedding_service_config()
    logger.info(f"[EMBEDDING] Using model={config.model_name}, project={config.project_id}, region={config.region}, timeout={config.timeout}s")

    text_input = TextEmbeddingInput(text=text_data, task_type=task_type)

    def call_model():
        return model.get_embeddings([text_input])

    # Use a single shared executor for better resource management
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call_model)
        try:
            logger.info(f"[EMBEDDING] Calling Gemini API...")
            vector = future.result(timeout=config.timeout)
            logger.info(f"[EMBEDDING] Successfully generated embedding, dimension={len(vector[0].values)}")
            return vector[0].values
        except concurrent.futures.TimeoutError:
            logger.error(f"[EMBEDDING ERROR] Timeout after {config.timeout} seconds")
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
            logger.error(f"[EMBEDDING ERROR] Gemini API error: {type(e).__name__}: {str(e)}")
            raise EmbeddingAPIException(f"Gemini API error: {str(e)}")
        except Exception as e:
            logger.error(f"[EMBEDDING ERROR] Unexpected error: {str(e)}", exc_info=True)
            raise EmbeddingException(f"Unexpected error during embedding: {str(e)}")


def generate_mock_embedding(text: str) -> List[float]:
    """
    Generate deterministic mock embeddings for testing.

    Uses hash of the text to seed the random generator for reproducibility.
    """
    seed = hash(text) % (2**32)
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(MOCK_EMBEDDING_DIMENSIONALITY)]


async def cohere_embedding(text_data: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Generate embeddings using Cohere's embed-multilingual-v3.0 model.

    Args:
        text_data: Text to generate embeddings for
        task_type: The task type for Cohere embeddings. Options:
                  - "RETRIEVAL_QUERY": For user queries/questions (maps to "search_query")
                  - "RETRIEVAL_DOCUMENT": For documents being indexed (maps to "search_document")

    Returns:
        List of float values representing the embedding

    Raises:
        EmbeddingTimeOutException: If the request times out
        EmbeddingAPIException: For API-related errors
        EmbeddingException: For other unexpected errors
    """
    logger = logging.getLogger(__name__)
    model_name = "embed-multilingual-v3.0"
    logger.info(f"[EMBEDDING] Starting Cohere embedding generation, model={model_name}, text_length={len(text_data)}, task_type={task_type}")
    
    settings = get_settings()
    if not settings.cohere_api_key:
        raise EmbeddingException("Cohere API key not configured")
    
    # Map task_type to Cohere's input_type
    input_type_map = {
        "RETRIEVAL_QUERY": "search_query",
        "RETRIEVAL_DOCUMENT": "search_document"
    }
    input_type = input_type_map.get(task_type, "search_document")
    
    try:
        co = cohere.Client(settings.cohere_api_key)
        logger.info(f"[EMBEDDING] Calling Cohere API with model={model_name}, input_type={input_type}")
        
        response = co.embed(
            texts=[text_data],
            model=model_name,
            input_type=input_type,
            embedding_types=["float"]
        )
        
        vector = response.embeddings.float[0]
        logger.info(f"[EMBEDDING] Successfully generated Cohere embedding, dimension={len(vector)}")
        return vector
        
    except Exception as e:
        logger.error(f"[EMBEDDING ERROR] Cohere API error: {str(e)}", exc_info=True)
        raise EmbeddingAPIException(f"Cohere API error: {str(e)}")


async def openai_embedding(text_data: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Generate embeddings using OpenAI's text-embedding-3-large model.

    Args:
        text_data: Text to generate embeddings for
        task_type: The task type (currently not used by OpenAI API, kept for consistency)

    Returns:
        List of float values representing the embedding

    Raises:
        EmbeddingTimeOutException: If the request times out
        EmbeddingAPIException: For API-related errors
        EmbeddingException: For other unexpected errors
    """
    logger = logging.getLogger(__name__)
    model_name = "text-embedding-3-large"
    logger.info(f"[EMBEDDING] Starting OpenAI embedding generation, model={model_name}, text_length={len(text_data)}, task_type={task_type}")
    
    settings = get_settings()
    if not settings.openai_api_key:
        raise EmbeddingException("OpenAI API key not configured")
    
    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info(f"[EMBEDDING] Calling OpenAI API with model={model_name}")
        
        response = await client.embeddings.create(
            input=text_data,
            model=model_name
        )
        
        vector = response.data[0].embedding
        logger.info(f"[EMBEDDING] Successfully generated OpenAI embedding, dimension={len(vector)}")
        return vector
        
    except Exception as e:
        logger.error(f"[EMBEDDING ERROR] OpenAI API error: {str(e)}", exc_info=True)
        raise EmbeddingAPIException(f"OpenAI API error: {str(e)}")


async def generate_embedding(
    text: str,
    configuration: EmbeddingConfiguration,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> Embedding:
    """
    Generate embeddings based on the specified configuration.

    Args:
        text: Text to generate embeddings for
        configuration: Embedding configuration to use
        task_type: The task type for Gemini embeddings. Options:
                  - "RETRIEVAL_QUERY": For user queries/questions
                  - "RETRIEVAL_DOCUMENT": For documents being indexed (default)

    Returns:
        Embedding object containing text and vector

    Raises:
        EmbeddingConfigurationException: For invalid configurations
        EmbeddingException: For other embedding-related errors
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[EMBEDDING SERVICE] Starting embedding generation with configuration={configuration.name if hasattr(configuration, 'name') else configuration}, task_type={task_type}")
    
    if not isinstance(configuration, EmbeddingConfiguration):
        logger.error(f"[EMBEDDING SERVICE ERROR] Invalid configuration type: {type(configuration)}")
        raise EmbeddingConfigurationException(
            "Invalid configuration for embedding service"
        )
    if not text:
        logger.error(f"[EMBEDDING SERVICE ERROR] Empty text provided")
        raise EmbeddingException("Text cannot be empty for embedding generation")
    if not configuration:
        logger.error(f"[EMBEDDING SERVICE ERROR] No configuration provided")
        raise EmbeddingConfigurationException(
            "Configuration cannot be empty for embedding generation"
        )
    try:
        if configuration == EmbeddingConfiguration.GEMINI:
            logger.info(f"[EMBEDDING SERVICE] Using Gemini configuration")
            vector = await gemini_embedding(text, task_type=task_type)
        elif configuration == EmbeddingConfiguration.COHERE:
            logger.info(f"[EMBEDDING SERVICE] Using Cohere configuration")
            vector = await cohere_embedding(text, task_type=task_type)
        elif configuration == EmbeddingConfiguration.OPENAI:
            logger.info(f"[EMBEDDING SERVICE] Using OpenAI configuration")
            vector = await openai_embedding(text, task_type=task_type)
        elif configuration == EmbeddingConfiguration.MOCK:
            logger.info(f"[EMBEDDING SERVICE] Using Mock configuration")
            vector = generate_mock_embedding(text)
        else:
            logger.error(f"[EMBEDDING SERVICE ERROR] Unsupported configuration: {configuration.name}")
            raise EmbeddingConfigurationException(
                f"Unsupported embedding configuration: {configuration.name}"
            )
        logger.info(f"[EMBEDDING SERVICE] Successfully created Embedding object with vector dimension={len(vector)}")
        return Embedding(text=text, vector=vector)
    except Exception as e:
        logging.error(f"[EMBEDDING SERVICE ERROR] Error in generate_embedding: {e}", exc_info=True)
        raise
