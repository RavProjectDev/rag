import concurrent.futures
import logging
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import vertexai
import cohere
from google.api_core.exceptions import (
    GoogleAPIError,
    InvalidArgument,
    DeadlineExceeded,
    PermissionDenied,
    Unauthenticated,
)
from google.auth import default
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from openai import AsyncOpenAI

from rag.app.core.config import get_settings, Environment
from rag.app.exceptions.embedding import *
from rag.app.schemas.data import EmbeddingConfiguration, Embedding

# Constants
DEFAULT_EMBEDDING_DIMENSIONALITY = 784
DEFAULT_TIMEOUT_SECONDS = 10
MOCK_EMBEDDING_DIMENSIONALITY = 784
COHERE_MULTILINGUAL_DIMENSIONALITY = 1024
OPENAI_TEXT_EMBEDDING_3_LARGE_DIMENSIONALITY = 3072


@dataclass
class EmbeddingServiceConfig:
    """Configuration for the embedding service."""

    project_id: str
    region: str
    model_name: str = "gemini-embedding-001"
    default_task: str = "RETRIEVAL_DOCUMENT"
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    output_dimensionality: Optional[int] = None  # None means use Gemini's default (3072)


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
        import os
        
        # In Cloud Run, unset GOOGLE_APPLICATION_CREDENTIALS if it points to non-existent file
        # This allows default() to use the service account credentials
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            if not os.path.exists(creds_path):
                # File doesn't exist, unset so default() uses service account
                os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            elif not os.path.isfile(creds_path):
                # Path exists but isn't a file, unset it
                os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        
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
    dimension_info = f"output_dimensionality={config.output_dimensionality}" if config.output_dimensionality is not None else "output_dimensionality=default (3072)"
    logger.info(f"[EMBEDDING] Using model={config.model_name}, project={config.project_id}, region={config.region}, timeout={config.timeout}s, {dimension_info}")

    text_input = TextEmbeddingInput(text=text_data, task_type=task_type)

    def call_model():
        # Only pass output_dimensionality if specified, otherwise use Gemini's default (3072)
        if config.output_dimensionality is not None:
            return model.get_embeddings(
                [text_input], output_dimensionality=config.output_dimensionality
            )
        else:
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


def _get_cohere_client() -> cohere.Client:
    """Get Cohere client instance."""
    settings = get_settings()
    if not settings.cohere_api_key:
        raise EmbeddingConfigurationException("Cohere API key not configured")
    return cohere.Client(api_key=settings.cohere_api_key)


def _get_openai_embedding_client() -> AsyncOpenAI:
    """Get OpenAI client for embeddings."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise EmbeddingConfigurationException("OpenAI API key not configured")
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def cohere_v3_embedding(text_data: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Generate embeddings using Cohere v3 model.
    
    Args:
        text_data: Text to generate embeddings for
        task_type: Task type - "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
        
    Returns:
        List of float values representing the embedding (1024 dimensions)
        
    Raises:
        EmbeddingException: For embedding-related errors
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[EMBEDDING] Starting Cohere v3 embedding generation, text_length={len(text_data)}, task_type={task_type}")
    
    try:
        client = _get_cohere_client()
        
        # Map task_type to Cohere input_type
        input_type = "search_query" if task_type == "RETRIEVAL_QUERY" else "search_document"
        
        # Cohere v3 embedding
        response = client.embed(
            texts=[text_data],
            model="embed-english-v3.0",
            input_type=input_type,
        )
        
        vector = response.embeddings[0]
        logger.info(f"[EMBEDDING] Successfully generated Cohere v3 embedding, dimension={len(vector)}")
        return vector
        
    except Exception as e:
        logger.error(f"[EMBEDDING ERROR] Cohere v3 embedding error: {str(e)}", exc_info=True)
        raise EmbeddingException(f"Cohere v3 embedding failed: {str(e)}")


async def cohere_multilingual_embedding(text_data: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Generate embeddings using Cohere multilingual model (embed-multilingual-v3.0).
    
    Args:
        text_data: Text to generate embeddings for
        task_type: Task type - "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
        
    Returns:
        List of float values representing the embedding (1024 dimensions)
        
    Raises:
        EmbeddingException: For embedding-related errors
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[EMBEDDING] Starting Cohere multilingual embedding generation, text_length={len(text_data)}, task_type={task_type}")
    
    try:
        client = _get_cohere_client()
        
        # Map task_type to Cohere input_type
        input_type = "search_query" if task_type == "RETRIEVAL_QUERY" else "search_document"
        
        # Cohere multilingual embedding
        response = client.embed(
            texts=[text_data],
            model="embed-multilingual-v3.0",
            input_type=input_type,
        )
        
        vector = response.embeddings[0]
        logger.info(f"[EMBEDDING] Successfully generated Cohere multilingual embedding, dimension={len(vector)}")
        return vector
        
    except Exception as e:
        logger.error(f"[EMBEDDING ERROR] Cohere multilingual embedding error: {str(e)}", exc_info=True)
        raise EmbeddingException(f"Cohere multilingual embedding failed: {str(e)}")


async def openai_text_embedding_3_large(text_data: str) -> List[float]:
    """
    Generate embeddings using OpenAI text-embedding-3-large model.
    
    Args:
        text_data: Text to generate embeddings for
        
    Returns:
        List of float values representing the embedding (3072 dimensions)
        
    Raises:
        EmbeddingException: For embedding-related errors
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[EMBEDDING] Starting OpenAI text-embedding-3-large generation, text_length={len(text_data)}")
    
    try:
        client = _get_openai_embedding_client()
        
        # OpenAI text-embedding-3-large
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input=text_data,
        )
        
        vector = response.data[0].embedding
        logger.info(f"[EMBEDDING] Successfully generated OpenAI embedding, dimension={len(vector)}")
        return vector
        
    except Exception as e:
        logger.error(f"[EMBEDDING ERROR] OpenAI embedding error: {str(e)}", exc_info=True)
        raise EmbeddingException(f"OpenAI embedding failed: {str(e)}")


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
        if configuration == EmbeddingConfiguration.GEMINI_RETRIEVAL_DOCUMENT:
            logger.info(f"[EMBEDDING SERVICE] Using Gemini Retrieval Document configuration")
            vector = await gemini_embedding(text, task_type=task_type)
        elif configuration == EmbeddingConfiguration.COHERE_MULTILINGUAL:
            logger.info(f"[EMBEDDING SERVICE] Using Cohere multilingual configuration")
            vector = await cohere_multilingual_embedding(text, task_type=task_type)
        elif configuration == EmbeddingConfiguration.OPENAI_TEXT_EMBEDDING_3_LARGE:
            logger.info(f"[EMBEDDING SERVICE] Using OpenAI text-embedding-3-large configuration")
            vector = await openai_text_embedding_3_large(text)
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
