"""
Configuration and Constants for Sanity Check

Defines embedding models, chunking strategies, and mapping functions.
"""

from rag.app.schemas.data import EmbeddingConfiguration, ChunkingStrategy

# Manifest API endpoint
MANIFEST_URL = "https://theravlegacy.org/api/manifest"

# Retry configuration for sanity check
SANITY_CHECK_RETRY_DELAY = 2  # Initial delay in seconds
SANITY_CHECK_MAX_DELAY = 60   # Maximum delay cap in seconds
SANITY_CHECK_BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier

# Available embedding models
EMBEDDING_MODELS = {
    "gemini": EmbeddingConfiguration.GEMINI_RETRIEVAL_DOCUMENT,
    "openai": EmbeddingConfiguration.OPENAI_TEXT_EMBEDDING_3_LARGE,
    "cohere": EmbeddingConfiguration.COHERE_MULTILINGUAL,
}

# Available chunking strategies
CHUNKING_STRATEGIES = {
    "fixed-size": ChunkingStrategy.FIXED_SIZE,
    "semantic": ChunkingStrategy.SEMANTIC,
    "sliding-window": ChunkingStrategy.SLIDING_WINDOW,
}

# Embedding dimensions for each model
EMBEDDING_DIMENSIONS = {
    EmbeddingConfiguration.GEMINI_RETRIEVAL_DOCUMENT: 784,
    EmbeddingConfiguration.OPENAI_TEXT_EMBEDDING_3_LARGE: 3072,
    EmbeddingConfiguration.COHERE_MULTILINGUAL: 1024,
}


def get_index_name(embedding_config: EmbeddingConfiguration) -> str:
    """
    Map embedding configuration to Pinecone index name.
    
    Args:
        embedding_config: Embedding configuration enum
        
    Returns:
        Index name: gemini, openai, or cohere
    """
    mapping = {
        EmbeddingConfiguration.GEMINI_RETRIEVAL_DOCUMENT: "gemini",
        EmbeddingConfiguration.OPENAI_TEXT_EMBEDDING_3_LARGE: "openai",
        EmbeddingConfiguration.COHERE_MULTILINGUAL: "cohere",
    }
    return mapping.get(embedding_config, "gemini")


def get_namespace_name(chunking_strategy: ChunkingStrategy) -> str:
    """
    Map chunking strategy to Pinecone namespace name.
    
    Args:
        chunking_strategy: Chunking strategy enum
        
    Returns:
        Namespace name: fixed-size, semantic, or sliding-window
    """
    mapping = {
        ChunkingStrategy.FIXED_SIZE: "fixed-size",
        ChunkingStrategy.SEMANTIC: "semantic",
        ChunkingStrategy.SLIDING_WINDOW: "sliding-window",
    }
    return mapping.get(chunking_strategy, "fixed-size")


def get_embedding_dimension(embedding_config: EmbeddingConfiguration) -> int:
    """
    Get the dimension for each embedding configuration.
    
    Args:
        embedding_config: Embedding configuration enum
        
    Returns:
        Dimension size for the embedding model
    """
    return EMBEDDING_DIMENSIONS.get(embedding_config, 784)

