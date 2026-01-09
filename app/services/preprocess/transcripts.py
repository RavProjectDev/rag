"""
Main preprocessing module for transcripts.

This module handles:
1. Text extraction from SRT/TXT files
2. Routing to appropriate chunking strategy based on configuration
3. Returning processed chunks

Chunking strategies are implemented in separate modules:
- chunking_strategies/fixed_size.py: Strategy A
- chunking_strategies/semantic.py: Strategy B
- chunking_strategies/sliding_window.py: Strategy C
"""
import logging
from typing import List, Tuple
from rag.app.schemas.data import Chunk, TypeOfFormat, ChunkingStrategy, EmbeddingConfiguration
from rag.app.core.config import get_settings
from rag.app.services.preprocess.chunking_strategies import (
    preprocess_raw_transcripts_fixed,
    preprocess_raw_transcripts_semantic,
    preprocess_raw_transcripts_sliding_window,
)
from rag.app.services.preprocess.constants import SEMANTIC_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


async def preprocess_raw_transcripts(
    raw_transcripts: List[Tuple[str, str]],
    data_format: TypeOfFormat = TypeOfFormat.SRT,
    chunking_strategy: ChunkingStrategy | None = None,
    embedding_configuration: EmbeddingConfiguration | None = None,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> List[Chunk]:
    """
    Main preprocessing function that routes to the appropriate chunking strategy.
    
    This function:
    1. Extracts text from SRT/TXT files
    2. Routes to the appropriate chunking strategy based on configuration
    3. Returns processed chunks
    
    Args:
        raw_transcripts: List of (filename, content) tuples
        data_format: Format of the data (SRT or TXT)
        chunking_strategy: Chunking strategy to use. If None, uses config default.
        embedding_configuration: Embedding config. Note: For semantic chunking, this parameter
                                is kept for API compatibility but ignored - semantic chunking
                                always uses BERT (all-MiniLM-L6-v2) internally for similarity
                                calculations. This config is only used later for final storage
                                embeddings.
        similarity_threshold: Similarity threshold for semantic chunking (default: 0.7)
        
    Returns:
        List of Chunk objects
        
    Raises:
        RuntimeError: If data_format is None
        ValueError: If unsupported format or strategy
    """
    if data_format is None:
        raise RuntimeError(f"Unknown data_format")
    
    # Get strategy from config if not provided
    settings = get_settings()
    if chunking_strategy is None:
        chunking_strategy = settings.chunking_strategy
    
    if embedding_configuration is None:
        embedding_configuration = settings.embedding_configuration
    
    logger.info(
        f"Starting preprocessing of {len(raw_transcripts)} transcripts "
        f"with format: {data_format.name}, strategy: {chunking_strategy.value}"
    )
    
    # Route to appropriate chunking strategy
    if chunking_strategy == ChunkingStrategy.FIXED_SIZE:
        logger.info(f"Using Strategy A: Fixed-Size Chunking")
        return preprocess_raw_transcripts_fixed(
            raw_transcripts=raw_transcripts,
            data_format=data_format,
        )
    
    elif chunking_strategy == ChunkingStrategy.SEMANTIC:
        logger.info(f"Using Strategy B: Semantic Chunking (uses BERT internally for similarity)")
        return await preprocess_raw_transcripts_semantic(
            raw_transcripts=raw_transcripts,
            data_format=data_format,
            embedding_configuration=embedding_configuration,
            similarity_threshold=similarity_threshold,
        )
    
    elif chunking_strategy == ChunkingStrategy.SLIDING_WINDOW:
        logger.info(f"Using Strategy C: Sliding Window Chunking")
        return preprocess_raw_transcripts_sliding_window(
            raw_transcripts=raw_transcripts,
            data_format=data_format,
        )
    
    else:
        logger.error(f"Unsupported chunking strategy: {chunking_strategy}")
        raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")


def translate_chunks(chunks: List[Chunk]) -> List[Tuple[str, Chunk]]:
    """
    Translate chunks using a translation mapping file.
    
    This function is kept for backward compatibility.
    """
    import json
    
    logger.info(f"Starting translation of {len(chunks)} chunks")

    try:
        with open(
            "/Users/dothanbardichev/Desktop/RAV/RavProject/embed/data_embedder/data/translations/assigned.json",
            "r",
        ) as f:
            translations = json.load(f)
        logger.debug(f"Loaded {len(translations)} translation mappings")
    except Exception as e:
        logger.error(f"Failed to load translation file: {str(e)}")
        raise

    result: List[Tuple[str, Chunk]] = []
    for i, chunk in enumerate(chunks, 1):
        if i % 100 == 0:
            logger.debug(f"Translated {i}/{len(chunks)} chunks")

        words = chunk.text_to_embed.lower().split()
        translated_words = [translations.get(word, word) for word in words]
        mapped_text = " ".join(translated_words)
        result.append((mapped_text, chunk))

    logger.info(f"Translation completed for {len(result)} chunks")
    return result
