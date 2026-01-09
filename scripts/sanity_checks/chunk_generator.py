"""
Chunk generation utilities

Handles preprocessing transcripts into chunks and generating chunk IDs.
"""

import asyncio
import hashlib
import logging
from typing import List

from rag.app.models.data import SanityData
from rag.app.schemas.data import (
    EmbeddingConfiguration,
    ChunkingStrategy,
    Chunk,
)
from rag.app.services.preprocess.transcripts import preprocess_raw_transcripts

from .config import (
    SANITY_CHECK_RETRY_DELAY,
    SANITY_CHECK_MAX_DELAY,
    SANITY_CHECK_BACKOFF_MULTIPLIER,
)
from .fetch import fetch_transcript_with_infinite_retry

logger = logging.getLogger(__name__)


def generate_chunk_id(
    sanity_id: str,
    full_text_id: str,
    text_to_embed: str
) -> str:
    """
    Generate chunk ID using SAME logic as Pinecone upload.
    
    Must match: _generate_unique_id in PineconeEmbeddingStore
    
    Args:
        sanity_id: Document ID
        full_text_id: Chunk identifier within document
        text_to_embed: The actual text content
        
    Returns:
        Unique chunk ID: {sanity_id}_{full_text_id}_{md5_hash}
    """
    text_hash = hashlib.md5(text_to_embed.encode()).hexdigest()[:8]
    return f"{sanity_id}_{full_text_id}_{text_hash}"


async def generate_chunks_with_infinite_retry(
    doc: SanityData,
    chunking_strategy: ChunkingStrategy,
    embedding_config: EmbeddingConfiguration,
    max_attempts: int | None = None
) -> List[Chunk]:
    """
    Generate chunks with infinite retries.
    
    This MUST succeed to compare expected vs actual state.
    Retries both transcript fetching and preprocessing.
    
    Args:
        doc: Document to process
        chunking_strategy: Strategy to use for chunking
        embedding_config: Embedding configuration (affects preprocessing)
        max_attempts: Optional maximum number of attempts (None = infinite)
        
    Returns:
        List of chunks
        
    Raises:
        Exception: Only if max_attempts is reached
    """
    attempt = 0
    delay = SANITY_CHECK_RETRY_DELAY
    
    while True:
        attempt += 1
        
        # Check max attempts if specified
        if max_attempts is not None and attempt > max_attempts:
            error_msg = f"Failed to generate chunks after {max_attempts} attempts"
            logger.error(f"[GENERATE FAILED] {doc.title}: {error_msg}")
            raise Exception(error_msg)
        
        try:
            # Fetch transcript (with its own infinite retry)
            contents = await fetch_transcript_with_infinite_retry(
                str(doc.transcriptURL),
                doc.title,
                max_attempts=max_attempts  # Pass through max_attempts
            )
            
            # Preprocess (this should never fail with valid input)
            raw_transcripts = [(doc.title, contents)]
            chunks = await preprocess_raw_transcripts(
                raw_transcripts=raw_transcripts,
                chunking_strategy=chunking_strategy,
                embedding_configuration=embedding_config,
            )
            
            if not chunks:
                raise Exception(f"Preprocessing returned 0 chunks for {doc.title}")
            
            if attempt > 1:
                logger.info(
                    f"[GENERATE SUCCESS] {doc.title}: {len(chunks)} chunks "
                    f"(succeeded on attempt {attempt})"
                )
            else:
                logger.debug(
                    f"[GENERATE SUCCESS] {doc.title}: {len(chunks)} chunks"
                )
            
            return chunks
            
        except Exception as e:
            logger.warning(
                f"[GENERATE RETRY] {doc.title} failed "
                f"(attempt {attempt}" + (f"/{max_attempts}" if max_attempts else "") + f"). "
                f"Retrying in {delay}s... Error: {e}"
            )
            
            # Log persistent issues
            if attempt % 10 == 0:
                logger.error(
                    f"[GENERATE PERSISTENT] {doc.title} still failing after {attempt} attempts. "
                    f"This may indicate a persistent issue."
                )
            
            await asyncio.sleep(delay)
            
            # Exponential backoff with cap
            delay = min(
                delay * SANITY_CHECK_BACKOFF_MULTIPLIER,
                SANITY_CHECK_MAX_DELAY
            )

