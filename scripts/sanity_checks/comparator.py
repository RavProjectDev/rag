"""
Chunk comparison logic

Compares expected chunks (from preprocessing) against actual chunks (in Pinecone).
"""

import logging
from typing import Dict, Any, List, Tuple

from rag.app.models.data import SanityData
from rag.app.schemas.data import (
    EmbeddingConfiguration,
    ChunkingStrategy,
    Chunk,
)
from rag.app.db.pinecone_connection import PineconeEmbeddingStore

from .chunk_generator import generate_chunks_with_infinite_retry, generate_chunk_id
from .pinecone_query import query_document_vectors

logger = logging.getLogger(__name__)


async def calculate_expected_chunk_ids(
    doc: SanityData,
    chunking_strategy: ChunkingStrategy,
    embedding_config: EmbeddingConfiguration,
    max_attempts: int | None = None
) -> Tuple[List[str], List[Chunk], Dict[str, Chunk]]:
    """
    Calculate expected chunk IDs with infinite retries.
    
    Args:
        doc: Document to process
        chunking_strategy: Strategy to use for chunking
        embedding_config: Embedding configuration
        max_attempts: Optional maximum number of attempts (None = infinite)
        
    Returns:
        Tuple of (chunk_ids, chunks_list, id_to_chunk_map)
    """
    # Generate chunks (will retry forever or until max_attempts)
    chunks = await generate_chunks_with_infinite_retry(
        doc, chunking_strategy, embedding_config, max_attempts
    )
    
    # Generate IDs using same logic as upload
    chunk_ids = []
    id_to_chunk = {}
    
    for chunk in chunks:
        chunk_id = generate_chunk_id(
            sanity_id=doc.id,
            full_text_id=chunk.full_text_id,
            text_to_embed=chunk.text_to_embed
        )
        chunk_ids.append(chunk_id)
        id_to_chunk[chunk_id] = chunk
    
    return chunk_ids, chunks, id_to_chunk


async def check_document_chunks(
    doc: SanityData,
    embedding_config: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
    connection: PineconeEmbeddingStore,
    namespace: str,
    max_attempts: int | None = None
) -> Dict[str, Any]:
    """
    Check which specific chunks are missing with detailed information.
    
    Uses infinite retries for chunk generation to ensure we have complete expected state.
    Compares text-based chunk IDs (no embedding generation needed).
    
    Args:
        doc: Document to check
        embedding_config: Embedding configuration
        chunking_strategy: Chunking strategy
        connection: Pinecone connection
        namespace: Namespace to query
        max_attempts: Optional maximum number of attempts (None = infinite)
        
    Returns:
        Dictionary with detailed consistency report
    """
    logger.info(
        f"[CHECK] {doc.title} | "
        f"embedding={embedding_config.value} | "
        f"chunking={chunking_strategy.value}"
    )
    
    try:
        # Get expected chunk IDs (will retry forever until success)
        expected_ids, chunks, id_to_chunk = await calculate_expected_chunk_ids(
            doc, chunking_strategy, embedding_config, max_attempts
        )
        expected_set = set(expected_ids)
        
        logger.info(
            f"[CHECK] {doc.title}: Generated {len(expected_ids)} expected chunks"
        )
        
        # Get actual chunk IDs from Pinecone
        actual_vectors = await query_document_vectors(
            connection, namespace, doc.id
        )
        actual_ids = [v["id"] for v in actual_vectors]
        actual_set = set(actual_ids)
        
        logger.info(
            f"[CHECK] {doc.title}: Found {len(actual_ids)} actual chunks in Pinecone"
        )
        
        # Find differences
        missing_ids = expected_set - actual_set
        extra_ids = actual_set - expected_set
        
        # Determine status
        if len(missing_ids) == 0 and len(extra_ids) == 0:
            status = "OK"
        elif len(actual_ids) == 0:
            status = "MISSING_ALL"
        elif len(missing_ids) > 0:
            status = "INCOMPLETE"
        else:
            status = "MISMATCH"
        
        # Build detailed report
        report = {
            "document_id": doc.id,
            "document_title": doc.title,
            "document_url": str(doc.transcriptURL),
            "expected_chunk_count": len(expected_set),
            "actual_chunk_count": len(actual_set),
            "status": status,
            "missing_chunks": {
                "count": len(missing_ids),
                "ids": sorted(list(missing_ids)),
                "details": []
            },
            "extra_chunks": {
                "count": len(extra_ids),
                "ids": sorted(list(extra_ids))
            }
        }
        
        # Add detailed information about each missing chunk
        for chunk_id in sorted(missing_ids):
            chunk = id_to_chunk[chunk_id]
            text_preview = chunk.text_to_embed[:200]
            if len(chunk.text_to_embed) > 200:
                text_preview += "..."
            
            report["missing_chunks"]["details"].append({
                "chunk_id": chunk_id,
                "full_text_id": chunk.full_text_id,
                "text_preview": text_preview,
                "text_length": len(chunk.text_to_embed),
                "time_start": chunk.time_start,
                "time_end": chunk.time_end,
                "chunk_size": chunk.chunk_size,
                "embed_size": chunk.embed_size,
            })
        
        # Log summary
        if status == "OK":
            logger.info(f"[CHECK OK] {doc.title}: All chunks present")
        else:
            logger.warning(
                f"[CHECK ISSUE] {doc.title}: "
                f"Missing={len(missing_ids)}, Extra={len(extra_ids)}"
            )
        
        return report
        
    except Exception as e:
        # This should theoretically never happen due to infinite retries
        # But catch it just in case
        logger.error(
            f"[CHECK ERROR] {doc.title}: Unexpected error: {e}",
            exc_info=True
        )
        return {
            "document_id": doc.id,
            "document_title": doc.title,
            "document_url": str(doc.transcriptURL),
            "status": "ERROR",
            "error": str(e),
            "expected_chunk_count": 0,
            "actual_chunk_count": 0,
        }

