"""
Pinecone query utilities

Handles querying Pinecone for statistics and document vectors.
"""

import logging
from typing import Dict, Any, List

from rag.app.db.pinecone_connection import PineconeEmbeddingStore

logger = logging.getLogger(__name__)


def get_pinecone_stats(
    connection: PineconeEmbeddingStore,
    namespace: str
) -> Dict[str, Any]:
    """
    Get statistics from Pinecone for a specific namespace.
    
    Args:
        connection: Pinecone connection
        namespace: Namespace to query
        
    Returns:
        Dictionary with vector_count and other stats
    """
    try:
        stats = connection.index.describe_index_stats()
        
        if namespace:
            namespace_stats = stats.namespaces.get(namespace, {})
            vector_count = namespace_stats.get("vector_count", 0)
        else:
            vector_count = stats.total_vector_count
        
        return {
            "vector_count": vector_count,
            "namespace": namespace,
            "total_vectors": stats.total_vector_count,
            "namespaces": list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') else [],
        }
    except Exception as e:
        logger.error(f"[PINECONE STATS] Failed to get stats: {e}")
        return {
            "vector_count": 0,
            "namespace": namespace,
            "error": str(e)
        }


async def query_document_vectors(
    connection: PineconeEmbeddingStore,
    namespace: str,
    sanity_id: str
) -> List[Dict[str, Any]]:
    """
    Query all vectors for a specific document (sanity_id) in a namespace.
    
    Uses Pinecone's query with metadata filter to find all chunks belonging
    to a specific document.
    
    Args:
        connection: Pinecone connection
        namespace: Namespace to query
        sanity_id: Document ID to filter by
        
    Returns:
        List of dictionaries with id, score, and metadata for each vector
    """
    try:
        # Create a dummy vector (zeros) just to query with filter
        # We'll use top_k=10000 to get all matches
        dimension = connection.dimension
        dummy_vector = [0.0] * dimension
        
        # Query with metadata filter for this document
        response = connection.index.query(
            vector=dummy_vector,
            top_k=10000,  # High number to get all chunks
            namespace=namespace,
            filter={"sanity_id": {"$eq": sanity_id}},
            include_metadata=True,
        )
        
        results = [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
            }
            for match in response.matches
        ]
        
        logger.debug(
            f"[PINECONE QUERY] Found {len(results)} vectors for "
            f"sanity_id={sanity_id} in namespace={namespace}"
        )
        
        return results
        
    except Exception as e:
        logger.warning(
            f"[PINECONE QUERY] Failed to query vectors for {sanity_id}: {e}"
        )
        return []

