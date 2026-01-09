"""
Sanity Check Module for RAG Database

This module verifies data consistency between expected state (from manifest + preprocessing)
and actual state (in Pinecone vector database).

Key Features:
- Text-based comparison (no embedding generation needed)
- Infinite retries for chunk generation
- Detailed reports of missing/extra chunks
- Support for all embedding models and chunking strategies
"""

from .config import (
    EMBEDDING_MODELS,
    CHUNKING_STRATEGIES,
    EMBEDDING_DIMENSIONS,
    get_index_name,
    get_namespace_name,
    get_embedding_dimension,
)
from .fetch import fetch_manifest, fetch_transcript_with_infinite_retry
from .chunk_generator import generate_chunks_with_infinite_retry, generate_chunk_id
from .pinecone_query import get_pinecone_stats, query_document_vectors
from .comparator import calculate_expected_chunk_ids, check_document_chunks
from .reporter import generate_report, save_report

__all__ = [
    "EMBEDDING_MODELS",
    "CHUNKING_STRATEGIES",
    "EMBEDDING_DIMENSIONS",
    "get_index_name",
    "get_namespace_name",
    "get_embedding_dimension",
    "fetch_manifest",
    "fetch_transcript_with_infinite_retry",
    "generate_chunks_with_infinite_retry",
    "generate_chunk_id",
    "get_pinecone_stats",
    "query_document_vectors",
    "calculate_expected_chunk_ids",
    "check_document_chunks",
    "generate_report",
    "save_report",
]

