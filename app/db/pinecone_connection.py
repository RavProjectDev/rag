from datetime import datetime
from typing import Dict, List, Any
import logging
import hashlib

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException

from rag.app.db.connections import (
    EmbeddingConnection,
    MetricsConnection,
    ExceptionsLogger,
)
from rag.app.exceptions.db import (
    RetrievalException,
    DataBaseException,
    InsertException,
    NoDocumentFoundException,
)
from rag.app.schemas.data import VectorEmbedding, SanityData, TranscriptData
from rag.app.models.data import DocumentModel, Metadata

logger = logging.getLogger(__name__)


class PineconeEmbeddingStore(EmbeddingConnection):
    """
    Pinecone implementation of EmbeddingConnection.
    
    This class provides vector storage and retrieval using Pinecone's vector database.
    """
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 784,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        environment: str | None = None,  # Deprecated: no longer needed, kept for backward compatibility
    ):
        """
        Initialize Pinecone connection.
        
        Args:
            api_key: Pinecone API key (automatically detects environment/region)
            index_name: Name of the Pinecone index
            dimension: Vector dimension (default: 784 for Gemini)
            metric: Distance metric (default: "cosine")
            cloud: Cloud provider (default: "aws")
            region: Cloud region (default: "us-east-1")
            environment: Deprecated - no longer needed. Pinecone SDK auto-detects from API key.
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        
        # Initialize Pinecone client (automatically detects environment/region from API key)
        self.pc = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        self._ensure_index_exists()
        
        # Get index reference
        self.index = self.pc.Index(index_name)
        
        logger.info(
            f"[PINECONE] Initialized connection to index '{index_name}' "
            f"with dimension={dimension}, metric={metric}"
        )
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"[PINECONE] Creating index '{self.index_name}'...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
                logger.info(f"[PINECONE] Index '{self.index_name}' created successfully")
            else:
                logger.info(f"[PINECONE] Index '{self.index_name}' already exists")
                
        except Exception as e:
            logger.error(f"[PINECONE ERROR] Failed to ensure index exists: {e}")
            raise DataBaseException(f"Failed to initialize Pinecone index: {e}")
    
    def _generate_unique_id(self, emb: 'VectorEmbedding') -> str:
        """
        Generate a unique ID for a chunk embedding.
        
        Uses sanity_id + full_text_id + hash of text_to_embed to ensure uniqueness
        even when multiple chunks share the same full_text_id.
        
        Args:
            emb: VectorEmbedding object
            
        Returns:
            Unique string ID for the chunk
        """
        # Hash the text_to_embed to make each chunk unique
        text_hash = hashlib.md5(emb.metadata.text_to_embed.encode()).hexdigest()[:8]
        return f"{emb.sanity_data.id}_{emb.metadata.full_text_id}_{text_hash}"
    
    async def insert(self, embedded_data: list[VectorEmbedding], namespace: str | None = None):
        """
        Insert embeddings into Pinecone.
        
        Args:
            embedded_data: List of VectorEmbedding objects to insert
            namespace: Optional namespace to insert into
        """
        if not embedded_data:
            return
        
        try:
            # Generate unique IDs for each chunk (sanity_id + full_text_id + text hash)
            # This ensures each chunk has a unique ID even when multiple chunks share the same full_text_id
            ids_to_check = [self._generate_unique_id(emb) for emb in embedded_data]
            
            # Fetch existing vectors (with namespace if provided)
            existing_ids = set()
            if namespace:
                try:
                    # Use query to check existing vectors in namespace
                    # Note: This is a workaround since fetch doesn't support namespace filtering
                    # For better performance, maintain a separate tracking mechanism
                    fetch_response = self.index.fetch(ids=ids_to_check, namespace=namespace)
                    existing_ids = set(fetch_response.vectors.keys())
                except Exception as e:
                    logger.warning(f"[PINECONE] Could not fetch existing IDs in namespace {namespace}: {e}. Proceeding with insert.")
            else:
                try:
                    fetch_response = self.index.fetch(ids=ids_to_check)
                    existing_ids = set(fetch_response.vectors.keys())
                except Exception as e:
                    logger.warning(f"[PINECONE] Could not fetch existing IDs: {e}. Proceeding with insert.")
            
            # Filter out existing embeddings
            embeddings_to_insert = [
                emb for emb in embedded_data 
                if self._generate_unique_id(emb) not in existing_ids
            ]
            
            if not embeddings_to_insert:
                logger.info(f"[PINECONE] No new embeddings to insert (all already exist) in namespace: {namespace}")
                return
            
            # Prepare vectors for upsert
            vectors_to_upsert = []
            for emb in embeddings_to_insert:
                # Use unique ID combining sanity_id, full_text_id, and text hash
                unique_id = self._generate_unique_id(emb)
                vector_dict = {
                    "id": unique_id,
                    "values": emb.vector,
                    "metadata": {
                        "text": emb.metadata.full_text,
                        "text_id": str(emb.metadata.full_text_id),
                        "chunk_size": emb.metadata.chunk_size,
                        "embed_size": emb.metadata.embed_size,
                        "name_space": emb.metadata.name_space,
                        "time_start": emb.metadata.time_start,
                        "time_end": emb.metadata.time_end,
                        "sanity_id": emb.sanity_data.id,
                        "sanity_slug": emb.sanity_data.slug,
                        "sanity_title": emb.sanity_data.title,
                        "sanity_hash": emb.sanity_data.hash,
                        "sanity_transcript_url": str(emb.sanity_data.transcriptURL),
                    }
                }
                vectors_to_upsert.append(vector_dict)
            
            # Upsert in batches (Pinecone recommends batch size of 100)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                logger.info(
                    f"[PINECONE] Upserted batch {i//batch_size + 1} "
                    f"({len(batch)} vectors) in namespace: {namespace}"
                )
            
            logger.info(
                f"[PINECONE] Successfully inserted {len(embeddings_to_insert)} embeddings "
                f"in namespace: {namespace}"
            )
            
        except PineconeException as e:
            logger.error(f"[PINECONE ERROR] Insert failed: {e}", exc_info=True)
            raise InsertException(f"Failed to insert into Pinecone: {e}")
        except Exception as e:
            logger.error(f"[PINECONE ERROR] Unexpected error during insert: {e}", exc_info=True)
            raise DataBaseException(f"Failed to insert documents: {e}")
    
    async def retrieve(
        self,
        embedded_data: List[float],
        name_spaces: list[str] | None = None,
        k: int = 5,
        threshold: float = 0.7,
    ) -> list[DocumentModel]:
        """
        Retrieve similar documents from Pinecone.
        
        Args:
            embedded_data: Query vector
            name_spaces: Optional list of namespaces to filter by
            k: Number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of DocumentModel objects
        """
        try:
            logger.info(
                f"[PINECONE] Starting vector search, k={k}, threshold={threshold}, "
                f"name_spaces={name_spaces}, vector_dimension={len(embedded_data)}"
            )
            
            # Build filter for namespaces if provided
            filter_dict = None
            if name_spaces:
                filter_dict = {"name_space": {"$in": name_spaces}}
            
            # Query Pinecone
            query_response = self.index.query(
                vector=embedded_data,
                top_k=k * 3,  # Get more results to account for filtering
                include_metadata=True,
                filter=filter_dict
            )
            
            # Process results
            documents: List[DocumentModel] = []
            seen_text_ids = set()
            
            for match in query_response.matches:
                score = match.score
                
                # Filter by threshold
                if score < threshold:
                    continue
                
                metadata = match.metadata
                text_id = metadata.get("text_id")
                
                # Deduplicate by text_id
                if text_id in seen_text_ids:
                    logger.debug(f"[PINECONE] Skipping duplicate text_id: {text_id}")
                    continue
                seen_text_ids.add(text_id)
                
                # Create Metadata object
                chunk_metadata = Metadata(
                    chunk_size=metadata.get("chunk_size", 0),
                    embed_size=metadata.get("embed_size", 0),
                    name_space=metadata.get("name_space", ""),
                    time_start=metadata.get("time_start"),
                    time_end=metadata.get("time_end"),
                    full_text_id=text_id,
                )
                
                # Create SanityData object
                sanity_data = SanityData(
                    id=metadata.get("sanity_id", ""),
                    slug=metadata.get("sanity_slug", ""),
                    title=metadata.get("sanity_title", ""),
                    hash=metadata.get("sanity_hash", ""),
                    transcriptURL=metadata.get("sanity_transcript_url", ""),
                )
                
                # Create DocumentModel
                document = DocumentModel(
                    _id=match.id,
                    text=metadata.get("text", ""),
                    metadata=chunk_metadata,
                    sanity_data=sanity_data,
                    score=float(score),
                )
                documents.append(document)
                
                # Stop if we have enough results
                if len(documents) >= k:
                    break
            
            if not documents:
                logger.warning(
                    f"[PINECONE] No documents found above threshold={threshold}"
                )
                raise NoDocumentFoundException
            
            logger.info(
                f"[PINECONE] Successfully retrieved {len(documents)} unique documents, "
                f"score_range=[{documents[-1].score:.4f}, {documents[0].score:.4f}]"
            )
            return documents
            
        except NoDocumentFoundException:
            raise
        except PineconeException as e:
            logger.error(f"[PINECONE ERROR] Retrieval failed: {e}", exc_info=True)
            raise RetrievalException(f"Pinecone failed to retrieve documents: {e}")
        except Exception as e:
            logger.error(f"[PINECONE ERROR] Unexpected error during retrieval: {e}", exc_info=True)
            raise DataBaseException(f"Failed to retrieve documents: {e}")
    
    async def get_all_unique_transcript_ids(self, namespace: str | None = None) -> list[TranscriptData]:
        """
        Get all unique transcript IDs from Pinecone.
        
        Note: This is an expensive operation in Pinecone as it requires
        fetching all vectors. Consider using metadata filtering or
        maintaining a separate index for this.
        
        Args:
            namespace: Optional namespace to filter by
        """
        try:
            logger.info(f"[PINECONE] Fetching all unique transcript IDs (namespace: {namespace})...")
            
            # Get index stats (with namespace if provided)
            stats = self.index.describe_index_stats()
            
            if namespace:
                namespace_stats = stats.namespaces.get(namespace, {})
                total_vectors = namespace_stats.get("vector_count", 0)
            else:
                total_vectors = stats.total_vector_count
            
            logger.warning(
                f"[PINECONE] get_all_unique_transcript_ids is expensive with {total_vectors} vectors "
                f"(namespace: {namespace}). Consider using a metadata store for this operation."
            )
            
            # For now, return empty list with a warning
            # In production, you should maintain transcript metadata separately
            # The insert method will handle duplicate checking via fetch
            logger.info("[PINECONE] Returning empty list - duplicate checking handled in insert method")
            return []
            
        except Exception as e:
            logger.error(f"[PINECONE ERROR] Failed to get transcript IDs: {e}", exc_info=True)
            raise DataBaseException(f"Failed to get transcript IDs: {e}")
    
    async def delete_document(self, transcript_id: str) -> bool:
        """
        Delete all vectors associated with a transcript ID.
        
        Args:
            transcript_id: The transcript ID to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            logger.info(f"[PINECONE] Deleting document with transcript_id: {transcript_id}")
            
            # Delete by ID (assuming transcript_id is used as vector ID)
            self.index.delete(ids=[transcript_id])
            
            logger.info(f"[PINECONE] Successfully deleted document: {transcript_id}")
            return True
            
        except Exception as e:
            logger.error(f"[PINECONE ERROR] Failed to delete document: {e}", exc_info=True)
            return False


class PineconeMetricsConnection(MetricsConnection):
    """
    Metrics logger for Pinecone (stores metrics in Pinecone as vectors).
    Note: This is not recommended for production - use a dedicated metrics store.
    """
    
    def __init__(self, index_name: str, api_key: str):
        self.index_name = index_name
        self.api_key = api_key
        logger.warning(
            "[PINECONE METRICS] Using Pinecone for metrics is not recommended. "
            "Consider using a dedicated metrics store like MongoDB or CloudWatch."
        )
    
    async def log(self, metric_type: str, data: Dict[str, Any]):
        """Log metrics - placeholder implementation."""
        logger.info(f"[PINECONE METRICS] {metric_type}: {data}")


class PineconeExceptionsLogger(ExceptionsLogger):
    """
    Exception logger for Pinecone (stores exceptions in Pinecone as vectors).
    Note: This is not recommended for production - use a dedicated exception store.
    """
    
    def __init__(self, index_name: str, api_key: str):
        self.index_name = index_name
        self.api_key = api_key
        logger.warning(
            "[PINECONE EXCEPTIONS] Using Pinecone for exceptions is not recommended. "
            "Consider using a dedicated exception store like MongoDB or Sentry."
        )
    
    async def log(self, exception_code: str | None, data: Dict[str, Any]):
        """Log exceptions - placeholder implementation."""
        if not exception_code:
            exception_code = "Base App Exception"
        logger.error(f"[PINECONE EXCEPTIONS] {exception_code}: {data}")

