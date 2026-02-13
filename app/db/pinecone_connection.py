import asyncio
import json
import logging
import uuid
from typing import List

from pinecone import Pinecone

from rag.app.db.connections import EmbeddingConnection
from rag.app.exceptions.db import (
    DataBaseException,
    InsertException,
    RetrievalException,
    NoDocumentFoundException,
)
from rag.app.models.data import DocumentModel, Metadata, SanityData
from rag.app.schemas.data import TranscriptData, VectorEmbedding

logger = logging.getLogger(__name__)


class PineconeEmbeddingStore(EmbeddingConnection):
    """
    Pinecone-backed implementation of the EmbeddingConnection interface.

    All Pinecone operations are executed in a background thread to avoid blocking
    the event loop, since the official client is synchronous.
    
    Architecture:
    - Index name: Based on embedding model (e.g., "gemini-embedding-001", "text-embedding-3-large")
    - Namespace: Based on chunking strategy (e.g., "fixed_size", "divided")
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str | None = None,
        namespace: str | None = None,
        host: str | None = None,
        chunks_collection=None,
    ):
        if not api_key:
            raise ValueError("pinecone_api_key is required when using Pinecone.")
        if not index_name and not host:
            raise ValueError("pinecone_index_name or pinecone_host is required.")

        # Store Pinecone client for creating dynamic index connections
        self.pc = Pinecone(api_key=api_key)
        
        # Store default configuration (from environment variables)
        self.default_index_name = index_name
        self.default_namespace = namespace
        
        # Get the default index - host takes precedence when provided (serverless-style connection)
        if host:
            self.index = self.pc.Index(host=host)
        else:
            self.index = self.pc.Index(index_name)
        
        self.namespace = namespace
        self.chunks_collection = chunks_collection
        
        # Cache for dynamically created index objects
        self._index_cache = {index_name: self.index}
        
        logger.info(
            f"[PINECONE] Initialized with default_index={index_name}, default_namespace={namespace}"
        )

    def _get_index(self, index_name: str):
        """
        Get or create a cached index object for the given index name.
        
        Args:
            index_name: Name of the Pinecone index
            
        Returns:
            Pinecone Index object
        """
        if index_name not in self._index_cache:
            self._index_cache[index_name] = self.pc.Index(index_name)
            logger.info(f"[PINECONE] Created new index connection: {index_name}")
        return self._index_cache[index_name]

    @staticmethod
    def _build_metadata(embedding: VectorEmbedding) -> dict:
        metadata = embedding.metadata
        sanity = embedding.sanity_data
        
        # Store text with fine-grained timestamps
        # Pinecone metadata only supports primitives, so serialize complex structures as JSON
        if isinstance(metadata.full_text, str):
            # Plain text (TXT files)
            text_value = metadata.full_text
        elif isinstance(metadata.full_text, list):
            # SRT chunks: list of [(text, [start, end]), ...]
            # Serialize as JSON string for Pinecone storage
            text_value = json.dumps(metadata.full_text)
        else:
            text_value = str(metadata.full_text)
        
        # Build metadata dict, filtering out None values (Pinecone doesn't accept nulls)
        metadata_dict = {
            "text": text_value,  # Stores structured data (JSON string for lists)
            "text_id": str(metadata.full_text_id),
            "chunk_size": metadata.chunk_size,
            "name_space": metadata.name_space,
            "text_hash": metadata.text_hash,  # SHA-256 hash of the text
            "sanity_id": sanity.id,
            "sanity_slug": sanity.slug,
            "sanity_title": sanity.title,
            "sanity_transcriptURL": str(sanity.transcriptURL),
            "sanity_hash": sanity.hash,
        }
        
        # Add optional fields only if they're not None
        if metadata.time_start is not None:
            metadata_dict["time_start"] = metadata.time_start
        if metadata.time_end is not None:
            metadata_dict["time_end"] = metadata.time_end
        if sanity.updated_at is not None:
            metadata_dict["sanity_updated_at"] = sanity.updated_at
            
        return metadata_dict

    async def insert(self, embedded_data: List[VectorEmbedding]):
        if not embedded_data:
            return
        vectors = []
        for emb in embedded_data:
            vectors.append(
                {
                    "id": str(uuid.uuid4()),  # Generate unique ID for each Pinecone vector
                    "values": emb.vector,
                    "metadata": self._build_metadata(emb),  # text_id stored in metadata for grouping
                }
            )
        try:
            await asyncio.to_thread(
                self.index.upsert, vectors=vectors, namespace=self.namespace
            )
            
            # Track chunks in MongoDB if configured
            if self.chunks_collection is not None:
                await self._track_chunks(embedded_data)
                
        except Exception as e:
            logger.exception("Failed to upsert vectors into Pinecone.")
            raise InsertException(f"Failed to upsert vectors into Pinecone: {e}")
    
    async def _track_chunks(self, embeddings: List[VectorEmbedding]):
        """
        Track chunks in a separate MongoDB collection for deduplication and indexing.
        
        Each chunk record contains all metadata from Pinecone:
        - _id: SHA-256 hash of the chunk text
        - text_id: The full_text_id (UUID) that groups related chunks
        - chunk_index: Global sequential index across all chunks
        - text: The chunk text (or JSON string for SRT)
        - chunk_size: Token count
        - name_space: Document identifier
        - sanity_id, sanity_slug, sanity_title, sanity_transcriptURL, sanity_hash
        - time_start, time_end (optional for SRT)
        - sanity_updated_at (optional)
        """
        if not embeddings:
            return
        
        try:
            # Get the maximum existing chunk_index globally (across all chunks)
            max_index_doc = await self.chunks_collection.find_one(
                {},
                sort=[("chunk_index", -1)]
            )
            start_index = (max_index_doc["chunk_index"] + 1) if max_index_doc else 0
            
            # Create chunk records with all metadata
            chunk_records = []
            for i, emb in enumerate(embeddings):
                metadata = emb.metadata
                sanity = emb.sanity_data
                text_id = str(metadata.full_text_id)
                
                # Prepare text value (same logic as Pinecone)
                if isinstance(metadata.full_text, str):
                    text_value = metadata.full_text
                elif isinstance(metadata.full_text, list):
                    text_value = json.dumps(metadata.full_text)
                else:
                    text_value = str(metadata.full_text)
                
                # Build chunk record with all metadata
                chunk_record = {
                    "_id": metadata.text_hash,  # SHA-256 hash as primary key
                    "text_id": text_id,
                    "chunk_index": start_index + i,
                    "text": text_value,
                    "chunk_size": metadata.chunk_size,
                    "name_space": metadata.name_space,
                    "embedding_model": emb.embedding_model,
                    "chunking_strategy": emb.chunking_strategy,
                    "sanity_id": sanity.id,
                    "sanity_slug": sanity.slug,
                    "sanity_title": sanity.title,
                    "sanity_transcriptURL": str(sanity.transcriptURL),
                    "sanity_hash": sanity.hash,
                }
                
                # Add optional fields only if they're not None
                if metadata.time_start is not None:
                    chunk_record["time_start"] = metadata.time_start
                if metadata.time_end is not None:
                    chunk_record["time_end"] = metadata.time_end
                if sanity.updated_at is not None:
                    chunk_record["sanity_updated_at"] = sanity.updated_at
                
                chunk_records.append(chunk_record)
            
            # Insert chunk records (use insert_many with ordered=False to skip duplicates)
            if chunk_records:
                try:
                    await self.chunks_collection.insert_many(chunk_records, ordered=False)
                    logger.info(f"[CHUNKS] Successfully tracked {len(chunk_records)} chunks with full metadata (indices {start_index}-{start_index + len(chunk_records) - 1})")
                except Exception as e:
                    # Some chunks might already exist (duplicate hashes), which is fine
                    logger.debug(f"[CHUNKS] Some chunks already exist (duplicates): {e}")
                    
        except Exception as e:
            logger.warning(f"[CHUNKS] Failed to track chunks: {e}", exc_info=True)
            # Don't raise - chunk tracking is supplementary, shouldn't block main insert

    async def retrieve(
        self,
        embedded_data: List[float],
        name_spaces: list[str] | None = None,
        k=7,
        threshold: float = 0.65,
        index_override: str | None = None,
        namespace_override: str | None = None,
    ) -> list[DocumentModel]:
        # Determine which index to use (defaults to env config if not specified)
        if index_override:
            query_index = self._get_index(index_override)
            query_index_name = index_override
        else:
            query_index = self.index
            query_index_name = self.default_index_name
        
        # Determine which namespace to use (defaults to env config if not specified)
        query_namespace = namespace_override if namespace_override is not None else self.namespace
        
        filter_payload = None
        if name_spaces:
            filter_payload = {"name_space": {"$in": name_spaces}}
        
        logger.info(
            f"[PINECONE QUERY] index={query_index_name}{' (default)' if not index_override else ''}, "
            f"namespace={query_namespace}{' (default)' if namespace_override is None else ''}, "
            f"name_spaces_filter={name_spaces}"
        )
        
        try:
            result = await asyncio.to_thread(
                query_index.query,
                vector=embedded_data,
                top_k=max(k * 3, k),
                include_metadata=True,
                namespace=query_namespace,
                filter=filter_payload,
            )
        except Exception as e:
            logger.exception("Pinecone query failed.")
            raise RetrievalException(f"Pinecone failed to retrieve documents: {e}")

        matches = result.get("matches") or []
        documents: list[DocumentModel] = []
        seen_text_ids = set()

        for match in matches:
            score = match.get("score") or 0
            if score < threshold:
                continue
            metadata_raw = match.get("metadata") or {}
            text_id = metadata_raw.get("text_id") or match.get("id")
            if text_id in seen_text_ids:
                continue
            seen_text_ids.add(text_id)

            try:
                # Deserialize text field if it's JSON (from SRT chunks)
                text_raw = metadata_raw.get("text", "")
                try:
                    # Try to parse as JSON - if it's a list of tuples, it was serialized
                    text_value = json.loads(text_raw)
                except (json.JSONDecodeError, TypeError):
                    # Plain string (TXT files) or already deserialized
                    text_value = text_raw
                
                metadata = Metadata(
                    chunk_size=metadata_raw.get("chunk_size", 0),
                    time_start=metadata_raw.get("time_start"),
                    time_end=metadata_raw.get("time_end"),
                    name_space=metadata_raw.get("name_space", ""),
                    text_hash=metadata_raw.get("text_hash"),
                )
                sanity_data = SanityData(
                    id=metadata_raw.get("sanity_id", ""),
                    slug=metadata_raw.get("sanity_slug", ""),
                    title=metadata_raw.get("sanity_title", ""),
                    transcriptURL=metadata_raw.get("sanity_transcriptURL", ""),
                    hash=metadata_raw.get("sanity_hash", ""),
                    _updatedAt=metadata_raw.get("sanity_updated_at"),
                )
                document = DocumentModel(
                    _id=str(text_id),
                    text=text_value,  # Deserialized structured data or plain string
                    metadata=metadata,
                    sanity_data=sanity_data,
                    score=float(score),
                )
                documents.append(document)
            except Exception as e:
                logger.warning(
                    "Skipping Pinecone match due to parsing error: %s; raw metadata=%s",
                    e,
                    metadata_raw,
                )
                continue

            if len(documents) >= k:
                break

        if not documents:
            raise NoDocumentFoundException
        return documents

    async def get_all_unique_transcript_ids(self) -> list[TranscriptData]:
        # Pinecone does not currently provide an efficient server-side listing of
        # unique metadata values. Returning an empty list keeps the sync job from
        # performing destructive operations by default.
        logger.info(
            "PineconeEmbeddingStore.get_all_unique_transcript_ids is not supported; returning empty list."
        )
        return []

    async def delete_document(self, transcript_id: str) -> bool:
        try:
            await asyncio.to_thread(
                self.index.delete,
                filter={"sanity_id": {"$eq": transcript_id}},
                namespace=self.namespace,
            )
            return True
        except Exception as e:
            raise DataBaseException(f"Failed to delete document in Pinecone: {e}")

