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
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str | None = None,
        namespace: str | None = None,
        host: str | None = None,
    ):
        if not api_key:
            raise ValueError("pinecone_api_key is required when using Pinecone.")
        if not index_name and not host:
            raise ValueError("pinecone_index_name or pinecone_host is required.")

        # New Pinecone API
        pc = Pinecone(api_key=api_key)
        
        # Get the index - host takes precedence when provided (serverless-style connection)
        if host:
            self.index = pc.Index(host=host)
        else:
            self.index = pc.Index(index_name)
        
        self.namespace = namespace

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
        except Exception as e:
            logger.exception("Failed to upsert vectors into Pinecone.")
            raise InsertException(f"Failed to upsert vectors into Pinecone: {e}")

    async def retrieve(
        self,
        embedded_data: List[float],
        name_spaces: list[str] | None = None,
        k=5,
        threshold: float = 0.7,
    ) -> list[DocumentModel]:
        filter_payload = None
        if name_spaces:
            filter_payload = {"name_space": {"$in": name_spaces}}
        try:
            result = await asyncio.to_thread(
                self.index.query,
                vector=embedded_data,
                top_k=max(k * 3, k),
                include_metadata=True,
                namespace=self.namespace,
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

