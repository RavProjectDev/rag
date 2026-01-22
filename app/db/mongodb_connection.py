from datetime import datetime
from typing import Dict, List, Any
import logging
import asyncio

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import OperationFailure, ExecutionTimeout

from rag.app.db.connections import (
    EmbeddingConnection,
    MetricsConnection,
    ExceptionsLogger,
)
from rag.app.exceptions.db import (
    RetrievalException,
    DataBaseException,
    InsertException,
    RetrievalTimeoutException,
    NoDocumentFoundException,
)
from rag.app.schemas.data import VectorEmbedding, SanityData, TranscriptData
from rag.app.models.data import DocumentModel, Metadata
from rag.app.core.config import get_settings

logger = logging.getLogger(__name__)


class MongoEmbeddingStore(EmbeddingConnection):
    def __init__(self, collection, index: str, vector_path: str, chunks_collection: AsyncIOMotorCollection | None = None):
        self.collection = collection
        self.index = index
        self.vector_path = vector_path
        self.chunks_collection = chunks_collection

    async def insert(self, embedded_data: list[VectorEmbedding]):
        ids_to_insert = list({emb.sanity_data.id for emb in embedded_data})
        if not ids_to_insert:
            return
        try:
            cursor = self.collection.find(
                {"sanity_data.id": {"$in": ids_to_insert}},  # Query for existing IDs
                {"sanity_data.id": 1},  # Only return the sanity_data.id field
            )
            existing = await cursor.to_list(length=None)
            existing_ids = {doc["sanity_data"]["id"] for doc in existing}
            embeddings_to_insert = [
                emb for emb in embedded_data if emb.sanity_data.id not in existing_ids
            ]
            documents = [emb.to_dict() for emb in embeddings_to_insert]
            if documents:
                await self.collection.insert_many(documents)
                
                # Track chunks in separate collection if configured
                if self.chunks_collection is not None:
                    await self._track_chunks(embeddings_to_insert)
                    
        except OperationFailure as e:
            raise InsertException(
                f"Failed to insert documents, Mongo config error: {e}"
            )
        except Exception as e:
            raise DataBaseException(f"Failed to insert documents: {e}")
    
    async def _track_chunks(self, embeddings: list[VectorEmbedding]):
        """
        Track chunks in a separate collection for deduplication and indexing.
        
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
            import json
            
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
        k=5,
        threshold: float = 0.85,
    ) -> list[DocumentModel]:
        settings = get_settings()
        max_retries = settings.max_retry_attempts
        timeout_ms = settings.retrieval_timeout_ms
        retry_delay = settings.retry_delay_seconds
        backoff_multiplier = settings.retry_backoff_multiplier

        # Increase the initial limit to account for potential duplicates
        initial_limit = min(k * 3, 1000)  # Get more documents initially

        logger.info(
            f"[RETRIEVE] Starting vector search in collection='{self.collection.name}', "
            f"index='{self.index}', vector_path='{self.vector_path}', "
            f"k={k}, threshold={threshold}, name_spaces={name_spaces}, "
            f"initial_limit={initial_limit}, vector_dimension={len(embedded_data)}, "
            f"max_retries={max_retries}, timeout_ms={timeout_ms}"
        )

        pipeline = []
        # 1. $vectorSearch first
        pipeline.append(
            {
                "$vectorSearch": {
                    "index": self.index,
                    "path": self.vector_path,
                    "queryVector": embedded_data,
                    "numCandidates": 300,
                    "limit": initial_limit,  # Use higher limit initially
                    "metric": "cosine",
                }
            }
        )
        # 2. Then optionally filter with $match if you have name_spaces
        if name_spaces is not None and len(name_spaces) > 0:
            pipeline.append({"$match": {"metadata.name_space": {"$in": name_spaces}}})
            logger.debug(f"[RETRIEVE] Added namespace filter: {name_spaces}")

        # Add the similarity score as a field named "score"
        pipeline.append({"$addFields": {"score": {"$meta": "vectorSearchScore"}}})

        # Filter documents with score >= THRESHOLD
        pipeline.append({"$match": {"score": {"$gte": threshold}}})

        pipeline.append(
            {
                "$group": {
                    "_id": "$text_id",
                    "doc": {"$first": "$$ROOT"},  # keep full document
                }
            }
        )

        # Replace root with the grouped document
        pipeline.append({"$replaceRoot": {"newRoot": "$doc"}})

        # Limit to k results
        pipeline.append({"$limit": k})

        # Optional: project only desired fields
        pipeline.append(
            {
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "score": 1,
                    "sanity_data": 1,
                    "text_id": 1,
                }
            }
        )

        logger.debug(f"[RETRIEVE] Aggregation pipeline: {pipeline}")

        # Retry logic with exponential backoff
        last_exception = None
        current_delay = retry_delay

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                logger.info(
                    f"[RETRIEVE] Attempt {attempt + 1}/{max_retries + 1} in collection='{self.collection.name}', "
                    f"index='{self.index}', timeout_ms={timeout_ms}"
                )
                
                cursor = self.collection.aggregate(pipeline, maxTimeMS=timeout_ms)
                results = await cursor.to_list(length=k)
                
                logger.info(
                    f"[RETRIEVE] Query completed successfully on attempt {attempt + 1}, "
                    f"raw_results_count={len(results)}, collection='{self.collection.name}', "
                    f"index='{self.index}'"
                )
                
                # If we get here, the query was successful
                break
                
            except ExecutionTimeout as e:
                last_exception = e
                logger.warning(
                    f"[RETRIEVE] ExecutionTimeout on attempt {attempt + 1}/{max_retries + 1} "
                    f"in collection='{self.collection.name}', index='{self.index}', "
                    f"maxTimeMS={timeout_ms}, error: {e}"
                )
                
                if attempt < max_retries:
                    logger.info(
                        f"[RETRIEVE] Retrying in {current_delay:.2f} seconds... "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_multiplier
                else:
                    logger.error(
                        f"[RETRIEVE ERROR] All {max_retries + 1} attempts failed with ExecutionTimeout "
                        f"in collection='{self.collection.name}', index='{self.index}', "
                        f"maxTimeMS={timeout_ms}"
                    )
                    raise RetrievalTimeoutException(
                        f"Failed to retrieve documents after {max_retries + 1} attempts. "
                        f"Request timed out: {e}"
                    )
                    
            except OperationFailure as e:
                # OperationFailure is typically not retryable (e.g., invalid query, permissions)
                logger.error(
                    f"[RETRIEVE ERROR] OperationFailure in collection='{self.collection.name}', "
                    f"index='{self.index}', error: {e}"
                )
                raise RetrievalException("Mongo failed to retrieve documents: {}".format(e))
                
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[RETRIEVE] Unexpected error on attempt {attempt + 1}/{max_retries + 1} "
                    f"in collection='{self.collection.name}', index='{self.index}', error: {str(e)}"
                )
                
                if attempt < max_retries:
                    logger.info(
                        f"[RETRIEVE] Retrying in {current_delay:.2f} seconds... "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_multiplier
                else:
                    logger.error(
                        f"[RETRIEVE ERROR] All {max_retries + 1} attempts failed with unexpected error "
                        f"in collection='{self.collection.name}', index='{self.index}', "
                        f"error: {str(e)}", exc_info=True
                    )
                    raise DataBaseException(f"Failed to retrieve documents after {max_retries + 1} attempts: {e}")

        # Process results
        documents: List[DocumentModel] = []
        seen_text_ids = set()  # Additional safety check in Python

        for result in results:
            text_id = result.get("text_id")

            # Skip if we've already seen this text_id (extra safety)
            if text_id in seen_text_ids:
                logger.debug(f"[RETRIEVE] Skipping duplicate text_id: {text_id}")
                continue
            seen_text_ids.add(text_id)

            metadata = Metadata(**result["metadata"])

            sanity_data = SanityData(**result["sanity_data"])

            document = DocumentModel(
                _id=str(result.get("_id")),  # Fixed: changed *id to _id
                text=result.get("text", ""),
                metadata=metadata,
                sanity_data=sanity_data,
                score=float(result["score"]),
            )
            documents.append(document)

        if not documents:
            logger.warning(
                f"[RETRIEVE] No documents found above threshold={threshold}, "
                f"collection='{self.collection.name}', index='{self.index}', "
                f"vector_path='{self.vector_path}', name_spaces={name_spaces}, "
                f"raw_results_before_dedup={len(results)}"
            )
            raise NoDocumentFoundException

        logger.info(
            f"[RETRIEVE] Successfully retrieved {len(documents)} unique documents, "
            f"collection='{self.collection.name}', index='{self.index}', "
            f"score_range=[{documents[-1].score:.4f}, {documents[0].score:.4f}]"
        )
        return documents

    async def get_all_unique_transcript_ids(self) -> list[TranscriptData]:

        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "unique_ids": {
                        "$addToSet": {
                            "transcript_id": "$sanity_data.id",
                            "transcript_hash": "$sanity_data.hash",
                        }
                    },
                }
            },
            {"$project": {"_id": 0, "unique_ids": 1}},
        ]

        cursor = self.collection.aggregate(pipeline)
        result = await cursor.to_list()
        raw_data = result[0].get("unique_ids") if result else []
        tmp = [TranscriptData(**item) for item in raw_data]
        return tmp

    async def delete_document(self, transcript_id: str) -> bool:
        result = await self.collection.delete_many({"sanity_data.id": transcript_id})
        return result.deleted_count > 0


class MongoMetricsConnection(MetricsConnection):
    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    async def log(self, metric_type: str, data: Dict[str, Any]):
        doc = {"type": metric_type, "timestamp": datetime.utcnow(), **data}
        await self.collection.insert_one(doc)


class MongoExceptionsLogger(ExceptionsLogger):
    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    async def log(self, exception_code: str | None, data: Dict[str, Any]):
        if not exception_code:
            exception_code = "Base App Exception"
        await self.collection.insert_one(
            {"type": exception_code, "timestamp": datetime.utcnow(), **data}
        )
