from datetime import datetime
from typing import Dict, List, Any

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


class MongoEmbeddingStore(EmbeddingConnection):
    def __init__(self, collection, index: str, vector_path: str):
        self.collection = collection
        self.index = index
        self.vector_path = vector_path

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
        except OperationFailure as e:
            raise InsertException(
                f"Failed to insert documents, Mongo config error: {e}"
            )
        except Exception as e:
            raise DataBaseException(f"Failed to insert documents: {e}")

    async def retrieve(
        self,
        embedded_data: List[float],
        name_spaces: list[str] | None = None,
        k=5,
        threshold: float = 0.85,
    ) -> list[DocumentModel]:
        # Increase the initial limit to account for potential duplicates
        initial_limit = min(k * 3, 1000)  # Get more documents initially

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

        try:
            cursor = self.collection.aggregate(pipeline, maxTimeMS=500)
            results = await cursor.to_list(length=k)
        except ExecutionTimeout as e:
            raise RetrievalTimeoutException(
                f"Failed to retrieve documents. Request timed out: {e}"
            )
        except OperationFailure as e:
            raise RetrievalException("Mongo failed to retrieve documents: {}".format(e))
        except Exception as e:
            raise DataBaseException(f"Failed to retrieve documents: {e}")

        documents: List[DocumentModel] = []
        seen_text_ids = set()  # Additional safety check in Python

        for result in results:
            text_id = result.get("text_id")

            # Skip if we've already seen this text_id (extra safety)
            if text_id in seen_text_ids:
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
            raise NoDocumentFoundException

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
