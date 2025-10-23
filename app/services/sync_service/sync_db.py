import logging
import os
import asyncio
import certifi
import httpx
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from rag.app.db.mongodb_connection import MongoEmbeddingStore
from rag.app.db.connections import EmbeddingConnection
from rag.app.models.data import SanityData
from rag.app.schemas.data import EmbeddingConfiguration, TranscriptData
from rag.app.services.data_upload_service import upload_documents, update_documents

logging.basicConfig(level=logging.INFO)

MANIFEST_URL = "https://the-rav-project.vercel.app/api/manifest"


async def post_data(payload=None):
    if payload is None:
        payload = {}
    async with httpx.AsyncClient() as client:
        response = await client.post(MANIFEST_URL, json=payload)
        if response.status_code != 200:
            return None
        return response.json()


async def run(
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
):
    logging.info("[run] Fetching manifest data...")
    manifest = await post_data()
    if manifest is None:
        logging.error("[run] Failed to fetch manifest.")
        return
    logging.info(f"[run] Fetched {len(manifest)} manifest entries.")

    logging.info("[run] Fetching existing transcript IDs from DB...")
    data: list[TranscriptData] = await connection.get_all_unique_transcript_ids()
    transcript_map = {t.transcript_id: t.transcript_hash for t in data}
    logging.info(f"[run] Found {len(data)} unique transcript IDs in DB.")

    documents_needed_to_be_uploaded: list[SanityData] = []
    documents_needed_to_be_updated: list[SanityData] = []

    logging.info("[run] Comparing manifest to database records...")
    for doc_id, content in manifest.items():
        if doc_id not in transcript_map:
            logging.info(f"[run] New document found: {doc_id}")
            documents_needed_to_be_uploaded.append(SanityData(id=doc_id, **content))
        elif transcript_map[doc_id] != content.get("hash"):
            logging.info(f"[run] Updated hash detected for: {doc_id}")
            documents_needed_to_be_updated.append(SanityData(id=doc_id, **content))

    logging.info(
        f"[run] {len(documents_needed_to_be_uploaded)} documents need to be uploaded."
    )
    logging.info(
        f"[run] {len(documents_needed_to_be_updated)} documents need to be updated."
    )

    if documents_needed_to_be_uploaded:
        logging.info("[run] Uploading new documents...")
        await upload_documents(
            documents_needed_to_be_uploaded, connection, embedding_configuration
        )
        logging.info("[run] Finished uploading new documents.")

    if documents_needed_to_be_updated:
        logging.info("[run] Updating modified documents...")
        await update_documents(
            documents_needed_to_be_updated, connection, embedding_configuration
        )
        logging.info("[run] Finished updating modified documents.")

    logging.info("[run] Processing complete.")


if __name__ == "__main__":

    async def main():
        load_dotenv()
        mongo_uri = os.getenv("MONGODB_URI")
        client = AsyncIOMotorClient(
            mongo_uri, tlsCAFile=certifi.where(), maxPoolSize=50
        )
        mongodb_db_name = "rav_dev"
        db = client[mongodb_db_name]
        vector_embedding_collection_name = "gemini_embeddings_v2"
        vector_embedding_collection = db[vector_embedding_collection_name]
        collection_index = os.getenv("COLLECTION_INDEX")
        vector_path = "vector"
        mongo_connection = MongoEmbeddingStore(
            collection=vector_embedding_collection,
            index=collection_index,
            vector_path=vector_path,
        )
        embedding_configuration = EmbeddingConfiguration.GEMINI
        await run(
            connection=mongo_connection, embedding_configuration=embedding_configuration
        )

    asyncio.run(main())
