import asyncio
import logging
from contextlib import asynccontextmanager

import certifi
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from rag.app.core.config import get_settings
from rag.app.db.mongodb_connection import MongoEmbeddingStore
from rag.app.db.pinecone_connection import PineconeEmbeddingStore
from rag.app.schemas.data import DataBaseConfiguration
from rag.app.services.sync_service.sync_db import run

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def _mongo_connection(settings):
    client = AsyncIOMotorClient(
        settings.mongodb_uri, tlsCAFile=certifi.where(), maxPoolSize=50
    )
    try:
        db = client[settings.mongodb_db_name]
        
        # Append chunking strategy to collection name if not already included
        collection_name = settings.mongodb_vector_collection
        if settings.chunking_strategy.value not in collection_name:
            collection_name = f"{collection_name}_{settings.chunking_strategy.value}"
        
        logger.info(f"Using MongoDB collection: {collection_name}")
        vector_collection = db[collection_name]
        conn = MongoEmbeddingStore(
            collection=vector_collection,
            index=settings.collection_index,
            vector_path=settings.vector_path,
        )
        yield conn
    finally:
        client.close()


@asynccontextmanager
async def build_connection(settings):
    """
    Context manager that yields an EmbeddingConnection based on configured backend.
    """
    if settings.database_configuration == DataBaseConfiguration.PINECONE:
        required = {
            "pinecone_api_key": settings.pinecone_api_key,
            "pinecone_index_name": settings.pinecone_index_name,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required Pinecone settings: {', '.join(missing)}"
            )
        # Use chunking strategy as namespace if not explicitly configured
        namespace = settings.pinecone_namespace or settings.chunking_strategy.value
        logger.info(f"Using Pinecone namespace: {namespace}")
        
        conn = PineconeEmbeddingStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            environment=settings.pinecone_environment,
            namespace=namespace,
            host=settings.pinecone_host,
        )
        try:
            yield conn
        finally:
            # Pinecone client does not expose an async close; rely on GC.
            pass
    else:
        async with _mongo_connection(settings) as conn:
            yield conn


async def main():
    load_dotenv()
    settings = get_settings()
    logger.info("Starting manifest upload script...")
    logger.info(
        "Backend=%s, embedding_configuration=%s",
        settings.database_configuration.value,
        settings.embedding_configuration.value,
    )

    async with build_connection(settings) as conn:
        await run(conn, settings.embedding_configuration)

    logger.info("Manifest upload script finished.")


if __name__ == "__main__":
    asyncio.run(main())

