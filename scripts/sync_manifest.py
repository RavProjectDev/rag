"""
Sync script that fetches configuration from the API /info endpoint.

This script:
1. Fetches embedding model and chunking strategy from the API's /info endpoint
2. Fetches all documents from the manifest
3. For each document, compares existing chunks (by hash) vs new chunks
4. Only embeds and inserts new chunks (skips existing ones)
5. Removes redundant chunks that exist in DB but not in new version

Configuration is pulled from the API to ensure sync uses the same settings.
If the API endpoint fails or doesn't exist, the script will fail (no defaults).
"""

import argparse
import asyncio
import logging
import sys
from contextlib import asynccontextmanager

import certifi
import httpx
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from rag.app.core.config import get_settings
from rag.app.db.mongodb_connection import MongoEmbeddingStore
from rag.app.db.pinecone_connection import PineconeEmbeddingStore
from rag.app.schemas.data import DataBaseConfiguration, EmbeddingConfiguration, ChunkingStrategy
from rag.app.services.sync_service.sync_db import run as run_sync

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Base API URL - change this for different environments (e.g., stg-api.theravlegacy.org)
BASE_API_URL = "https://api.theravlegacy.org"


async def fetch_api_configuration(api_url: str) -> tuple[EmbeddingConfiguration, ChunkingStrategy]:
    """
    Fetch configuration from the API /info endpoint.
    
    Args:
        api_url: Base API URL (e.g., https://api.theravlegacy.org)
    
    Returns:
        Tuple of (embedding_configuration, chunking_strategy)
    
    Raises:
        SystemExit: If the endpoint doesn't exist or returns an error
    """
    info_endpoint = f"{api_url}/api/v1/info/"
    
    logger.info(f"[CONFIG] Fetching configuration from {info_endpoint}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(info_endpoint)
            
            if response.status_code != 200:
                logger.error(
                    f"[CONFIG] Failed to fetch configuration from API. "
                    f"Status: {response.status_code}, Response: {response.text}"
                )
                logger.error("[CONFIG] Cannot proceed without valid configuration from API.")
                sys.exit(1)
            
            config = response.json()
            
            # Validate required fields
            if "embedding_model" not in config or "chunking_strategy" not in config:
                logger.error(
                    f"[CONFIG] API response missing required fields. "
                    f"Expected 'embedding_model' and 'chunking_strategy'. Got: {config}"
                )
                sys.exit(1)
            
            embedding_model = config["embedding_model"]
            chunking_strategy = config["chunking_strategy"]
            
            logger.info(f"[CONFIG] ✓ Embedding model: {embedding_model}")
            logger.info(f"[CONFIG] ✓ Chunking strategy: {chunking_strategy}")
            logger.info(f"[CONFIG] ✓ Database backend: {config.get('database_backend', 'unknown')}")
            logger.info(f"[CONFIG] ✓ Environment: {config.get('environment', 'unknown')}")
            
            # Convert to enum values
            try:
                embedding_config = EmbeddingConfiguration(embedding_model)
                chunking_config = ChunkingStrategy(chunking_strategy)
            except ValueError as e:
                logger.error(
                    f"[CONFIG] Invalid configuration values from API: {e}"
                )
                sys.exit(1)
            
            return embedding_config, chunking_config
            
    except httpx.ConnectError as e:
        logger.error(f"[CONFIG] Cannot connect to API at {info_endpoint}: {e}")
        logger.error("[CONFIG] Please check the BASE_API_URL and ensure the API is running.")
        sys.exit(1)
    except httpx.TimeoutException as e:
        logger.error(f"[CONFIG] Timeout fetching configuration from API: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[CONFIG] Unexpected error fetching configuration: {e}", exc_info=True)
        sys.exit(1)


@asynccontextmanager
async def _mongo_connection(settings, chunking_strategy: ChunkingStrategy):
    client = AsyncIOMotorClient(
        settings.mongodb_uri, tlsCAFile=certifi.where(), maxPoolSize=50
    )
    try:
        db = client[settings.mongodb_db_name]
        
        # Append chunking strategy to collection name if not already included
        collection_name = settings.mongodb_vector_collection
        if chunking_strategy.value not in collection_name:
            collection_name = f"{collection_name}_{chunking_strategy.value}"
        
        logger.info(f"Using MongoDB collection: {collection_name}")
        vector_collection = db[collection_name]
        
        # Get chunks collection for tracking chunk metadata
        chunks_collection = db["chunks"]
        logger.info(f"Using MongoDB chunks collection: chunks")
        
        conn = MongoEmbeddingStore(
            collection=vector_collection,
            index=settings.collection_index,
            vector_path=settings.vector_path,
            chunks_collection=chunks_collection,
        )
        yield conn, chunks_collection
    finally:
        client.close()


@asynccontextmanager
async def build_connection(settings, embedding_model: EmbeddingConfiguration, chunking_strategy: ChunkingStrategy):
    """
    Context manager that yields an EmbeddingConnection based on configured backend.
    
    Args:
        settings: Application settings
        embedding_model: Embedding model to use
        chunking_strategy: Chunking strategy to use
        
    Yields:
        Tuple of (connection, chunks_collection)
    """
    if settings.database_configuration == DataBaseConfiguration.PINECONE:
        required = {
            "pinecone_api_key": settings.pinecone_api_key,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required Pinecone settings: {', '.join(missing)}"
            )
        
        # Index name is based on embedding model
        index_name = settings.pinecone_index_name or embedding_model.value
        
        # Namespace is based on chunking strategy
        namespace = settings.pinecone_namespace or chunking_strategy.value
        
        logger.info(
            f"[PINECONE] Using index={index_name} (embedding_model={embedding_model.value}), "
            f"namespace={namespace} (chunking_strategy={chunking_strategy.value})"
        )
        
        # Get MongoDB connection for chunks tracking
        mongo_client = AsyncIOMotorClient(
            settings.mongodb_uri, tlsCAFile=certifi.where(), maxPoolSize=50
        )
        mongo_db = mongo_client[settings.mongodb_db_name]
        chunks_collection = mongo_db["chunks"]
        logger.info(f"[PINECONE] MongoDB chunks collection configured: chunks")
        
        conn = PineconeEmbeddingStore(
            api_key=settings.pinecone_api_key,
            index_name=index_name,
            environment=settings.pinecone_environment,
            namespace=namespace,
            host=settings.pinecone_host,
            chunks_collection=chunks_collection,
        )
        try:
            yield conn, chunks_collection
        finally:
            # Pinecone client does not expose an async close; rely on GC.
            mongo_client.close()
    else:
        async with _mongo_connection(settings, chunking_strategy) as result:
            yield result


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sync documents using configuration from API /info endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync using configuration from API
  python scripts/sync_manifest.py

  # Use a different API URL (e.g., staging)
  # (Edit BASE_API_URL constant at the top of this file)

Note:
  - This script fetches embedding_model and chunking_strategy from the API's /info endpoint
  - If the endpoint fails or doesn't exist, the script will exit with an error
  - No defaults are used; the script requires a valid API response to proceed
        """
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default=BASE_API_URL,
        help=f"Base API URL to fetch configuration from. Default: {BASE_API_URL}"
    )
    
    return parser.parse_args()


async def main():
    load_dotenv()
    args = parse_arguments()
    
    logger.info("="*60)
    logger.info("SYNC SCRIPT - Hash-based synchronization")
    logger.info("="*60)
    
    # Fetch configuration from API
    api_url = args.api_url
    logger.info(f"[CONFIG] Using API URL: {api_url}")
    
    embedding_model, chunking_strategy = await fetch_api_configuration(api_url)
    
    # Get settings for database connection
    settings = get_settings()
    
    logger.info("="*60)
    logger.info("CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Database backend: {settings.database_configuration.value}")
    logger.info(f"Embedding model: {embedding_model.value}")
    logger.info(f"Chunking strategy: {chunking_strategy.value}")
    logger.info("="*60)
    
    # Build connection and run sync
    async with build_connection(settings, embedding_model, chunking_strategy) as (conn, chunks_collection):
        await run_sync(
            connection=conn,
            embedding_configuration=embedding_model,
            chunking_strategy=chunking_strategy,
            chunks_collection=chunks_collection,
        )
    
    logger.info("="*60)
    logger.info("SYNC COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Script failed with error: {e}", exc_info=True)
        sys.exit(1)
