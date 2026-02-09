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
from rag.app.services.data_upload_service import upload_documents
from rag.app.models.data import SanityData

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MANIFEST_URL = "https://the-rav-project.vercel.app/api/manifest"


async def fetch_manifest():
    """Fetch all documents from the manifest API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(MANIFEST_URL, json={})
        if response.status_code != 200:
            logger.error(f"Failed to fetch manifest: {response.status_code}")
            return None
        return response.json()


async def upload_from_manifest(
    conn,
    embedding_model: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
):
    """
    Simple upload: fetch all documents from manifest and upload them.
    No smart syncing, no deduplication checks - just upload.
    """
    logger.info("[UPLOAD] Fetching manifest data...")
    manifest = await fetch_manifest()
    if manifest is None:
        logger.error("[UPLOAD] Failed to fetch manifest.")
        return
    
    logger.info(f"[UPLOAD] Fetched {len(manifest)} documents from manifest")
    
    # Convert manifest to SanityData objects
    documents = []
    for doc_id, content in manifest.items():
        try:
            sanity_data = SanityData(id=doc_id, **content)
            documents.append(sanity_data)
        except Exception as e:
            logger.error(f"[UPLOAD] Failed to parse document {doc_id}: {e}")
            continue
    
    logger.info(f"[UPLOAD] Uploading {len(documents)} documents...")
    
    # Use the upload service to process all documents
    await upload_documents(
        documents=documents,
        connection=conn,
        embedding_configuration=embedding_model,
        chunking_strategy=chunking_strategy,
    )
    
    logger.info(f"[UPLOAD] Finished uploading {len(documents)} documents")


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
        yield conn
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
            yield conn
        finally:
            # Pinecone client does not expose an async close; rely on GC.
            mongo_client.close()
    else:
        async with _mongo_connection(settings, chunking_strategy) as conn:
            yield conn


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload documents to vector database with specified embedding model and chunking strategy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings from environment
  python upload_manifest.py

  # Specify embedding model
  python upload_manifest.py --embedding-model text-embedding-3-large

  # Specify both embedding model and chunking strategy
  python upload_manifest.py --embedding-model embed-multilingual-v3.0 --chunking-strategy divided

  # List available options
  python upload_manifest.py --list-options
        """
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        choices=[e.value for e in EmbeddingConfiguration],
        help="Embedding model to use for document processing. If not specified, uses EMBEDDING_CONFIGURATION from environment."
    )
    
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        choices=[c.value for c in ChunkingStrategy],
        help="Chunking strategy to use for document processing. If not specified, uses CHUNKING_STRATEGY from environment."
    )
    
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List all available embedding models and chunking strategies, then exit."
    )
    
    return parser.parse_args()


def list_options():
    """Print available embedding models and chunking strategies."""
    print("\n=== Available Embedding Models ===")
    for model in EmbeddingConfiguration:
        print(f"  - {model.value}")
    
    print("\n=== Available Chunking Strategies ===")
    for strategy in ChunkingStrategy:
        print(f"  - {strategy.value}")
    
    print("\n=== Current Environment Settings ===")
    settings = get_settings()
    print(f"  Database: {settings.database_configuration.value}")
    print(f"  Embedding Model: {settings.embedding_configuration.value}")
    print(f"  Chunking Strategy: {settings.chunking_strategy.value}")
    print()


async def main():
    load_dotenv()
    args = parse_arguments()
    
    # Handle --list-options flag
    if args.list_options:
        list_options()
        return
    
    settings = get_settings()
    
    # CLI args have highest precedence and always override .env
    # Order: CLI args > config defaults (ignore .env for these settings when CLI provided)
    if args.embedding_model:
        embedding_model = EmbeddingConfiguration(args.embedding_model)
        logger.info(f"Using embedding model from CLI (overrides .env): {embedding_model.value}")
    else:
        embedding_model = settings.embedding_configuration
        logger.info(f"Using embedding model from environment: {embedding_model.value}")
    
    if args.chunking_strategy:
        chunking_strategy = ChunkingStrategy(args.chunking_strategy)
        logger.info(f"Using chunking strategy from CLI (overrides .env): {chunking_strategy.value}")
    else:
        chunking_strategy = settings.chunking_strategy
        logger.info(f"Using chunking strategy from environment: {chunking_strategy.value}")
    
    logger.info("="*60)
    logger.info("UPLOAD SCRIPT - Simple manifest upload (no sync)")
    logger.info("="*60)
    logger.info(
        "Configuration: backend=%s, embedding_model=%s, chunking_strategy=%s",
        settings.database_configuration.value,
        embedding_model.value,
        chunking_strategy.value,
    )

    async with build_connection(settings, embedding_model, chunking_strategy) as conn:
        await upload_from_manifest(conn, embedding_model, chunking_strategy)

    logger.info("="*60)
    logger.info("UPLOAD COMPLETE")
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

