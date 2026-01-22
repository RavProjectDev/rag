import logging
import os
import asyncio
import certifi
import httpx
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Set, List

from rag.app.db.mongodb_connection import MongoEmbeddingStore
from rag.app.db.pinecone_connection import PineconeEmbeddingStore
from rag.app.db.connections import EmbeddingConnection
from rag.app.models.data import SanityData
from rag.app.schemas.data import (
    EmbeddingConfiguration,
    ChunkingStrategy,
    Chunk,
    VectorEmbedding,
    Embedding,
)
from rag.app.services.data_upload_service import (
    fetch_transcript,
    process_transcript_contents,
)
from rag.app.services.embedding import generate_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MANIFEST_URL = "https://the-rav-project.vercel.app/api/manifest"


async def post_data(payload=None):
    """Fetch manifest data from the API."""
    if payload is None:
        payload = {}
    async with httpx.AsyncClient() as client:
        response = await client.post(MANIFEST_URL, json=payload)
        if response.status_code != 200:
            return None
        return response.json()


async def get_existing_chunk_hashes_by_slug(
    chunks_collection,
    sanity_slug: str,
    embedding_model: str,
    chunking_strategy: str,
) -> Set[str]:
    """
    Get all existing chunk hashes for a specific document and configuration.
    
    Filters by:
    - sanity_slug: Document identifier
    - embedding_model: Embedding model used
    - chunking_strategy: Chunking strategy used
    
    This ensures each (embedding_model, chunking_strategy) combination is independent.
    
    Args:
        chunks_collection: MongoDB chunks collection
        sanity_slug: The sanity_slug to filter by
        embedding_model: Embedding model to filter by (e.g., "openai", "gemini-embedding-001")
        chunking_strategy: Chunking strategy to filter by (e.g., "fixed_size", "divided")
    
    Returns:
        Set of SHA-256 hashes for this document and configuration
    """
    try:
        cursor = chunks_collection.find(
            {
                "sanity_slug": sanity_slug,
                "embedding_model": embedding_model,
                "chunking_strategy": chunking_strategy,
            },
            {"_id": 1}
        )
        existing_hashes = set()
        async for doc in cursor:
            existing_hashes.add(doc["_id"])
        logger.info(
            f"[SYNC] Found {len(existing_hashes)} existing chunk hashes for "
            f"slug='{sanity_slug}', model='{embedding_model}', strategy='{chunking_strategy}'"
        )
        return existing_hashes
    except Exception as e:
        logger.error(
            f"[SYNC] Failed to get existing chunk hashes for "
            f"slug='{sanity_slug}', model='{embedding_model}', strategy='{chunking_strategy}': {e}"
        )
        return set()


async def check_hash_in_pinecone(connection: PineconeEmbeddingStore, chunk_hash: str) -> bool:
    """
    Check if a chunk hash exists in Pinecone metadata.
    
    Args:
        connection: Pinecone connection
        chunk_hash: SHA-256 hash to check
    
    Returns:
        True if hash exists, False otherwise
    """
    try:
        # Query Pinecone by metadata filter
        result = await asyncio.to_thread(
            connection.index.query,
            vector=[0.0] * 3072,  # Dummy vector for metadata-only query
            top_k=1,
            include_metadata=True,
            namespace=connection.namespace,
            filter={"text_hash": {"$eq": chunk_hash}}
        )
        return len(result.get("matches", [])) > 0
    except Exception as e:
        logger.debug(f"[SYNC] Error checking hash in Pinecone: {e}")
        return False


async def process_and_sync_chunks(
    chunks: List[Chunk],
    sanity_data: SanityData,
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
    existing_hashes_for_doc: Set[str],
) -> tuple[int, Set[str]]:
    """
    Process chunks and sync only new ones for a specific document.
    
    Args:
        chunks: List of chunks to process
        sanity_data: Document metadata
        connection: Database connection
        embedding_configuration: Embedding model configuration
        existing_hashes_for_doc: Set of existing chunk hashes for this document
    
    Returns:
        Tuple of (number of chunks inserted, set of new hashes for this document)
    """
    inserted_count = 0
    new_hashes_for_doc = set()
    
    for idx, chunk in enumerate(chunks, 1):
        chunk_hash = chunk.text_hash
        new_hashes_for_doc.add(chunk_hash)
        
        # Check if chunk already exists for this document
        if chunk_hash in existing_hashes_for_doc:
            logger.debug(f"[SYNC] Chunk {idx}/{len(chunks)} already exists (hash: {chunk_hash[:16]}...)")
            continue
        
        logger.info(f"[SYNC] Processing new chunk {idx}/{len(chunks)} (hash: {chunk_hash[:16]}...)")
        
        try:
            # Generate embedding
            data: Embedding = await generate_embedding(
                text=chunk.text_to_embed,
                configuration=embedding_configuration,
                task_type="RETRIEVAL_DOCUMENT",
            )
            
            # Create vector embedding
            embedding = VectorEmbedding(
                vector=data.vector,
                dimension=len(data.vector),
                metadata=chunk,
                sanity_data=sanity_data,
                embedding_model=embedding_configuration.value,
                chunking_strategy=chunking_strategy.value,
            )
            
            # Insert into database
            await connection.insert([embedding])
            inserted_count += 1
            
            if inserted_count % 10 == 0:
                logger.info(f"[SYNC] Progress: {inserted_count} new chunks inserted")
                
        except Exception as e:
            logger.error(f"[SYNC] Failed to process chunk {idx}: {e}", exc_info=True)
            continue
    
    return inserted_count, new_hashes_for_doc


async def remove_redundant_chunks_for_document(
    chunks_collection,
    connection: EmbeddingConnection,
    sanity_slug: str,
    embedding_model: str,
    chunking_strategy: str,
    existing_hashes_for_doc: Set[str],
    new_hashes_for_doc: Set[str],
):
    """
    Remove chunks for a specific document and configuration that exist in DB but not in the new version.
    
    Only removes chunks matching the specific (sanity_slug, embedding_model, chunking_strategy) combination.
    This ensures different configurations can coexist for the same document.
    
    Args:
        chunks_collection: MongoDB chunks collection
        connection: Database connection for deleting from vector store
        sanity_slug: Document slug to filter by
        embedding_model: Embedding model to filter by
        chunking_strategy: Chunking strategy to filter by
        existing_hashes_for_doc: Set of hashes currently in DB for this document+config
        new_hashes_for_doc: Set of hashes from new version of this document+config
    """
    redundant_hashes = existing_hashes_for_doc - new_hashes_for_doc
    
    if not redundant_hashes:
        logger.debug(
            f"[SYNC] No redundant chunks to remove for "
            f"slug='{sanity_slug}', model='{embedding_model}', strategy='{chunking_strategy}'"
        )
        return
    
    logger.info(
        f"[SYNC] Found {len(redundant_hashes)} redundant chunks for "
        f"slug='{sanity_slug}', model='{embedding_model}', strategy='{chunking_strategy}'"
    )
    
    try:
        # Remove from chunks collection
        result = await chunks_collection.delete_many({"_id": {"$in": list(redundant_hashes)}})
        logger.info(f"[SYNC] Removed {result.deleted_count} chunks from chunks collection for '{sanity_slug}'")
        
        # For Pinecone, delete by metadata filter
        if isinstance(connection, PineconeEmbeddingStore):
            logger.info(f"[SYNC] Removing redundant chunks from Pinecone for '{sanity_slug}'...")
            for chunk_hash in redundant_hashes:
                try:
                    await asyncio.to_thread(
                        connection.index.delete,
                        filter={"text_hash": {"$eq": chunk_hash}},
                        namespace=connection.namespace
                    )
                except Exception as e:
                    logger.warning(f"[SYNC] Failed to delete chunk {chunk_hash[:16]}... from Pinecone: {e}")
            logger.info(f"[SYNC] Finished removing {len(redundant_hashes)} redundant chunks from Pinecone")
        
        # For MongoDB, delete from vector collection
        elif isinstance(connection, MongoEmbeddingStore):
            logger.info(f"[SYNC] Removing redundant chunks from MongoDB for '{sanity_slug}'...")
            result = await connection.collection.delete_many(
                {"metadata.text_hash": {"$in": list(redundant_hashes)}}
            )
            logger.info(f"[SYNC] Removed {result.deleted_count} documents from vector collection")
            
    except Exception as e:
        logger.error(f"[SYNC] Failed to remove redundant chunks for '{sanity_slug}': {e}", exc_info=True)


async def sync_document(
    sanity_data: SanityData,
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
    chunks_collection,
) -> dict:
    """
    Sync a single document using per-document hash comparison.
    
    Flow:
    1. Get existing hashes for this document (by sanity_slug)
    2. Fetch and chunk the new document
    3. Compare hashes: existing vs new
    4. Insert only new chunks
    5. Remove redundant chunks for this document
    
    Returns:
        Dict with sync statistics for this document
    """
    # Get existing chunk hashes for this document AND configuration
    existing_hashes_for_doc = await get_existing_chunk_hashes_by_slug(
        chunks_collection,
        sanity_data.slug,
        embedding_configuration.value,
        chunking_strategy.value,
    )
    
    # Fetch and chunk the document
    logger.info(f"[SYNC] Fetching transcript for '{sanity_data.title}'...")
    transcript_content = await fetch_transcript(str(sanity_data.transcriptURL))
    
    logger.info(f"[SYNC] Chunking document with strategy: {chunking_strategy.value}")
    chunks = process_transcript_contents(
        sanity_data.title,
        transcript_content,
        chunking_strategy
    )
    logger.info(f"[SYNC] Created {len(chunks)} chunks from '{sanity_data.title}'")
    
    # Process and sync chunks for this document
    inserted_count, new_hashes_for_doc = await process_and_sync_chunks(
        chunks,
        sanity_data,
        connection,
        embedding_configuration,
        chunking_strategy,
        existing_hashes_for_doc,
    )
    
    # Log results
    if inserted_count > 0:
        logger.info(f"[SYNC] Inserted {inserted_count}/{len(chunks)} new chunks from '{sanity_data.title}'")
    else:
        logger.info(f"[SYNC] All chunks already exist for '{sanity_data.title}'")
    
    # Remove redundant chunks for this document and configuration
    await remove_redundant_chunks_for_document(
        chunks_collection,
        connection,
        sanity_data.slug,
        embedding_configuration.value,
        chunking_strategy.value,
        existing_hashes_for_doc,
        new_hashes_for_doc
    )
    
    return {
        "title": sanity_data.title,
        "slug": sanity_data.slug,
        "total_chunks": len(chunks),
        "new_chunks": inserted_count,
        "reused_chunks": len(chunks) - inserted_count,
        "removed_chunks": len(existing_hashes_for_doc - new_hashes_for_doc),
    }


async def run(
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
    chunks_collection=None,
):
    """
    Main sync function using per-document hash-based comparison.
    
    Flow:
    1. Fetch all documents from manifest
    2. For each document:
       a. Get existing hashes for this document (by sanity_slug)
       b. Chunk the document
       c. Compare hashes (existing vs new for this doc)
       d. Insert only new chunks
       e. Remove redundant chunks for this document
    3. Provide summary report
    """
    logger.info("="*60)
    logger.info("SYNC SERVICE - Hash-based per-document synchronization")
    logger.info("="*60)
    
    # Fetch manifest
    logger.info("[SYNC] Fetching manifest data...")
    manifest = await post_data()
    if manifest is None:
        logger.error("[SYNC] Failed to fetch manifest.")
        return
    logger.info(f"[SYNC] Fetched {len(manifest)} documents from manifest")
    
    # Validate chunks collection
    if chunks_collection is None:
        logger.error("[SYNC] No chunks collection provided, cannot perform hash-based sync")
        return
    
    # Track overall statistics
    total_documents = 0
    total_chunks = 0
    total_inserted = 0
    total_reused = 0
    total_removed = 0
    
    # Process each document independently
    for idx, (doc_id, content) in enumerate(manifest.items(), 1):
        logger.info(f"\n[SYNC] {'='*50}")
        logger.info(f"[SYNC] Document {idx}/{len(manifest)}: {content.get('title', doc_id)}")
        logger.info(f"[SYNC] {'='*50}")
        
        try:
            # Create SanityData object
            sanity_data = SanityData(id=doc_id, **content)
            
            # Sync this document
            stats = await sync_document(
                sanity_data,
                connection,
                embedding_configuration,
                chunking_strategy,
                chunks_collection
            )
            
            # Update totals
            total_documents += 1
            total_chunks += stats["total_chunks"]
            total_inserted += stats["new_chunks"]
            total_reused += stats["reused_chunks"]
            total_removed += stats["removed_chunks"]
            
            logger.info(f"[SYNC] Document '{stats['title']}' complete:")
            logger.info(f"[SYNC]   - Total chunks: {stats['total_chunks']}")
            logger.info(f"[SYNC]   - New: {stats['new_chunks']}")
            logger.info(f"[SYNC]   - Reused: {stats['reused_chunks']}")
            logger.info(f"[SYNC]   - Removed: {stats['removed_chunks']}")
                
        except Exception as e:
            logger.error(f"[SYNC] Failed to sync document {doc_id}: {e}", exc_info=True)
            continue
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("SYNC COMPLETE - Summary")
    logger.info("="*60)
    logger.info(f"Documents processed: {total_documents}/{len(manifest)}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"New chunks inserted: {total_inserted}")
    logger.info(f"Existing chunks reused: {total_reused}")
    logger.info(f"Redundant chunks removed: {total_removed}")
    if total_chunks > 0:
        efficiency = (total_reused / total_chunks) * 100
        logger.info(f"Efficiency: {efficiency:.1f}% chunks reused (avoided re-embedding)")
    logger.info("="*60)


if __name__ == "__main__":

    async def main():
        load_dotenv()
        mongo_uri = os.getenv("MONGODB_URI")
        client = AsyncIOMotorClient(
            mongo_uri, tlsCAFile=certifi.where(), maxPoolSize=50
        )
        mongodb_db_name = "rav_dev"
        db = client[mongodb_db_name]
        vector_embedding_collection_name = "gemini_embeddings_v3"
        vector_embedding_collection = db[vector_embedding_collection_name]
        chunks_collection = db["chunks"]
        collection_index = os.getenv("COLLECTION_INDEX")
        vector_path = "vector"
        mongo_connection = MongoEmbeddingStore(
            collection=vector_embedding_collection,
            index=collection_index,
            vector_path=vector_path,
            chunks_collection=chunks_collection,
        )
        embedding_configuration = EmbeddingConfiguration.GEMINI
        chunking_strategy = ChunkingStrategy.FIXED_SIZE
        await run(
            connection=mongo_connection,
            embedding_configuration=embedding_configuration,
            chunking_strategy=chunking_strategy,
            chunks_collection=chunks_collection,
        )

    asyncio.run(main())
