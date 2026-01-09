"""
Unified Configurable Upload Script

This script processes documents with configurable embedding models and chunking strategies.
Configuration is done via environment variables.

Environment Variables:
    EMBEDDING_MODELS: Comma-separated list of embedding models (e.g., "gemini,openai")
                      Options: gemini, openai, cohere
                      Default: all models if not set
    
    CHUNKING_STRATEGIES: Comma-separated list of chunking strategies (e.g., "semantic,fixed-size")
                         Options: fixed-size, semantic, sliding-window
                         Default: all strategies if not set
    
    DRY_RUN: Set to "true" to show what would be processed without actually processing
             Default: false
    
    PINECONE_API_KEY: Required - Your Pinecone API key
    
    PINECONE_CLOUD: Optional - Cloud provider (default: "aws")
    
    PINECONE_REGION: Optional - Region (default: "us-east-1")

Usage Examples:
    # Process all combinations (no env vars needed)
    python -m rag.scripts.upload

    # Specific embedding model
    EMBEDDING_MODELS=gemini python -m rag.scripts.upload

    # Multiple embedding models
    EMBEDDING_MODELS=gemini,openai python -m rag.scripts.upload

    # Specific chunking strategy
    CHUNKING_STRATEGIES=semantic python -m rag.scripts.upload

    # Multiple chunking strategies
    CHUNKING_STRATEGIES=fixed-size,semantic python -m rag.scripts.upload

    # Combine both
    EMBEDDING_MODELS=gemini CHUNKING_STRATEGIES=semantic python -m rag.scripts.upload

    # Dry run
    DRY_RUN=true python -m rag.scripts.upload

Storage:
- Index: Based on embedding model (gemini, openai, cohere)
- Namespace: Based on chunking strategy (fixed-size, semantic, sliding-window)
"""
import logging
import os
import asyncio
import httpx
import json
from itertools import product
from typing import List, Tuple
from datetime import datetime
from dotenv import load_dotenv

from rag.app.db.pinecone_connection import PineconeEmbeddingStore
from rag.app.models.data import SanityData
from rag.app.schemas.data import (
    EmbeddingConfiguration, 
    ChunkingStrategy,
    Chunk,
    VectorEmbedding,
)
from rag.app.services.preprocess.transcripts import preprocess_raw_transcripts
from rag.app.services.embedding import generate_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MANIFEST_URL = "https://theravlegacy.org/api/manifest"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
FAILED_CHUNKS_FILE = "failed_chunks.json"

# Available configurations
EMBEDDING_MODELS = {
    "gemini": EmbeddingConfiguration.GEMINI_RETRIEVAL_DOCUMENT,
    "openai": EmbeddingConfiguration.OPENAI_TEXT_EMBEDDING_3_LARGE,
    "cohere": EmbeddingConfiguration.COHERE_MULTILINGUAL,
}

CHUNKING_STRATEGIES = {
    "fixed-size": ChunkingStrategy.FIXED_SIZE,
    "semantic": ChunkingStrategy.SEMANTIC,
    "sliding-window": ChunkingStrategy.SLIDING_WINDOW,
}

# Embedding dimensions
EMBEDDING_DIMENSIONS = {
    EmbeddingConfiguration.GEMINI_RETRIEVAL_DOCUMENT: 784,
    EmbeddingConfiguration.OPENAI_TEXT_EMBEDDING_3_LARGE: 3072,
    EmbeddingConfiguration.COHERE_MULTILINGUAL: 1024,
}


def get_index_name(embedding_config: EmbeddingConfiguration) -> str:
    """
    Map embedding configuration to Pinecone index name.
    
    Returns:
        Index name: gemini, openai, or cohere
    """
    mapping = {
        EmbeddingConfiguration.GEMINI_RETRIEVAL_DOCUMENT: "gemini",
        EmbeddingConfiguration.OPENAI_TEXT_EMBEDDING_3_LARGE: "openai",
        EmbeddingConfiguration.COHERE_MULTILINGUAL: "cohere",
    }
    return mapping.get(embedding_config, "gemini")


def get_namespace_name(chunking_strategy: ChunkingStrategy) -> str:
    """
    Map chunking strategy to Pinecone namespace name.
    
    Returns:
        Namespace name: fixed-size, semantic, or sliding-window
    """
    mapping = {
        ChunkingStrategy.FIXED_SIZE: "fixed-size",
        ChunkingStrategy.SEMANTIC: "semantic",
        ChunkingStrategy.SLIDING_WINDOW: "sliding-window",
    }
    return mapping.get(chunking_strategy, "fixed-size")


def get_embedding_dimension(embedding_config: EmbeddingConfiguration) -> int:
    """
    Get the dimension for each embedding configuration.
    
    Returns:
        Dimension size for the embedding model
    """
    return EMBEDDING_DIMENSIONS.get(embedding_config, 784)


async def fetch_manifest() -> dict:
    """Fetch manifest data from the API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(MANIFEST_URL, json={})
        if response.status_code != 200:
            return None
        return response.json()


async def fetch_transcript(transcript_url: str) -> str:
    """Fetch transcript content from URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(transcript_url)
    if not response.content:
        raise Exception(f"Failed to fetch transcript from {transcript_url}")
    return response.content.decode("utf-8")


async def embed_chunk_with_retry(
    chunk: Chunk,
    chunk_index: int,
    total_chunks: int,
    doc: SanityData,
    embedding_config: EmbeddingConfiguration,
    connection: PineconeEmbeddingStore,
    namespace: str,
    max_retries: int = MAX_RETRIES
) -> bool:
    """
    Attempt to embed and insert a chunk with retry logic.
    
    Args:
        chunk: Chunk to embed
        chunk_index: Index of the chunk (0-based)
        total_chunks: Total number of chunks
        doc: Document being processed
        embedding_config: Embedding configuration to use
        connection: Pinecone connection to insert into
        namespace: Namespace to insert into
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if successful, False if all retries failed
    """
    for attempt in range(max_retries):
        try:
            # Generate embedding
            embedding = await generate_embedding(
                text=chunk.text_to_embed,
                configuration=embedding_config,
                task_type="RETRIEVAL_DOCUMENT",
            )
            
            # Create VectorEmbedding
            vector_embedding = VectorEmbedding(
                vector=embedding.vector,
                dimension=len(embedding.vector),
                metadata=chunk,
                sanity_data=doc,
            )
            
            # Insert immediately
            await connection.insert([vector_embedding], namespace=namespace)
            
            if attempt > 0:
                logger.info(
                    f"[RETRY SUCCESS] Chunk {chunk_index + 1}/{total_chunks} "
                    f"succeeded on attempt {attempt + 1}"
                )
            
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"[RETRY] Chunk {chunk_index + 1}/{total_chunks} failed "
                    f"(attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait_time}s... Error: {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"[FAILED] Chunk {chunk_index + 1}/{total_chunks} failed "
                    f"after {max_retries} attempts: {e}",
                    exc_info=True
                )
                return False
    
    return False


async def process_document_with_config(
    doc: SanityData,
    embedding_config: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
    connection: PineconeEmbeddingStore,
    namespace: str,
) -> Tuple[int, List[Tuple[int, Chunk]]]:
    """
    Process a single document with specific embedding and chunking configuration.
    Inserts embeddings immediately after generation to avoid large batch upserts.
    
    Args:
        doc: Document to process
        embedding_config: Embedding configuration to use
        chunking_strategy: Chunking strategy to use
        connection: Pinecone connection to insert into
        namespace: Namespace to insert into
        
    Returns:
        Tuple of (inserted_count, failed_chunks_list)
        where failed_chunks_list contains (index, chunk) tuples
    """
    logger.info(
        f"[PROCESS] Processing {doc.title} with "
        f"embedding={embedding_config.value}, chunking={chunking_strategy.value}"
    )
    
    # Fetch transcript content
    contents = await fetch_transcript(str(doc.transcriptURL))
    
    # Preprocess with specific chunking strategy
    raw_transcripts = [(doc.title, contents)]
    chunks: list[Chunk] = await preprocess_raw_transcripts(
        raw_transcripts=raw_transcripts,
        chunking_strategy=chunking_strategy,
        embedding_configuration=embedding_config,
    )
    
    logger.info(f"[PROCESS] Created {len(chunks)} chunks with {chunking_strategy.value}")
    
    # Track successful and failed chunks
    inserted_count = 0
    failed_chunks = []
    
    # Process all chunks with retry logic
    for i, chunk in enumerate(chunks):
        if (i + 1) % 10 == 0:
            logger.info(f"[PROCESS] Embedding chunk {i + 1}/{len(chunks)}")
        
        success = await embed_chunk_with_retry(
            chunk=chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            doc=doc,
            embedding_config=embedding_config,
            connection=connection,
            namespace=namespace,
        )
        
        if success:
            inserted_count += 1
        else:
            failed_chunks.append((i, chunk))
    
    logger.info(
        f"[PROCESS] Generated and inserted {inserted_count}/{len(chunks)} embeddings "
        f"with {embedding_config.value}"
    )
    
    if failed_chunks:
        logger.error(
            f"[PROCESS] {len(failed_chunks)} chunks failed after all retries for {doc.title}"
        )
    
    return inserted_count, failed_chunks


def save_failed_chunks(
    failed_data: List[dict],
    filename: str = FAILED_CHUNKS_FILE
):
    """
    Save failed chunks to a JSON file for manual recovery.
    
    Args:
        failed_data: List of dictionaries containing failed chunk information
        filename: Path to the JSON file to save
    """
    try:
        # Load existing failed chunks if file exists
        existing_data = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        
        # Append new failures
        existing_data.extend(failed_data)
        
        # Save back to file
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"[RECOVERY] Saved {len(failed_data)} failed chunks to {filename}")
    except Exception as e:
        logger.error(f"[RECOVERY] Failed to save failed chunks: {e}")


async def run_upload(
    embedding_configs: List[EmbeddingConfiguration],
    chunking_strategies: List[ChunkingStrategy],
    pinecone_api_key: str,
    pinecone_cloud: str = "aws",
    pinecone_region: str = "us-east-1",
    dry_run: bool = False,
):
    """
    Run upload for specified embedding configs and chunking strategies.
    
    Combinations are processed sequentially to ensure consistency.
    
    For each combination:
    1. Use Pinecone index based on embedding model (gemini, openai, cohere)
    2. Use namespace based on chunking strategy (fixed-size, semantic, sliding-window)
    3. Process all documents with that configuration
    4. Insert into the appropriate index/namespace
    
    Args:
        embedding_configs: List of embedding configurations to process
        chunking_strategies: List of chunking strategies to process
        pinecone_api_key: Pinecone API key
        pinecone_cloud: Pinecone cloud provider
        pinecone_region: Pinecone region
        dry_run: If True, only show what would be processed without actually processing
    """
    # Create Cartesian product of configurations
    combinations = list(product(embedding_configs, chunking_strategies))
    
    logger.info(f"[UPLOAD] Processing {len(combinations)} combination(s) sequentially")
    logger.info(f"[UPLOAD] Embedding configs: {[c.value for c in embedding_configs]}")
    logger.info(f"[UPLOAD] Chunking strategies: {[s.value for s in chunking_strategies]}")
    
    if dry_run:
        logger.info("\n[DRY RUN] Would process the following combinations:")
        for idx, (embedding_config, chunking_strategy) in enumerate(combinations, 1):
            index_name = get_index_name(embedding_config)
            namespace_name = get_namespace_name(chunking_strategy)
            dimension = get_embedding_dimension(embedding_config)
            logger.info(
                f"  {idx}. Embedding: {embedding_config.value} | "
                f"Chunking: {chunking_strategy.value} | "
                f"Index: {index_name} | "
                f"Namespace: {namespace_name} | "
                f"Dimension: {dimension}"
            )
        logger.info("[DRY RUN] Exiting without processing")
        return
    
    # Fetch manifest
    logger.info("[UPLOAD] Fetching manifest...")
    manifest = await fetch_manifest()
    if not manifest:
        logger.error("[UPLOAD] Failed to fetch manifest")
        return
    
    documents = [SanityData(id=doc_id, **content) for doc_id, content in manifest.items()]
    logger.info(f"[UPLOAD] Found {len(documents)} documents in manifest")
    
    # Track all failed chunks across all documents
    all_failed_chunks = []
    
    # Process each combination sequentially
    for idx, (embedding_config, chunking_strategy) in enumerate(combinations, 1):
        index_name = get_index_name(embedding_config)
        namespace_name = get_namespace_name(chunking_strategy)
        dimension = get_embedding_dimension(embedding_config)
        
        logger.info(
            f"\n{'='*80}\n"
            f"[UPLOAD] Processing combination {idx}/{len(combinations)}\n"
            f"[UPLOAD] Embedding: {embedding_config.value}\n"
            f"[UPLOAD] Chunking: {chunking_strategy.value}\n"
            f"[UPLOAD] Index: {index_name}\n"
            f"[UPLOAD] Namespace: {namespace_name}\n"
            f"[UPLOAD] Dimension: {dimension}\n"
            f"{'='*80}"
        )
        
        # Create Pinecone connection for this index
        connection = PineconeEmbeddingStore(
            api_key=pinecone_api_key,
            index_name=index_name,
            dimension=dimension,
            metric="cosine",
            cloud=pinecone_cloud,
            region=pinecone_region,
        )
        
        # Process each document
        # Note: Duplicate checking is handled in the insert method via fetch
        # Embeddings are inserted immediately after generation to avoid large batch upserts
        for doc_idx, doc in enumerate(documents, 1):
            logger.info(
                f"[UPLOAD] [{doc_idx}/{len(documents)}] Processing {doc.title}"
            )
            
            try:
                # Process document and collect failed chunks
                inserted_count, failed_chunks = await process_document_with_config(
                    doc=doc,
                    embedding_config=embedding_config,
                    chunking_strategy=chunking_strategy,
                    connection=connection,
                    namespace=namespace_name,
                )
                
                # Log failures and save for later recovery
                if failed_chunks:
                    failed_data = {
                        "timestamp": datetime.now().isoformat(),
                        "document_title": doc.title,
                        "document_id": doc.id,
                        "embedding_config": embedding_config.value,
                        "chunking_strategy": chunking_strategy.value,
                        "index_name": index_name,
                        "namespace": namespace_name,
                        "failed_chunk_indices": [i for i, _ in failed_chunks],
                        "total_chunks": inserted_count + len(failed_chunks),
                    }
                    all_failed_chunks.append(failed_data)
                
                logger.info(
                    f"[UPLOAD] [{doc_idx}/{len(documents)}] Successfully processed {doc.title} - "
                    f"inserted {inserted_count} embeddings into index '{index_name}' "
                    f"namespace '{namespace_name}'"
                )
                
            except Exception as e:
                logger.error(
                    f"[UPLOAD] [{doc_idx}/{len(documents)}] Failed to process {doc.title}: {e}",
                    exc_info=True
                )
                # Don't continue - log this as a completely failed document
                all_failed_chunks.append({
                    "timestamp": datetime.now().isoformat(),
                    "document_title": doc.title,
                    "document_id": doc.id,
                    "embedding_config": embedding_config.value,
                    "chunking_strategy": chunking_strategy.value,
                    "index_name": index_name,
                    "namespace": namespace_name,
                    "error": str(e),
                    "status": "DOCUMENT_FAILED_COMPLETELY"
                })
                continue
        
        logger.info(
            f"[UPLOAD] Completed combination {idx}/{len(combinations)}: "
            f"index '{index_name}', namespace '{namespace_name}'"
        )
    
    # Save all failed chunks to file
    if all_failed_chunks:
        save_failed_chunks(all_failed_chunks)
        logger.error(
            f"\n{'='*80}\n"
            f"[RECOVERY] {len(all_failed_chunks)} document(s) had failures\n"
            f"[RECOVERY] Failed chunks saved to '{FAILED_CHUNKS_FILE}'\n"
            f"[RECOVERY] Review and re-run failed chunks manually\n"
            f"{'='*80}"
        )
    
    logger.info(f"\n[UPLOAD] All {len(combinations)} combination(s) completed!")


def parse_embedding_configs_from_env() -> List[EmbeddingConfiguration] | None:
    """
    Parse embedding configurations from environment variable.
    
    Returns:
        List of embedding configurations, or None if not set
    """
    env_value = os.getenv("EMBEDDING_MODELS")
    if not env_value:
        return None
    
    models = [m.strip().lower() for m in env_value.split(",")]
    configs = []
    for model in models:
        if model in EMBEDDING_MODELS:
            configs.append(EMBEDDING_MODELS[model])
        else:
            logger.warning(f"Unknown embedding model '{model}' in EMBEDDING_MODELS env var. Skipping.")
    
    return configs if configs else None


def parse_chunking_strategies_from_env() -> List[ChunkingStrategy] | None:
    """
    Parse chunking strategies from environment variable.
    
    Returns:
        List of chunking strategies, or None if not set
    """
    env_value = os.getenv("CHUNKING_STRATEGIES")
    if not env_value:
        return None
    
    strategies = [s.strip().lower() for s in env_value.split(",")]
    configs = []
    for strategy in strategies:
        if strategy in CHUNKING_STRATEGIES:
            configs.append(CHUNKING_STRATEGIES[strategy])
        else:
            logger.warning(f"Unknown chunking strategy '{strategy}' in CHUNKING_STRATEGIES env var. Skipping.")
    
    return configs if configs else None


async def main():
    """Main entry point."""
    load_dotenv()
    
    # Determine which embedding configs to use from environment variable
    env_configs = parse_embedding_configs_from_env()
    if env_configs:
        embedding_configs = env_configs
        logger.info("[CONFIG] Using EMBEDDING_MODELS environment variable")
    else:
        # Default: all models
        embedding_configs = list(EMBEDDING_MODELS.values())
        logger.info("[CONFIG] Using default embedding models (all)")
    
    # Determine which chunking strategies to use from environment variable
    env_strategies = parse_chunking_strategies_from_env()
    if env_strategies:
        chunking_strategies = env_strategies
        logger.info("[CONFIG] Using CHUNKING_STRATEGIES environment variable")
    else:
        # Default: all strategies
        chunking_strategies = list(CHUNKING_STRATEGIES.values())
        logger.info("[CONFIG] Using default chunking strategies (all)")
    
    # Check for dry run in environment variable
    dry_run = os.getenv("DRY_RUN", "").lower() == "true"
    if dry_run:
        logger.info("[CONFIG] DRY_RUN mode enabled")
    
    # Pinecone configuration
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
    
    if not pinecone_api_key:
        logger.error("Missing required environment variable: PINECONE_API_KEY")
        return
    
    # Run upload with selected configurations
    await run_upload(
        embedding_configs=embedding_configs,
        chunking_strategies=chunking_strategies,
        pinecone_api_key=pinecone_api_key,
        pinecone_cloud=pinecone_cloud,
        pinecone_region=pinecone_region,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    asyncio.run(main())

