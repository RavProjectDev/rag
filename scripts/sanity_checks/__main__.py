"""
Sanity Check Main Entry Point

Run comprehensive sanity check across all configurations.

Usage:
    # Check all configurations
    python -m rag.scripts.sanity_checks
    
    # Check specific embedding model
    EMBEDDING_MODELS=gemini python -m rag.scripts.sanity_checks
    
    # Check specific chunking strategy
    CHUNKING_STRATEGIES=semantic python -m rag.scripts.sanity_checks
    
    # Custom output file
    OUTPUT_FILE=check_2026_01_08.json python -m rag.scripts.sanity_checks
    
    # Limit retry attempts (for testing)
    MAX_ATTEMPTS=5 python -m rag.scripts.sanity_checks
"""

import asyncio
import logging
import os
from itertools import product
from typing import List
from dotenv import load_dotenv

from rag.app.models.data import SanityData
from rag.app.schemas.data import EmbeddingConfiguration, ChunkingStrategy
from rag.app.db.pinecone_connection import PineconeEmbeddingStore

from .config import (
    EMBEDDING_MODELS,
    CHUNKING_STRATEGIES,
    get_index_name,
    get_namespace_name,
    get_embedding_dimension,
)
from .fetch import fetch_manifest
from .pinecone_query import get_pinecone_stats
from .comparator import check_document_chunks
from .reporter import generate_report, save_report, print_summary, print_missing_chunks_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            logger.warning(f"Unknown embedding model '{model}'. Skipping.")
    
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
            logger.warning(f"Unknown chunking strategy '{strategy}'. Skipping.")
    
    return configs if configs else None


async def run_sanity_check(
    embedding_configs: List[EmbeddingConfiguration],
    chunking_strategies: List[ChunkingStrategy],
    pinecone_api_key: str,
    pinecone_cloud: str = "aws",
    pinecone_region: str = "us-east-1",
    output_file: str = "sanity_check_report.json",
    max_attempts: int | None = None,
):
    """
    Run comprehensive sanity check across all configurations.
    
    Args:
        embedding_configs: List of embedding configurations to check
        chunking_strategies: List of chunking strategies to check
        pinecone_api_key: Pinecone API key
        pinecone_cloud: Pinecone cloud provider
        pinecone_region: Pinecone region
        output_file: Path to output JSON report
        max_attempts: Optional maximum retry attempts (None = infinite)
    """
    logger.info("[SANITY CHECK] Starting sanity check...")
    
    # Fetch manifest
    logger.info("[SANITY CHECK] Fetching manifest...")
    manifest = await fetch_manifest()
    if not manifest:
        logger.error("[SANITY CHECK] Failed to fetch manifest")
        return
    
    documents = [SanityData(id=doc_id, **content) for doc_id, content in manifest.items()]
    logger.info(f"[SANITY CHECK] Found {len(documents)} documents in manifest")
    
    # Create combinations
    combinations = list(product(embedding_configs, chunking_strategies))
    logger.info(f"[SANITY CHECK] Checking {len(combinations)} configuration(s)")
    
    # Store all configuration reports
    all_configurations = []
    
    # Check each combination
    for idx, (embedding_config, chunking_strategy) in enumerate(combinations, 1):
        index_name = get_index_name(embedding_config)
        namespace_name = get_namespace_name(chunking_strategy)
        dimension = get_embedding_dimension(embedding_config)
        
        logger.info(
            f"\n{'='*80}\n"
            f"[SANITY CHECK] Configuration {idx}/{len(combinations)}\n"
            f"[SANITY CHECK] Embedding: {embedding_config.value}\n"
            f"[SANITY CHECK] Chunking: {chunking_strategy.value}\n"
            f"[SANITY CHECK] Index: {index_name}\n"
            f"[SANITY CHECK] Namespace: {namespace_name}\n"
            f"{'='*80}"
        )
        
        # Create Pinecone connection
        connection = PineconeEmbeddingStore(
            api_key=pinecone_api_key,
            index_name=index_name,
            dimension=dimension,
            metric="cosine",
            cloud=pinecone_cloud,
            region=pinecone_region,
        )
        
        # Get overall stats
        stats = get_pinecone_stats(connection, namespace_name)
        logger.info(
            f"[SANITY CHECK] Pinecone stats: {stats['vector_count']} vectors in namespace '{namespace_name}'"
        )
        
        # Configuration report
        config_report = {
            "embedding_config": embedding_config.value,
            "chunking_strategy": chunking_strategy.value,
            "index_name": index_name,
            "namespace": namespace_name,
            "pinecone_stats": stats,
            "documents": [],
            "summary": {
                "ok": 0,
                "missing_all": 0,
                "incomplete": 0,
                "mismatch": 0,
                "errors": 0,
            }
        }
        
        # Check each document
        for doc_idx, doc in enumerate(documents, 1):
            if doc_idx % 10 == 0:
                logger.info(f"[SANITY CHECK] Checked {doc_idx}/{len(documents)} documents...")
            
            doc_report = await check_document_chunks(
                doc=doc,
                embedding_config=embedding_config,
                chunking_strategy=chunking_strategy,
                connection=connection,
                namespace=namespace_name,
                max_attempts=max_attempts,
            )
            
            config_report["documents"].append(doc_report)
            
            # Update summary
            status = doc_report["status"]
            if status == "OK":
                config_report["summary"]["ok"] += 1
            elif status == "MISSING_ALL":
                config_report["summary"]["missing_all"] += 1
            elif status == "INCOMPLETE":
                config_report["summary"]["incomplete"] += 1
            elif status == "MISMATCH":
                config_report["summary"]["mismatch"] += 1
            elif status == "ERROR":
                config_report["summary"]["errors"] += 1
        
        # Log configuration summary
        logger.info(
            f"[SANITY CHECK] Configuration summary:\n"
            f"  - OK: {config_report['summary']['ok']}\n"
            f"  - Missing All: {config_report['summary']['missing_all']}\n"
            f"  - Incomplete: {config_report['summary']['incomplete']}\n"
            f"  - Mismatch: {config_report['summary']['mismatch']}\n"
            f"  - Errors: {config_report['summary']['errors']}"
        )
        
        all_configurations.append(config_report)
    
    # Generate and save report
    full_report = generate_report(documents, all_configurations)
    save_report(full_report, output_file)
    
    # Print summaries
    print_summary(full_report)
    print_missing_chunks_summary(full_report)
    
    logger.info(f"\n[SANITY CHECK] Complete! Report saved to: {output_file}")
    
    return full_report


async def main():
    """Main entry point."""
    load_dotenv()
    
    # Parse configuration from environment
    env_configs = parse_embedding_configs_from_env()
    embedding_configs = env_configs if env_configs else list(EMBEDDING_MODELS.values())
    logger.info(f"[CONFIG] Embedding models: {[c.value for c in embedding_configs]}")
    
    env_strategies = parse_chunking_strategies_from_env()
    chunking_strategies = env_strategies if env_strategies else list(CHUNKING_STRATEGIES.values())
    logger.info(f"[CONFIG] Chunking strategies: {[s.value for s in chunking_strategies]}")
    
    # Output file
    output_file = os.getenv("OUTPUT_FILE", "sanity_check_report.json")
    logger.info(f"[CONFIG] Output file: {output_file}")
    
    # Max attempts (for testing, default is infinite)
    max_attempts_str = os.getenv("MAX_ATTEMPTS")
    max_attempts = int(max_attempts_str) if max_attempts_str else None
    if max_attempts:
        logger.info(f"[CONFIG] Max retry attempts: {max_attempts}")
    else:
        logger.info("[CONFIG] Max retry attempts: Infinite")
    
    # Pinecone configuration
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
    
    if not pinecone_api_key:
        logger.error("Missing required environment variable: PINECONE_API_KEY")
        return
    
    # Run sanity check
    await run_sanity_check(
        embedding_configs=embedding_configs,
        chunking_strategies=chunking_strategies,
        pinecone_api_key=pinecone_api_key,
        pinecone_cloud=pinecone_cloud,
        pinecone_region=pinecone_region,
        output_file=output_file,
        max_attempts=max_attempts,
    )


if __name__ == "__main__":
    asyncio.run(main())

