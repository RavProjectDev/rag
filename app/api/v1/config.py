import logging
import time
from fastapi import APIRouter, HTTPException, Depends
from pinecone import Pinecone

from rag.app.core.config import get_settings
from rag.app.schemas.response import (
    AvailableConfigurationsResponse,
    PineconeIndexConfiguration,
    NamespaceDetail,
    SimpleConfigurationsResponse,
    SimpleIndexConfig,
    EnhancedConfigurationsResponse,
    EmbeddingModelConfig,
    ChunkingStrategyInfo,
)
from rag.app.services.auth import verify_jwt_token

router = APIRouter()
logger = logging.getLogger(__name__)

# Cache for Pinecone configurations
_config_cache = {"data": None, "timestamp": 0}
CACHE_TTL = 300  # 5 minutes

# Chunking strategy descriptions
CHUNKING_STRATEGY_DESCRIPTIONS = {
    "fixed_size": "Fixed-size chunks: Accumulates text until reaching a token limit (e.g., 250 tokens). Simple and fast.",
    "divided": "Divided chunks: Creates larger chunks then subdivides them into smaller overlapping pieces. Provides context overlap.",
    "sentence_fixed_regex": "Sentence-aware fixed: Creates chunks that respect sentence boundaries using punctuation detection. Chunks end at natural sentence breaks.",
    "sentence_divided_regex": "Sentence-aware divided: Combines sentence-aware chunking with subdivision. Creates sentence-based main chunks, then subdivides for overlap.",
}


@router.get(
    "/available-configs",
    response_model=EnhancedConfigurationsResponse,
    summary="Get available embedding models and chunking strategies",
    description=(
        "Returns all available embedding model + chunking strategy combinations with descriptions. "
        "This endpoint queries Pinecone directly to provide an up-to-date list "
        "of available configurations. Each combination shows which chunking strategies "
        "are available for each embedding model, along with clear descriptions of what each strategy does. "
        "Results are cached for 5 minutes. Use force_refresh=true to bypass cache."
    ),
)
async def get_available_configurations(
    user_id: str = Depends(verify_jwt_token),
    force_refresh: bool = False,
) -> EnhancedConfigurationsResponse:
    """
    Get all available embedding model + chunking strategy combinations with descriptions.
    
    This endpoint dynamically fetches configuration options from Pinecone,
    eliminating the need for hardcoded configurations. Returns embedding models
    (indexes) with their available chunking strategies (namespaces), plus helpful
    descriptions so users understand what each strategy does.
    
    Args:
        user_id: Authenticated user ID (from JWT token)
        force_refresh: If True, bypass cache and fetch fresh data from Pinecone
        
    Returns:
        EnhancedConfigurationsResponse with embedding models, chunking strategies, and descriptions
        
    Raises:
        HTTPException 503: If Pinecone is not configured
        HTTPException 500: If fetching configurations fails
    """
    settings = get_settings()
    
    # Check if Pinecone is configured
    if not settings.pinecone_api_key:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "pinecone_not_configured",
                "message": "Pinecone is not configured. Cannot fetch available configurations.",
            },
        )
    
    # Check cache
    current_time = time.time()
    if (
        not force_refresh
        and _config_cache["data"]
        and (current_time - _config_cache["timestamp"]) < CACHE_TTL
    ):
        logger.info("[CONFIG] Returning cached Pinecone configurations")
        return _config_cache["data"]
    
    logger.info("[CONFIG] Fetching fresh Pinecone configurations from API")
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=settings.pinecone_api_key)
        
        # List all indexes
        indexes_list = pc.list_indexes()
        
        model_configs = []
        
        for index_info in indexes_list:
            index_name = index_info.name
            
            try:
                # Get the index object
                index = pc.Index(index_name)
                
                # Get index stats (includes namespace information)
                stats = index.describe_index_stats()
                
                # Extract namespaces (chunking strategies)
                namespaces_dict = stats.get("namespaces", {})
                namespaces = sorted(list(namespaces_dict.keys()))  # Sort alphabetically
                
                # Build chunking strategy info with descriptions
                chunking_strategies = []
                for namespace in namespaces:
                    strategy_info = ChunkingStrategyInfo(
                        name=namespace,
                        description=CHUNKING_STRATEGY_DESCRIPTIONS.get(
                            namespace, 
                            "Custom chunking strategy"
                        )
                    )
                    chunking_strategies.append(strategy_info)
                
                model_configs.append(
                    EmbeddingModelConfig(
                        embedding_model=index_name,
                        chunking_strategies=chunking_strategies,
                    )
                )
                
                logger.info(
                    f"[CONFIG] Fetched embedding model '{index_name}': {len(namespaces)} chunking strategies"
                )
                
            except Exception as index_error:
                logger.warning(
                    f"[CONFIG] Failed to fetch stats for index '{index_name}': {index_error}"
                )
                # Continue processing other indexes even if one fails
                continue
        
        # Sort by embedding model name alphabetically
        model_configs.sort(key=lambda x: x.embedding_model)
        
        # Build enhanced response
        response = EnhancedConfigurationsResponse(
            available_combinations=model_configs,
            defaults={
                "embedding_model": settings.pinecone_index_name,
                "chunking_strategy": settings.pinecone_namespace,
            },
            strategy_descriptions=CHUNKING_STRATEGY_DESCRIPTIONS,
        )
        
        # Update cache
        _config_cache["data"] = response
        _config_cache["timestamp"] = current_time
        
        logger.info(
            f"[CONFIG] Successfully fetched {len(model_configs)} embedding models. "
            f"Cached for {CACHE_TTL}s"
        )
        
        return response
        
    except Exception as e:
        logger.exception("[CONFIG] Failed to fetch Pinecone configurations")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "pinecone_fetch_failed",
                "message": f"Failed to fetch Pinecone configurations: {str(e)}",
            },
        )
