from fastapi import APIRouter, Request
from rag.app.core.config import get_settings
from rag.app.schemas.response import ConfigInfoResponse

router = APIRouter()


@router.get(
    "/",
    response_model=ConfigInfoResponse,
    summary="Get current configuration",
    description="Returns the current embedding model, chunking strategy, and database backend configuration.",
)
def get_config_info(request: Request) -> ConfigInfoResponse:
    """
    Get current configuration information.
    
    This endpoint returns the active configuration that the API is using,
    including the embedding model, chunking strategy, and database backend.
    This is used by sync scripts to ensure they use the same configuration.
    """
    settings = get_settings()
    return ConfigInfoResponse(
        embedding_model=settings.embedding_configuration.value,
        chunking_strategy=settings.chunking_strategy.value,
        database_backend=settings.database_configuration.value,
        environment=settings.environment.value,
    )
