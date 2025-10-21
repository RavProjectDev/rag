from fastapi import APIRouter, Request
from rag.app.core.config import get_settings
from rag.app.schemas.response import HealthResponse

router = APIRouter()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Service health probe",
    description="Returns basic health information about the API service.",
)
def get_health(request: Request) -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        environment=settings.environment.value,
    )
