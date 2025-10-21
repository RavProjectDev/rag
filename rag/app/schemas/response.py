from typing import Any, Optional
from pydantic import BaseModel, Field

from rag.app.models.data import SanityData, Metadata, DocumentModel


class TranscriptData(BaseModel):
    sanity_data: SanityData
    metadata: Metadata
    score: float

    def to_dict(self) -> dict:
        return {
            **self.metadata.model_dump(),
            "sanity_data": self.sanity_data.to_dict(),
            "score": self.score,
        }


class ChatResponse(BaseModel):
    message: str
    transcript_data: list[TranscriptData]
    prompt_id: str = None


class FormFullResponse(BaseModel):
    responses: list[ChatResponse]


class UploadResponse(BaseModel):
    message: str


class FormGetChunksResponse(BaseModel):
    embedding_type: str
    documents: list[DocumentModel]


class ErrorResponse(BaseModel):
    """Standard error response shape for API endpoints.

    - code: stable, machine-readable error code
    - message: human-readable description
    - request_id: optional correlation id to trace logs
    - details: optional contextual payload
    """

    code: str = Field(..., description="Stable machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    request_id: Optional[str] = Field(
        default=None, description="Correlation id for tracing this request"
    )
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Optional context for debugging"
    )


class SuccessResponse(BaseModel):
    """Minimal success envelope used for non-resource actions."""

    success: bool = True
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health probe response payload."""

    status: str = Field(..., description="Service status, e.g., 'ok'")
    version: Optional[str] = Field(default=None, description="Service version")
    environment: Optional[str] = Field(default=None, description="Deployment env")
