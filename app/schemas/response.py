from typing import Any, Optional
from pydantic import BaseModel, Field

from rag.app.models.data import SanityData, Metadata, DocumentModel


class TranscriptData(BaseModel):
    sanity_data: SanityData
    metadata: Metadata
    score: float
    text_id: str

    def to_dict(self) -> dict:
        return {
            **self.metadata.model_dump(),
            "sanity_data": self.sanity_data.to_dict(),
            "score": self.score,
            "text_id": self.text_id,
        }


class UsedQuote(BaseModel):
    """Individual quote that was used from a source."""
    number: int
    text: str
    timestamp: Optional[str] = None


class SourceItem(BaseModel):
    slug: str
    text_id: Optional[str] = None
    full_text: str  # Full text with bolded used quotes (using **text** for bold)
    used_quotes: list[UsedQuote]  # List of individual quotes that were used
    timestamp_range: Optional[str] = None  # Overall timestamp range for the full document


class ChatResponse(BaseModel):
    main_text: str
    sources: list[SourceItem]


class RetrieveDocumentsResponse(BaseModel):
    request_id: str
    cleaned_question: str
    requested_top_k: int
    documents: list[DocumentModel]
    transcript_data: list[TranscriptData]
    message: Optional[str] = None


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


class RateLimitInfoResponse(BaseModel):
    """User rate limit information response."""

    user_id: str = Field(..., description="User ID")
    current_usage: int = Field(..., description="Current number of requests used this month")
    remaining: int = Field(..., description="Number of requests remaining this month")
    limit: int = Field(..., description="Maximum requests allowed per month")
    reset_at: str = Field(..., description="ISO 8601 timestamp when the rate limit resets")
    reset_in_seconds: int = Field(..., description="Seconds until rate limit resets")


class ConfigInfoResponse(BaseModel):
    """Configuration information response."""

    embedding_model: str = Field(..., description="Current embedding model configuration")
    chunking_strategy: str = Field(..., description="Current chunking strategy configuration")
    database_backend: str = Field(..., description="Current database backend (mongo or pinecone)")
    environment: str = Field(..., description="Current environment (PRD, STG, TEST)")
