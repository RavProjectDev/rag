from typing import Any, Optional
from pydantic import BaseModel, Field
import uuid

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
    thread_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Thread ID returned from Supabase (if query was submitted)"
    )


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


class NamespaceDetail(BaseModel):
    """Details about a Pinecone namespace."""
    
    namespace: str = Field(..., description="Namespace name (typically chunking strategy)")
    vector_count: int = Field(..., description="Number of vectors in this namespace")


class PineconeIndexConfiguration(BaseModel):
    """Configuration details for a single Pinecone index."""
    
    index_name: str = Field(..., description="Pinecone index name")
    dimension: int = Field(..., description="Vector dimension of the index")
    metric: str = Field(..., description="Distance metric (e.g., 'cosine', 'euclidean')")
    namespaces: list[str] = Field(..., description="List of available namespace names")
    total_vector_count: int = Field(..., description="Total number of vectors across all namespaces")
    namespace_details: list[NamespaceDetail] = Field(..., description="Detailed information about each namespace")


class AvailableConfigurationsResponse(BaseModel):
    """Response containing all available Pinecone configurations."""
    
    available_configurations: list[PineconeIndexConfiguration] = Field(
        ..., 
        description="List of all available Pinecone index configurations"
    )
    default_index: Optional[str] = Field(
        None, 
        description="Default Pinecone index name from config"
    )
    default_namespace: Optional[str] = Field(
        None, 
        description="Default Pinecone namespace from config"
    )


class SimpleIndexConfig(BaseModel):
    """Simplified index configuration showing just index and namespaces."""
    
    index: str = Field(..., description="Pinecone index name")
    namespaces: list[str] = Field(..., description="List of available namespace names")


class SimpleConfigurationsResponse(BaseModel):
    """Simplified response showing just indexes and their namespaces."""
    
    indexes: list[SimpleIndexConfig] = Field(
        ..., 
        description="List of indexes with their available namespaces"
    )
    defaults: dict[str, Optional[str]] = Field(
        ...,
        description="Default index and namespace from environment config"
    )


class ChunkingStrategyInfo(BaseModel):
    """Information about a chunking strategy."""
    
    name: str = Field(..., description="Chunking strategy identifier")
    description: str = Field(..., description="What this chunking strategy does")


class EmbeddingModelConfig(BaseModel):
    """Configuration for an embedding model with available chunking strategies."""
    
    embedding_model: str = Field(..., description="Embedding model name (e.g., 'gemini-embedding-001')")
    chunking_strategies: list[ChunkingStrategyInfo] = Field(
        ..., 
        description="Available chunking strategies for this embedding model"
    )


class EnhancedConfigurationsResponse(BaseModel):
    """Enhanced response showing embedding models with chunking strategies and descriptions."""
    
    available_combinations: list[EmbeddingModelConfig] = Field(
        ..., 
        description="Available embedding model + chunking strategy combinations"
    )
    defaults: dict[str, Optional[str]] = Field(
        ...,
        description="Default embedding model (index) and chunking strategy (namespace) from environment config"
    )
    strategy_descriptions: dict[str, str] = Field(
        ...,
        description="Detailed descriptions of all chunking strategies"
    )
