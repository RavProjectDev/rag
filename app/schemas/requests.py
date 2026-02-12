from pydantic import BaseModel, HttpUrl, field_validator, Field
from enum import Enum, auto
import uuid

from rag.app.models.data import SanityData
from rag.app.services.prompts import PromptType


class TypeOfRequest(str, Enum):
    STREAM = "STREAM"
    FULL = "FULL"


class ChatRequest(BaseModel):
    question: str
    type_of_request: TypeOfRequest
    name_spaces: list[str] | None = None
    prompt_type: PromptType = Field(
        default=PromptType.STRUCTURED_JSON,
        description="Prompt type to use for generation. Defaults to Production mode."
    )
    pinecone_index: str | None = Field(
        default=None,
        description=(
            "Optional Pinecone index name to query (e.g., 'gemini-embedding-001'). "
            "If not specified, uses PINECONE_INDEX_NAME from environment variables. "
            "Use /api/v1/config/available-configs to see available options."
        )
    )
    pinecone_namespace: str | None = Field(
        default=None,
        description=(
            "Optional Pinecone namespace for chunking strategy (e.g., 'sentence_fixed_regex'). "
            "If not specified, uses PINECONE_NAMESPACE from environment variables. "
            "Use /api/v1/config/available-configs to see available options."
        )
    )
    thread_id: uuid.UUID | None = Field(
        default=None,
        description="Optional thread ID to append query to. If not provided, a new thread will be created."
    )
    submit_query: bool = Field(
        default=True,
        description="Whether to save the query and response to Supabase. Set to false to skip persistence."
    )

    @classmethod
    @field_validator("question")
    def question_validator(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be empty")
        return v


class RetrieveDocumentsRequest(BaseModel):
    question: str
    name_spaces: list[str] | None = None
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top documents to return (1-50).",
    )
    pinecone_index: str | None = Field(
        default=None,
        description=(
            "Optional Pinecone index name to query (e.g., 'gemini-embedding-001'). "
            "If not specified, uses PINECONE_INDEX_NAME from environment variables. "
            "Use /api/v1/config/available-configs to see available options."
        )
    )
    pinecone_namespace: str | None = Field(
        default=None,
        description=(
            "Optional Pinecone namespace for chunking strategy (e.g., 'sentence_fixed_regex'). "
            "If not specified, uses PINECONE_NAMESPACE from environment variables. "
            "Use /api/v1/config/available-configs to see available options."
        )
    )

    @classmethod
    @field_validator("question")
    def question_validator(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be empty")
        return v


class FormRequest(BaseModel):
    question: str
    prompt_type: PromptType = Field(
        default=PromptType.STRUCTURED_JSON,
        description="Prompt type to use for generation. Defaults to Production mode."
    )

    @classmethod
    @field_validator("question")
    def question_validator(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be empty")
        return v
