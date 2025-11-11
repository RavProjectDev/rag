from pydantic import BaseModel, HttpUrl, field_validator, Field
from enum import Enum, auto

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
