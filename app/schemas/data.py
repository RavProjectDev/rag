import uuid
from enum import Enum, auto
from typing import List, Union, Optional

from pydantic import BaseModel

from rag.app.models.data import SanityData


class Chunk(BaseModel):
    """
    Represents a chunk of text with metadata and character position.

    Attributes:
        full_text_id (uuid.UUID): Unique identifier shared by all chunks from the same text segment.
        full_text (str): The complete text segment (e.g., 200 words).
        text_to_embed (str): The specific portion to embed (e.g., 50 words).
        chunk_size (int): The total size of the full text segment in words.
        embed_size (int): The size of this specific embedding chunk in words.
        time_start (str): Start timestamp from SRT file, if available.
        time_end (str): End timestamp from SRT file, if available.
        name_space (str): Identifier for the source transcript/document.
    """

    full_text_id: uuid.UUID
    full_text: str
    text_to_embed: str
    chunk_size: int
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    name_space: str
    embed_size: int

    def to_dict(self):
        return {
            "chunk_size": self.chunk_size,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "name_space": self.name_space,
            "embed_size": self.embed_size,
            "full_text_id": str(self.full_text_id),
        }

    model_config = {"env_file": ".env", "arbitrary_types_allowed": True}


class VectorEmbedding(BaseModel):
    """
    Represents an embedding vector for a chunk of text.

    Attributes:
        vector (list[float]): The embedding vector representation.
        dimension (int): The dimensionality of the embedding.
        metadata (Chunk): The associated Chunk object containing the source text and metadata.
    """

    vector: List[float]
    dimension: int
    metadata: Chunk
    sanity_data: SanityData

    def to_dict(self) -> dict:
        return {
            "vector": self.vector,
            "text": self.metadata.full_text,
            "text_id": str(self.metadata.full_text_id),  # Top-level for MongoDB grouping
            "metadata": self.metadata.to_dict(),
            "sanity_data": self.sanity_data.to_dict(),
        }


class Embedding(BaseModel):
    text: str
    vector: List[float]


class DataSourceConfiguration(Enum):
    LOCAL = auto()


class TypeOfFormat(Enum):
    SRT = auto()
    TXT = auto()


class DataBaseConfiguration(Enum):
    PINECONE = auto()
    MONGO = auto()


class EmbeddingConfiguration(Enum):
    BERT_SMALL = "all-MiniLM-L6-v2"
    BERT_SMALL_TRANSLATED = "all-MiniLM-L6-v2"
    GEMINI = "gemini-embedding-001"
    MOCK = "mock"


class LLMModel(Enum):
    GPT_4 = "o4-mini"
    MOCK = "mock"


class TranscriptData(BaseModel):
    transcript_id: str
    transcript_hash: str
