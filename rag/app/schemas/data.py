import uuid
from enum import Enum, auto
from typing import List, Union, Optional

from pydantic import BaseModel

from rag.app.models.data import SanityData


class Chunk(BaseModel):
    """
    Represents a chunk of text with metadata and character position.

    Attributes:
        metadata (dict[str, str]): Additional metadata about the chunk (e.g., source, date).
        text_to_embed (str): The raw text content of the chunk.
        chunk_size (int): The total size of the chunk in characters.
        char_start (int): The starting character index of the chunk in the original text.
        char_end (int): The ending character index of the chunk in the original text.
    """

    full_text_id: uuid = uuid.uuid4()
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
