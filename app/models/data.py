from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Union, List, Tuple


class Metadata(BaseModel):
    chunk_size: int
    time_start: str | None = None
    time_end: str | None = None
    name_space: str
    text_hash: str | None = None  # SHA-256 hash of the text


class SanityData(BaseModel):
    id: str
    updated_at: Optional[str] = Field(alias="_updatedAt", default=None)
    slug: str
    title: str
    transcriptURL: HttpUrl
    hash: str

    class Config:
        allow_population_by_field_name = True

    def to_dict(self) -> dict:
        data = self.model_dump(by_alias=True)
        # Ensure URL is serialized as string
        if "transcriptURL" in data:
            data["transcriptURL"] = str(data["transcriptURL"])
        return data


class Prompt(BaseModel):
    id: str
    value: str


class DocumentModel(BaseModel):
    id: str = Field(..., alias="_id")  # e.g. {"$oid": "687c65e061b769c8ff78779f"}
    text: Union[str, List[Tuple[str, Tuple[Optional[str], Optional[str]]]]]
    metadata: Metadata
    sanity_data: SanityData
    score: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.model_dump(),
            "sanity_data": self.sanity_data.to_dict(),
            "score": self.score,
        }
