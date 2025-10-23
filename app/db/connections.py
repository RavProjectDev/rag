import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from rag.app.schemas.data import VectorEmbedding
from rag.app.models.data import DocumentModel
from contextlib import asynccontextmanager
from rag.app.schemas.data import TranscriptData


class EmbeddingConnection(ABC):
    """
    Abstract class that represents a connection to the database.
    Providing an additional abstraction to connect to the database.
    """

    @abstractmethod
    async def insert(self, embedded_data: List[VectorEmbedding]) -> list[DocumentModel]:
        """
        Inserts one vector to the database.
        :param collection:
        :param embedded_data:
        A transcript "chunk"
        :return:
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        embedded_data: List[float],
        name_spaces: list[str] | None = None,
        threshold: float = 0.85,
    ):
        """
        Retrieves documents based on vector
        """
        pass

    @abstractmethod
    async def get_all_unique_transcript_ids(self) -> list[TranscriptData]:
        pass

    @abstractmethod
    async def delete_document(self, transcript_id: str) -> bool:
        pass


class MetricsConnection(ABC):
    @abstractmethod
    async def log(self, metric_type: str, data: Dict[str, Any]):
        pass

    @asynccontextmanager
    async def timed(self, metric_type: str, data: Dict[str, Any]):
        """
        Context manager to time a block and automatically log the duration.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            data_with_duration = {**data, "duration": f"{duration:.4f}"}
            await self.log(metric_type, data_with_duration)


class ExceptionsLogger(ABC):
    @abstractmethod
    async def log(self, exception_code: str | None, data: Dict[str, Any]):
        pass
