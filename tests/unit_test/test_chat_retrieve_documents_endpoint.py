import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.app.api.v1.chat import router as chat_router
from rag.app.db.connections import EmbeddingConnection, MetricsConnection
from rag.app.dependencies import (
    get_embedding_conn,
    get_metrics_conn,
    get_embedding_configuration,
    get_llm_configuration,
)
from rag.app.exceptions.db import NoDocumentFoundException
from rag.app.models.data import DocumentModel, Metadata, SanityData
from rag.app.schemas.data import EmbeddingConfiguration, LLMModel


def build_document(identifier: int, score: float) -> DocumentModel:
    metadata = Metadata(
        chunk_size=100,
        time_start="00:00:00,000",
        time_end="00:00:05,000",
        name_space="default",
    )
    sanity_data = SanityData(
        id=f"doc-{identifier}",
        slug=f"slug-{identifier}",
        title=f"title-{identifier}",
        transcriptURL="https://example.com/transcript",
        hash=f"hash-{identifier}",
    )
    return DocumentModel(
        _id=str(identifier),
        text=f"Document {identifier}",
        metadata=metadata,
        sanity_data=sanity_data,
        score=score,
    )


class FakeEmbeddingConnection(EmbeddingConnection):
    def __init__(self, documents: list[DocumentModel], raise_no_docs: bool = False):
        self.documents = documents
        self.raise_no_docs = raise_no_docs
        self.last_k = None
        self.last_name_spaces = None

    async def insert(self, embedded_data, *args, **kwargs):
        return []

    async def retrieve(
        self,
        embedded_data,
        name_spaces: list[str] | None = None,
        k: int = 5,
        threshold: float = 0.85,
    ):
        self.last_name_spaces = name_spaces
        self.last_k = k
        if self.raise_no_docs:
            raise NoDocumentFoundException()
        return self.documents[:k]

    async def get_all_unique_transcript_ids(self):
        return []

    async def delete_document(self, transcript_id: str) -> bool:
        return False


class FakeMetricsConnection(MetricsConnection):
    def __init__(self):
        self.logged: list[dict] = []

    async def log(self, metric_type: str, data: dict):
        self.logged.append({"metric_type": metric_type, "data": data})


@pytest.fixture
def client_factory():
    def _create(documents=None, raise_no_docs=False):
        docs = documents or []
        embedding_conn = FakeEmbeddingConnection(docs, raise_no_docs=raise_no_docs)
        metrics_conn = FakeMetricsConnection()

        app = FastAPI()
        app.include_router(chat_router, prefix="/api/v1/chat")

        app.dependency_overrides[get_embedding_conn] = lambda: embedding_conn
        app.dependency_overrides[get_metrics_conn] = lambda: metrics_conn
        app.dependency_overrides[get_embedding_configuration] = (
            lambda: EmbeddingConfiguration.MOCK
        )
        app.dependency_overrides[get_llm_configuration] = lambda: LLMModel.MOCK

        client = TestClient(app)
        return client, embedding_conn, metrics_conn

    return _create


def test_retrieve_documents_respects_top_k(client_factory):
    documents = [
        build_document(1, 0.98),
        build_document(2, 0.96),
        build_document(3, 0.94),
    ]
    client, embedding_conn, _ = client_factory(documents=documents)

    payload = {"question": "What is halakha?", "top_k": 2}
    response = client.post("/api/v1/chat/documents", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert body["requested_top_k"] == 2
    assert len(body["documents"]) == 2
    assert embedding_conn.last_k == 2
    assert body["cleaned_question"] == "halakha?"
    assert len(body["transcript_data"]) == 2
    returned_slugs = [doc["sanity_data"]["slug"] for doc in body["documents"]]
    assert returned_slugs == ["slug-1", "slug-2"]


def test_retrieve_documents_emits_metrics(client_factory):
    documents = [build_document(1, 0.98)]
    client, _, metrics_conn = client_factory(documents=documents)

    response = client.post("/api/v1/chat/documents", json={"question": "Where now?"})

    assert response.status_code == 200
    metric_types = [entry["metric_type"] for entry in metrics_conn.logged]
    assert metric_types.count("EMBEDDING") == 1
    assert metric_types.count("RETRIEVAL") == 1
    for entry in metrics_conn.logged:
        assert "duration" in entry["data"]


def test_retrieve_documents_handles_no_document_found(client_factory):
    client, _, _ = client_factory(documents=[], raise_no_docs=True)

    response = client.post("/api/v1/chat/documents", json={"question": "What now?"})

    assert response.status_code == 200
    body = response.json()

    assert body["documents"] == []
    assert body["transcript_data"] == []
    assert body["message"] == NoDocumentFoundException.message_to_ui

