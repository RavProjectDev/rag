import uuid

import pytest

from rag.app.db.mongodb_connection import MongoEmbeddingStore
from rag.app.exceptions.db import NoDocumentFoundException
from rag.tests.factories import make_vector_embedding
from rag.tests.test_db.conf import AsyncCollectionWrapper


@pytest.mark.asyncio
async def test_retrieve_no_document_found():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    with pytest.raises(NoDocumentFoundException):
        await store.retrieve(embedded_data=[0.1] * 748)


@pytest.mark.asyncio
async def test_retrieve_no_document_found():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    with pytest.raises(NoDocumentFoundException):
        await store.retrieve(embedded_data=[0.1] * 748, name_spaces=["mynamespace"])


@pytest.mark.asyncio
async def test_insert_and_retrieve():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    doc = make_vector_embedding()
    await collection.insert_many([doc.to_dict()])
    result = await store.retrieve(embedded_data=[0.1] * 748)
    expected_doc = result[0]
    assert expected_doc.sanity_data == doc.sanity_data
    assert expected_doc.score == 1


@pytest.mark.asyncio
async def test_insert_retrieve_with_name_space():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    doc = make_vector_embedding()
    await collection.insert_many([doc.to_dict()])
    result = await store.retrieve(
        embedded_data=[0.1] * 748, name_spaces=[doc.metadata.name_space]
    )
    expected_doc = result[0]
    assert expected_doc.sanity_data == doc.sanity_data


@pytest.mark.asyncio
async def test_insert_retrieve_with_namespace_not_found():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    doc = make_vector_embedding()
    await collection.insert_many([doc.to_dict()])
    with pytest.raises(NoDocumentFoundException):
        await store.retrieve(embedded_data=[0.1] * 748, name_spaces=[""])


@pytest.mark.asyncio
async def test_insert_retrieve_with_namespace_not_found_v2():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    doc = make_vector_embedding()
    await collection.insert_many([doc.to_dict()])
    with pytest.raises(NoDocumentFoundException):
        await store.retrieve(embedded_data=[0.1] * 748, name_spaces=["hello", "world"])


@pytest.mark.asyncio
async def test_insert_retrieve_with_namespace_not_found_v3():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    doc = make_vector_embedding()
    await collection.insert_many([doc.to_dict()])
    num = 1000
    name_spaces = [str(uuid.uuid4()) for _ in range(num)]
    with pytest.raises(NoDocumentFoundException):
        await store.retrieve(embedded_data=[0.1] * 748, name_spaces=name_spaces)
