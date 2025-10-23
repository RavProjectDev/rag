import pytest

from rag.app.db.mongodb_connection import MongoEmbeddingStore
from rag.tests.factories import make_vector_embedding
from rag.tests.test_db.conf import AsyncCollectionWrapper


@pytest.mark.asyncio
async def test_insert_successful():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    vector_embedding = make_vector_embedding()
    await store.insert(embedded_data=[vector_embedding])

    cursor = collection.find({})
    all_docs = await cursor.to_list()
    doc = all_docs[0]

    assert doc["sanity_data"]["id"] == vector_embedding.sanity_data.id
    assert doc["sanity_data"]["slug"] == vector_embedding.sanity_data.slug
    assert doc["sanity_data"]["title"] == vector_embedding.sanity_data.title
    assert doc["sanity_data"]["transcriptURL"] == str(
        vector_embedding.sanity_data.transcriptURL
    )
    assert doc["sanity_data"]["hash"] == vector_embedding.sanity_data.hash

    assert doc["metadata"]["chunk_size"] == vector_embedding.metadata.chunk_size
    assert doc["metadata"]["name_space"] == vector_embedding.metadata.name_space

    assert doc["vector"] == vector_embedding.vector


@pytest.mark.asyncio
async def test_insert_duplicate_vectors():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    vector_embedding = make_vector_embedding()
    await store.insert(embedded_data=[vector_embedding])
    await store.insert(embedded_data=[vector_embedding])

    cursor = collection.find({})
    all_docs = await cursor.to_list()
    assert len(all_docs) == 1


@pytest.mark.asyncio
async def test_insert_same_vectors_with_the_same_sanity_id():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    vector_embedding_1 = make_vector_embedding(
        randomize_metadata=True, randomize_vector=True
    )
    vector_embedding_2 = make_vector_embedding(
        randomize_metadata=True, randomize_vector=True
    )
    await store.insert(embedded_data=[vector_embedding_1])
    await store.insert(embedded_data=[vector_embedding_2])
    cursor = collection.find({})
    all_docs = await cursor.to_list()
    assert len(all_docs) == 1


@pytest.mark.asyncio
async def test_insert_many_vectors():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    num = 1000
    await store.insert(
        embedded_data=[
            make_vector_embedding(randomize_metadata=True, randomize_vector=True)
            for _ in range(num)
        ]
    )
    cursor = collection.find({})
    all_docs = await cursor.to_list()
    assert len(all_docs) == num


@pytest.mark.asyncio
async def test_insert_many_vectors_where_half_are_dups():
    collection = AsyncCollectionWrapper()
    store = MongoEmbeddingStore(collection, index="myindex", vector_path="vector")
    num = 1000
    embedded_data = [
        make_vector_embedding(randomize_metadata=True, randomize_vector=True)
        for _ in range(num)
    ]
    await store.insert(embedded_data=embedded_data)
    await store.insert(embedded_data=embedded_data)
    cursor = collection.find({})
    all_docs = await cursor.to_list()
    assert len(all_docs) == num
