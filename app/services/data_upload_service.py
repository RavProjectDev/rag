import logging

import httpx

from rag.app.db.connections import EmbeddingConnection
from rag.app.exceptions.upload import SRTFileNotFound
from rag.app.services.preprocess.transcripts import preprocess_raw_transcripts
from rag.app.schemas.data import (
    Chunk,
    VectorEmbedding,
    EmbeddingConfiguration,
    Embedding,
    SanityData,
    ChunkingStrategy,
)
from rag.app.services.embedding import generate_embedding
from rag.app.core.config import get_settings


async def pre_process_uploaded_document(
    upload_request: SanityData,
    embedding_configuration: EmbeddingConfiguration,
    connection: EmbeddingConnection,
):
    """
    Process and upload document chunks incrementally.
    Each chunk is embedded and inserted immediately rather than batching.
    """
    contents = await fetch_transcript(str(upload_request.transcriptURL))
    chunks = process_transcript_contents(upload_request.title, contents)
    await generate_and_insert_embeddings(
        chunks, embedding_configuration, upload_request, connection
    )


async def fetch_transcript(transcript_url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(transcript_url)
    if not response.content:
        raise SRTFileNotFound
    return response.content.decode("utf-8")


def process_transcript_contents(title: str, raw_text: str) -> list[Chunk]:
    """
    Process transcript contents using the configured chunking strategy.
    """
    settings = get_settings()
    raw_transcripts = [(title, raw_text)]
    return preprocess_raw_transcripts(
        raw_transcripts, 
        chunking_strategy=settings.chunking_strategy
    )


async def generate_all_embeddings(
    chunks: list[Chunk],
    configuration: EmbeddingConfiguration,
    upload_request: SanityData,
) -> list[VectorEmbedding]:

    sanity_data = SanityData(**upload_request.model_dump())
    return await embedding_helper(chunks, configuration, sanity_data)


async def generate_and_insert_embeddings(
    chunks: list[Chunk],
    configuration: EmbeddingConfiguration,
    sanity_data: SanityData,
    connection: EmbeddingConnection,
):
    """
    Generate embeddings for chunks and insert each one immediately.
    This provides better progress tracking and reduces memory usage for large documents.
    """
    total_chunks = len(chunks)
    logging.info(f"[EMBED] Starting incremental embedding and insertion for {total_chunks} chunks")
    
    for idx, chunk in enumerate(chunks, 1):
        # Generate embedding for this chunk
        data: Embedding = await generate_embedding(
            text=chunk.text_to_embed,
            configuration=configuration,
            task_type="RETRIEVAL_DOCUMENT",  # Document being indexed
        )
        
        # Create vector embedding
        embedding = VectorEmbedding(
            vector=data.vector,
            dimension=len(data.vector),
            metadata=chunk,
            sanity_data=sanity_data,
        )
        
        # Insert immediately
        await connection.insert([embedding])
        
        if idx % 10 == 0 or idx == total_chunks:
            logging.info(f"[EMBED] Progress: {idx}/{total_chunks} chunks embedded and inserted")
    
    logging.info(f"[EMBED] Completed embedding and insertion of all {total_chunks} chunks")


async def embedding_helper(
    chunks: list[Chunk],
    configuration: EmbeddingConfiguration,
    sanity_data: SanityData,
) -> list[VectorEmbedding]:
    """
    Legacy batch embedding function. Kept for backward compatibility.
    Generates all embeddings and returns them as a list.
    """
    embeddings = []
    for chunk in chunks:
        data: Embedding = await generate_embedding(
            text=chunk.text_to_embed,
            configuration=configuration,
            task_type="RETRIEVAL_DOCUMENT",  # Document being indexed
        )
        embeddings.append(
            VectorEmbedding(
                vector=data.vector,
                dimension=len(data.vector),
                metadata=chunk,
                sanity_data=sanity_data,
            )
        )
    return embeddings


async def upload_documents(
    documents: list[SanityData],
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
):
    for doc in documents:
        await upload_document(doc, connection, embedding_configuration)


async def upload_document(
    doc: SanityData,
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
):
    logging.info(f"[run] Uploading new document: {doc.title} (id: {doc.id})...")
    await pre_process_uploaded_document(
        upload_request=doc,
        embedding_configuration=embedding_configuration,
        connection=connection,
    )
    logging.info(f"[run] Finished uploading document: {doc.title}")


async def update_document(
    document: SanityData,
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
):
    deleted = await delete_document(document.transcript_id, connection)
    if not deleted:
        return
    await upload_document(document, connection, embedding_configuration)


async def update_documents(
    documents: list[SanityData],
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
):
    for doc in documents:
        await update_document(
            doc, connection, embedding_configuration=embedding_configuration
        )


async def delete_documents(
    documents: list[SanityData], connection: EmbeddingConnection
):
    for doc in documents:
        await delete_document(doc.transcript_id, connection)


async def delete_document(sanity_data_id: str, connection: EmbeddingConnection) -> bool:
    return await connection.delete_document(sanity_data_id)
