import logging
import asyncio
import random

import httpx

from rag.app.db.connections import EmbeddingConnection
from rag.app.exceptions.upload import SRTFileNotFound
from rag.app.exceptions.embedding import (
    EmbeddingException,
    EmbeddingTimeOutException,
    EmbeddingAPIException,
    EmbeddingConfigurationException,
)
from rag.app.exceptions.db import InsertException
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

# Retry configuration for embedding and insertion
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 10.0  # seconds


async def pre_process_uploaded_document(
    upload_request: SanityData,
    embedding_configuration: EmbeddingConfiguration,
    connection: EmbeddingConnection,
    chunking_strategy: ChunkingStrategy,
):
    """
    Process and upload document chunks incrementally.
    Each chunk is embedded and inserted immediately rather than batching.
    """
    contents = await fetch_transcript(str(upload_request.transcriptURL))
    chunks = process_transcript_contents(upload_request.title, contents, chunking_strategy)
    await generate_and_insert_embeddings(
        chunks, embedding_configuration, upload_request, connection, chunking_strategy
    )


async def fetch_transcript(transcript_url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(transcript_url)
    if not response.content:
        raise SRTFileNotFound
    return response.content.decode("utf-8")


def process_transcript_contents(title: str, raw_text: str, chunking_strategy: ChunkingStrategy) -> list[Chunk]:
    """
    Process transcript contents using the configured chunking strategy.
    """
    raw_transcripts = [(title, raw_text)]
    return preprocess_raw_transcripts(
        raw_transcripts, 
        chunking_strategy=chunking_strategy
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
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
):
    """
    Generate embeddings for chunks and insert each one immediately with retry logic.
    This provides better progress tracking and reduces memory usage for large documents.
    
    Retries on transient errors:
    - EmbeddingTimeOutException
    - EmbeddingAPIException
    - InsertException
    
    Does NOT retry on permanent errors like EmbeddingConfigurationException.
    """
    total_chunks = len(chunks)
    logging.info(f"[EMBED] Starting incremental embedding and insertion for {total_chunks} chunks")
    
    for idx, chunk in enumerate(chunks, 1):
        # Retry logic with exponential backoff
        last_exception = None
        current_delay = INITIAL_RETRY_DELAY
        
        for attempt in range(MAX_RETRIES + 1):  # +1 for initial attempt
            try:
                logging.info(f"[EMBED] Chunk {idx}/{total_chunks}: Attempt {attempt + 1}/{MAX_RETRIES + 1}")
                
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
                    embedding_model=configuration.value,
                    chunking_strategy=chunking_strategy.value,
                )
                
                # Insert immediately
                await connection.insert([embedding])
                
                # Success! Break out of retry loop
                if idx % 10 == 0 or idx == total_chunks:
                    logging.info(f"[EMBED] Progress: {idx}/{total_chunks} chunks embedded and inserted")
                break
                
            except EmbeddingConfigurationException as e:
                # Configuration errors are permanent - don't retry
                logging.error(f"[EMBED ERROR] Configuration error for chunk {idx}/{total_chunks}: {e}")
                raise
                
            except (EmbeddingTimeOutException, EmbeddingAPIException, InsertException) as e:
                # Transient errors - retry with backoff
                last_exception = e
                exception_type = type(e).__name__
                logging.warning(
                    f"[EMBED RETRY] {exception_type} on chunk {idx}/{total_chunks}, "
                    f"attempt {attempt + 1}/{MAX_RETRIES + 1}: {str(e)}"
                )
                
                if attempt < MAX_RETRIES:
                    # Exponential backoff with jitter
                    jitter = random.uniform(0, 0.1 * current_delay)
                    delay = min(current_delay + jitter, MAX_RETRY_DELAY)
                    logging.info(f"[EMBED RETRY] Waiting {delay:.2f}s before retry...")
                    await asyncio.sleep(delay)
                    current_delay *= 2  # Exponential backoff
                else:
                    # Max retries exceeded
                    logging.error(
                        f"[EMBED ERROR] Max retries ({MAX_RETRIES}) exceeded for chunk {idx}/{total_chunks}"
                    )
                    raise last_exception
                    
            except EmbeddingException as e:
                # Other embedding errors - retry
                last_exception = e
                logging.warning(
                    f"[EMBED RETRY] EmbeddingException on chunk {idx}/{total_chunks}, "
                    f"attempt {attempt + 1}/{MAX_RETRIES + 1}: {str(e)}"
                )
                
                if attempt < MAX_RETRIES:
                    jitter = random.uniform(0, 0.1 * current_delay)
                    delay = min(current_delay + jitter, MAX_RETRY_DELAY)
                    logging.info(f"[EMBED RETRY] Waiting {delay:.2f}s before retry...")
                    await asyncio.sleep(delay)
                    current_delay *= 2
                else:
                    logging.error(
                        f"[EMBED ERROR] Max retries ({MAX_RETRIES}) exceeded for chunk {idx}/{total_chunks}"
                    )
                    raise last_exception
                    
            except Exception as e:
                # Unexpected errors - log and raise immediately
                logging.error(
                    f"[EMBED ERROR] Unexpected error for chunk {idx}/{total_chunks}: {str(e)}",
                    exc_info=True
                )
                raise
    
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
                embedding_model=configuration.value,
                chunking_strategy=ChunkingStrategy.FIXED_SIZE.value,  # Default for legacy function
            )
        )
    return embeddings


async def upload_documents(
    documents: list[SanityData],
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
):
    for doc in documents:
        await upload_document(doc, connection, embedding_configuration, chunking_strategy)


async def upload_document(
    doc: SanityData,
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
):
    logging.info(f"[run] Uploading new document: {doc.title} (id: {doc.id})...")
    await pre_process_uploaded_document(
        upload_request=doc,
        embedding_configuration=embedding_configuration,
        connection=connection,
        chunking_strategy=chunking_strategy,
    )
    logging.info(f"[run] Finished uploading document: {doc.title}")


async def update_document(
    document: SanityData,
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
):
    deleted = await delete_document(document.transcript_id, connection)
    if not deleted:
        return
    await upload_document(document, connection, embedding_configuration, chunking_strategy)


async def update_documents(
    documents: list[SanityData],
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
):
    for doc in documents:
        await update_document(
            doc, connection, embedding_configuration=embedding_configuration, chunking_strategy=chunking_strategy
        )


async def delete_documents(
    documents: list[SanityData], connection: EmbeddingConnection
):
    for doc in documents:
        await delete_document(doc.transcript_id, connection)


async def delete_document(sanity_data_id: str, connection: EmbeddingConnection) -> bool:
    return await connection.delete_document(sanity_data_id)
