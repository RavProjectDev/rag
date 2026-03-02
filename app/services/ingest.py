"""
Document ingest pipeline: fetch transcript → chunk → embed → insert.

Used by:
- app/services/sync_service/sync_db.py  (webhook + sync service)
- scripts/upload_manifest.py            (CLI bulk-upload script)
"""

import asyncio
import logging
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

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 10.0


async def fetch_transcript(transcript_url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(transcript_url)
    if not response.content:
        raise SRTFileNotFound
    return response.content.decode("utf-8")


def process_transcript_contents(
    title: str, raw_text: str, chunking_strategy: ChunkingStrategy
) -> list[Chunk]:
    raw_transcripts = [(title, raw_text)]
    return preprocess_raw_transcripts(raw_transcripts, chunking_strategy=chunking_strategy)


async def generate_and_insert_embeddings(
    chunks: list[Chunk],
    configuration: EmbeddingConfiguration,
    sanity_data: SanityData,
    connection: EmbeddingConnection,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
):
    """
    Generate embeddings for each chunk and insert immediately with retry/backoff.

    Retries on transient errors: EmbeddingTimeOutException, EmbeddingAPIException,
    InsertException.  Does NOT retry on EmbeddingConfigurationException.
    """
    total_chunks = len(chunks)
    logging.info(
        f"[INGEST] Starting incremental embed+insert for {total_chunks} chunks"
    )

    for idx, chunk in enumerate(chunks, 1):
        last_exception = None
        current_delay = INITIAL_RETRY_DELAY

        for attempt in range(MAX_RETRIES + 1):
            try:
                logging.info(
                    f"[INGEST] Chunk {idx}/{total_chunks}: attempt {attempt + 1}/{MAX_RETRIES + 1}"
                )

                data: Embedding = await generate_embedding(
                    text=chunk.text_to_embed,
                    configuration=configuration,
                    task_type="RETRIEVAL_DOCUMENT",
                )

                embedding = VectorEmbedding(
                    vector=data.vector,
                    dimension=len(data.vector),
                    metadata=chunk,
                    sanity_data=sanity_data,
                    embedding_model=configuration.value,
                    chunking_strategy=chunking_strategy.value,
                )

                await connection.insert([embedding])

                if idx % 10 == 0 or idx == total_chunks:
                    logging.info(
                        f"[INGEST] Progress: {idx}/{total_chunks} chunks embedded and inserted"
                    )
                break

            except EmbeddingConfigurationException as e:
                logging.error(
                    f"[INGEST ERROR] Configuration error for chunk {idx}/{total_chunks}: {e}"
                )
                raise

            except (EmbeddingTimeOutException, EmbeddingAPIException, InsertException) as e:
                last_exception = e
                logging.warning(
                    f"[INGEST RETRY] {type(e).__name__} on chunk {idx}/{total_chunks}, "
                    f"attempt {attempt + 1}/{MAX_RETRIES + 1}: {e}"
                )
                if attempt < MAX_RETRIES:
                    jitter = random.uniform(0, 0.1 * current_delay)
                    delay = min(current_delay + jitter, MAX_RETRY_DELAY)
                    logging.info(f"[INGEST RETRY] Waiting {delay:.2f}s before retry...")
                    await asyncio.sleep(delay)
                    current_delay *= 2
                else:
                    logging.error(
                        f"[INGEST ERROR] Max retries ({MAX_RETRIES}) exceeded for chunk {idx}/{total_chunks}"
                    )
                    raise last_exception

            except EmbeddingException as e:
                last_exception = e
                logging.warning(
                    f"[INGEST RETRY] EmbeddingException on chunk {idx}/{total_chunks}, "
                    f"attempt {attempt + 1}/{MAX_RETRIES + 1}: {e}"
                )
                if attempt < MAX_RETRIES:
                    jitter = random.uniform(0, 0.1 * current_delay)
                    delay = min(current_delay + jitter, MAX_RETRY_DELAY)
                    logging.info(f"[INGEST RETRY] Waiting {delay:.2f}s before retry...")
                    await asyncio.sleep(delay)
                    current_delay *= 2
                else:
                    logging.error(
                        f"[INGEST ERROR] Max retries ({MAX_RETRIES}) exceeded for chunk {idx}/{total_chunks}"
                    )
                    raise last_exception

            except Exception as e:
                logging.error(
                    f"[INGEST ERROR] Unexpected error for chunk {idx}/{total_chunks}: {e}",
                    exc_info=True,
                )
                raise

    logging.info(f"[INGEST] Completed embed+insert of all {total_chunks} chunks")


async def upload_document(
    doc: SanityData,
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
):
    logging.info(f"[INGEST] Uploading document: {doc.title} (id: {doc.id})...")
    contents = await fetch_transcript(str(doc.transcriptURL))
    chunks = process_transcript_contents(doc.title, contents, chunking_strategy)
    await generate_and_insert_embeddings(
        chunks, embedding_configuration, doc, connection, chunking_strategy
    )
    logging.info(f"[INGEST] Finished uploading document: {doc.title}")


async def upload_documents(
    documents: list[SanityData],
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
    chunking_strategy: ChunkingStrategy,
):
    for doc in documents:
        await upload_document(doc, connection, embedding_configuration, chunking_strategy)
