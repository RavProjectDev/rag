import asyncio

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from rag.app.core.config import get_settings
from rag.app.db.connections import EmbeddingConnection
from rag.app.dependencies import get_embedding_conn, get_embedding_configuration
from rag.app.exceptions.db import DataBaseException
from rag.app.exceptions.embedding import (
    EmbeddingConfigurationException,
    EmbeddingException,
    EmbeddingAPIException,
    EmbeddingTimeOutException,
)
from rag.app.models.data import DocumentModel
from rag.app.schemas.data import EmbeddingConfiguration
from rag.app.schemas.requests import ChatRequest
from rag.app.schemas.response import ChatResponse, TranscriptData, ErrorResponse, SourceItem, UsedQuote
from rag.app.services.embedding import generate_embedding

router = APIRouter()


@router.post(
    "/full",
    response_model=ChatResponse,
    summary="Mock full chat response",
    description="Returns a delayed echo of the question with synthesized transcript data.",
)
async def stream(
    request: ChatRequest,
    embedding_conn: EmbeddingConnection = Depends(get_embedding_conn),
    embedding_configuration: EmbeddingConfiguration = Depends(
        get_embedding_configuration
    ),
) -> ChatResponse:
    await asyncio.sleep(5)
    question = request.question
    transcript_data = await helper(
        message=question,
        embedding_conn=embedding_conn,
        embedding_configuration=embedding_configuration,
    )

    return ChatResponse(
        main_text=question,
        sources=[
            SourceItem(
                slug=item.sanity_data.slug,
                text_id=item.text_id,
                full_text="Mock full text for testing purposes.",
                used_quotes=[
                    UsedQuote(
                        number=1,
                        text="Mock quote",
                        timestamp=(
                            f"{item.metadata.time_start}-{item.metadata.time_end}"
                            if item.metadata.time_start and item.metadata.time_end
                            else item.metadata.time_start or item.metadata.time_end
                        ),
                    )
                ],
                timestamp_range=(
                    f"{item.metadata.time_start}-{item.metadata.time_end}"
                    if item.metadata.time_start and item.metadata.time_end
                    else item.metadata.time_start or item.metadata.time_end
                ),
            )
            for item in transcript_data
        ],
    )


async def helper(
    message: str, embedding_conn, embedding_configuration
) -> list[TranscriptData]:
    try:
        embedding = await generate_embedding(
            text=message,
            configuration=embedding_configuration,
            task_type="RETRIEVAL_DOCUMENT",  # Changed to RETRIEVAL_DOCUMENT for Gemini
        )

    except (
        EmbeddingException,
        EmbeddingConfigurationException,
        EmbeddingAPIException,
        EmbeddingTimeOutException,
    ) as e:
        raise e
    except Exception as e:
        raise EmbeddingException(f"Unexpected embedding error: {str(e)}")
    if embedding is None:
        raise EmbeddingException(f"Could not generate embedding for {message}")

    vector: list[float] = embedding.vector
    data: list[DocumentModel]
    try:
        data = await embedding_conn.retrieve(embedded_data=vector)
    except DataBaseException:
        raise
    except Exception as e:
        raise DataBaseException(f"Database retrieval failed: {str(e)}")

    transcript_data: list[TranscriptData] = []
    for datum in data:
        transcript_data.append(
            TranscriptData(
                sanity_data=datum.sanity_data,
                metadata=datum.metadata,
                score=datum.score,
                text_id=datum.id,
            )
        )
    return transcript_data


