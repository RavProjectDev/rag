import asyncio
import json

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse

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
from rag.app.schemas.response import ChatResponse, TranscriptData, ErrorResponse
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

    return ChatResponse(message=question, transcript_data=transcript_data)


async def helper(
    message: str, embedding_conn, embedding_configuration
) -> list[TranscriptData]:
    try:
        embedding = await generate_embedding(
            text=message,
            configuration=embedding_configuration,
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
            )
        )
    return transcript_data


@router.post(
    "/stream",
    summary="Mock streaming chat response",
    description="Simulates SSE stream for frontend testing; optionally triggers an error mid-stream.",
)
async def stream(request: Request):
    """
    Simulated streaming chat endpoint for frontend testing.

    This endpoint mimics the behavior of a production LLM streaming API
    by sending data chunks as Server-Sent Events (SSE). It allows frontend
    developers to test their streaming handling logic without incurring
    LLM costs.

    Request Body (JSON):
    --------------------
    - simulate_error (bool, optional):
        If true, the stream will intentionally raise an error partway through
        streaming to simulate failure scenarios. This helps test frontend
        error handling. Default is false.

        Example:
        {
            "simulate_error": true
        }

    Streaming Response:
    -------------------
    - Emits events in text/event-stream format.
    - The first event sends a JSON object with mock metadata:
        data: {"metadata": {...}}

    - Subsequent events send one chunk at a time:
        data: {"data": "Lorem"}

    - If simulate_error is enabled, after about 10 chunks:
        data: {"error": "Simulated streaming error occurred."}

    - The stream ends with:
        data: [DONE]

    Returns:
    --------
    StreamingResponse object emitting Server-Sent Events with simulated
    chunks of data, or a JSON error response in case of timeouts or
    unexpected exceptions.
    """
    event = None
    settings = get_settings()
    try:
        event = await request.json()

        simulate_error = event.get("simulate_error", False)

        async def inner_stream():
            chunks = [
                "Lorem",
                "ipsum",
                "dolor",
                "sit",
                "amet,",
                "consectetur",
                "adipiscing",
                "elit,",
                "sed",
                "do",
                "eiusmod",
                "tempor",
                "incididunt",
                "ut",
                "labore",
                "et",
                "dolore",
                "magna",
                "aliqua.",
                "Ut",
                "enim",
                "ad",
                "minim",
                "veniam,",
                "quis",
                "nostrud",
                "exercitation",
                "ullamco",
                "laboris",
                "nisi",
                "ut",
                "aliquip",
                "ex",
                "ea",
                "commodo",
                "consequat.",
                "Duis",
                "aute",
                "irure",
                "dolor",
                "in",
                "reprehenderit",
                "in",
                "voluptate",
                "velit",
                "esse",
                "cillum",
                "dolore",
                "eu",
                "fugiat",
                "nulla",
                "pariatur.",
                "Excepteur",
                "sint",
                "occaecat",
                "cupidatat",
                "non",
                "proident,",
                "sunt",
                "in",
                "culpa",
                "qui",
                "officia",
                "deserunt",
                "mollit",
                "anim",
                "id",
                "est",
                "laborum.",
            ]

            metadata = {"mock": True, "note": "This is a simulated streaming response."}

            async def event_generator():
                yield f"data: {json.dumps({'metadata': metadata})}\n\n"
                for i, chunk in enumerate(chunks):
                    # Simulate error after a few tokens if requested
                    if simulate_error and i == 10:
                        yield f"data: {json.dumps({'error': 'Simulated streaming error occurred.'})}\n\n"
                        raise Exception("Simulated streaming error occurred.")

                    yield f"data: {json.dumps({'data': chunk})}\n\n"
                    await asyncio.sleep(0.1)
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        return await asyncio.wait_for(
            inner_stream(), timeout=settings.external_api_timeout
        )
    # test
    except asyncio.TimeoutError:
        return JSONResponse(
            content={
                "error": f"Streaming request timed out after {settings.external_api_timeout} seconds."
            },
            status_code=500,
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
