import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from starlette import status

from rag.app.core.config import get_settings
from rag.app.db.connections import EmbeddingConnection, MetricsConnection
from rag.app.dependencies import (
    get_embedding_conn,
    get_metrics_conn,
    get_embedding_configuration,
    get_llm_configuration,
)
from rag.app.exceptions.base import BaseAppException
from rag.app.exceptions.db import DataBaseException, NoDocumentFoundException
from rag.app.exceptions.embedding import (
    EmbeddingConfigurationException,
    EmbeddingException,
    EmbeddingTimeOutException,
    EmbeddingAPIException,
)
from rag.app.exceptions.llm import (
    LLMBaseException,
)
from rag.app.models.data import DocumentModel, Prompt
from rag.app.schemas.data import EmbeddingConfiguration, LLMModel
from rag.app.schemas.requests import ChatRequest, TypeOfRequest
from rag.app.schemas.response import ChatResponse, TranscriptData, ErrorResponse
from rag.app.services.embedding import generate_embedding
from rag.app.services.llm import stream_llm_response, generate_prompt, get_llm_response
from rag.app.services.preprocess.user_input import pre_process_user_query

router = APIRouter()


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Create chat completion (streaming or full)",
    description=(
        "Returns a full completion or a Server-Sent Events stream based on type_of_request.\n"
        "SSE emits 'transcript_data' first, then incremental 'data' tokens, and terminates with [DONE]."
    ),
    responses={
        200: {
            "description": "Returns ChatResponse (JSON) or StreamingResponse (SSE)",
            "content": {
                "text/event-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        },
        400: {"model": ErrorResponse, "description": "Bad request"},
        408: {"model": ErrorResponse, "description": "Timeout"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def handler(
    chat_request: ChatRequest,
    embedding_conn: EmbeddingConnection = Depends(get_embedding_conn),
    metrics_conn: MetricsConnection = Depends(get_metrics_conn),
    embedding_configuration: EmbeddingConfiguration = Depends(
        get_embedding_configuration
    ),
    llm_configuration: LLMModel = Depends(get_llm_configuration),
) -> ChatResponse | StreamingResponse:
    """
    Asynchronous chat completion endpoint for handling streaming and non-streaming chat requests.

    Args:
        chat_request: The validated chat request model.
        embedding_conn: Database connection for embeddings.
        metrics_conn: Connection for logging metrics.
        embedding_configuration: Configuration for generating embeddings.
        llm_configuration: LLM model configuration.

    Returns:
        ChatResponse | StreamingResponse: Non-streaming or streaming response.

    Raises:
        HTTPException: For validation, database, embedding, LLM, or unexpected errors.
    """
    settings = get_settings()

    try:
        # Generate prompt and metadata
        transcript_data: list[TranscriptData]

        prompt, transcript_data = await asyncio.wait_for(
            generate(
                user_question=chat_request.question,
                embedding_configuration=embedding_configuration,
                connection=embedding_conn,
                metrics_connection=metrics_conn,
                name_spaces=chat_request.name_spaces,  # Pass metrics_conn for logging
            ),
            timeout=settings.external_api_timeout,
        )
        if chat_request.type_of_request == TypeOfRequest.STREAM:

            async def event_generator():
                """Asynchronous generator for Server-Sent Events (SSE)."""
                yield f"data: {json.dumps({'transcript_data': [item.to_dict() for item in transcript_data]})}\n\n"

                async for chunk in stream_llm_response(
                    metrics_connection=metrics_conn,
                    prompt=prompt.value,
                    model=llm_configuration.value,
                ):
                    yield f"data: {json.dumps({'data': chunk})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        else:

            async def full_response():
                llm_response = await get_llm_response(
                    metrics_connection=metrics_conn,
                    prompt=prompt.value,
                    model=llm_configuration,
                )
                return ChatResponse(
                    message=llm_response,
                    transcript_data=transcript_data,
                    prompt_id=prompt.id,
                )

        return await full_response()
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Timeout while waiting for chat request",
        )

    except NoDocumentFoundException as e:
        return ChatResponse(message=e.message_to_ui, transcript_data=[])
    except BaseAppException as e:
        raise HTTPException(
            status_code=e.status_code, detail={"code": e.code, "error": e.message}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"code": "internal_server_error", "message": str(e)}
        )


async def generate(
    user_question: str,
    embedding_configuration: EmbeddingConfiguration,
    connection: EmbeddingConnection,
    metrics_connection: MetricsConnection,
    name_spaces: list[str] = None,
) -> (Prompt, list[TranscriptData]):
    """
    Generate an LLM prompt and retrieve relevant context.

    Args:
        user_question: The question inputted by user.
        embedding_configuration: Configuration for generating embeddings.
        connection: Database connection for retrieving documents.
        metrics_connection: Connection for logging metrics.
        name_spaces: Name spaces optional.

    Returns:
        Tuple[str, List[dict]]: The generated prompt and list of metadata.

    Raises:
        InputValidationError: If the user question is empty.
        EmbeddingException: For embedding-related errors.
        DataBaseException: For database retrieval errors.
        LLMException: For prompt generation errors.
    """
    request_id = uuid.uuid4().hex
    # Preprocess question
    cleaned_question = pre_process_user_query(user_question)

    # Generate embedding with metrics logging
    embedding = None
    async with metrics_connection.timed(
        metric_type="EMBEDDING", data={"request_id": request_id}
    ):
        try:
            embedding = await generate_embedding(
                text=cleaned_question,
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
            raise EmbeddingException(
                f"Could not generate embedding for {user_question}"
            )

    # Retrieve matching documents with metrics logging
    vector: list[float] = embedding.vector
    data: list[DocumentModel] = []
    async with metrics_connection.timed(
        metric_type="RETRIEVAL", data={"request_id": request_id}
    ):
        try:
            data = await connection.retrieve(
                embedded_data=vector, name_spaces=name_spaces
            )
        except DataBaseException:
            raise
        except Exception as e:
            raise DataBaseException(f"Database retrieval failed: {str(e)}")

    try:
        prompt = generate_prompt(cleaned_question, data)
    except LLMBaseException as e:
        raise e
    except Exception as e:
        raise LLMBaseException(str(e))
    transcript_data: list[TranscriptData] = []
    for datum in data:
        transcript_data.append(
            TranscriptData(
                sanity_data=datum.sanity_data,
                metadata=datum.metadata,
                score=datum.score,
            )
        )

    return prompt, transcript_data
