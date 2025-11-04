import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
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
from rag.app.services.prompts import PromptType

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
                name_spaces=chat_request.name_spaces,
                prompt_id=chat_request.prompt_type,
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
                import logging
                logger = logging.getLogger(__name__)
                
                llm_response = await get_llm_response(
                    metrics_connection=metrics_conn,
                    prompt=prompt.value,
                    model=llm_configuration,
                )
                # If we used the structured JSON prompt, validate JSON and return it
                if prompt.id == PromptType.STRUCTURED_JSON.value:
                    try:
                        parsed = json.loads(llm_response)
                        main_text = parsed.get("main_text", "")
                        sources_raw = parsed.get("sources", [])
                        
                        logger.info(f"[FULL RESPONSE] LLM returned {len(sources_raw)} extracted quotes")
                        
                        # Normalize sources into expected shape
                        sources = []
                        slug_counts = {}  # Track quotes per slug
                        
                        for idx, s in enumerate(sources_raw):
                            if not isinstance(s, dict):
                                continue
                            extracted_text = s.get("text", "")
                            slug = s.get("slug", "")
                            source_entry = {
                                "slug": slug,
                                "timestamp": s.get("timestamp"),
                                "text": extracted_text,
                            }
                            sources.append(source_entry)
                            
                            # Count quotes per slug
                            slug_counts[slug] = slug_counts.get(slug, 0) + 1
                            
                            # Log extraction quality
                            text_length = len(extracted_text)
                            word_count = len(extracted_text.split())
                            logger.info(
                                f"[FULL RESPONSE] Quote {idx+1}: slug={source_entry['slug']}, "
                                f"timestamp={source_entry['timestamp']}, "
                                f"text_length={text_length} chars, word_count={word_count} words"
                            )
                        
                        # Log distribution of quotes across sources
                        unique_slugs = len(slug_counts)
                        logger.info(
                            f"[FULL RESPONSE] Returning {len(sources)} total quotes from {unique_slugs} unique source(s)"
                        )
                        for slug, count in slug_counts.items():
                            logger.info(f"[FULL RESPONSE] Source '{slug}': {count} quote(s) extracted")
                        return ChatResponse(
                            main_text=main_text,
                            sources=sources,
                        )
                    except json.JSONDecodeError as e:
                        logger.error(f"[FULL RESPONSE ERROR] JSON decode error: {e}, response preview: {llm_response[:500]}")
                        raise HTTPException(
                            status_code=500,
                            detail={
                                "code": "invalid_llm_json",
                                "message": "LLM returned invalid JSON for FULL response",
                            },
                        )
                    except Exception as e:
                        logger.error(f"[FULL RESPONSE ERROR] Unexpected error: {e}", exc_info=True)
                        raise HTTPException(
                            status_code=500,
                            detail={
                                "code": "invalid_llm_json",
                                "message": "LLM returned invalid JSON for FULL response",
                            },
                        )
                else:
                    # Non-structured prompts fallback: map into ChatResponse
                    return ChatResponse(
                        main_text=llm_response,
                        sources=[],
                    )

        return await full_response()
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Timeout while waiting for chat request",
        )

    except NoDocumentFoundException as e:
        return ChatResponse(main_text=e.message_to_ui, sources=[])
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
    prompt_id: PromptType = PromptType.PRODUCTION,
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
    import logging
    logger = logging.getLogger(__name__)
    
    request_id = uuid.uuid4().hex
    logger.info(f"[GENERATE START] request_id={request_id}, question='{user_question}', name_spaces={name_spaces}, prompt_id={prompt_id}")
    
    # Preprocess question
    cleaned_question = pre_process_user_query(user_question)
    logger.info(f"[GENERATE] request_id={request_id}, cleaned_question='{cleaned_question}'")

    # Generate embedding with metrics logging
    embedding = None
    logger.info(f"[GENERATE] request_id={request_id}, Starting embedding generation with config={embedding_configuration}")
    
    async with metrics_connection.timed(
        metric_type="EMBEDDING", data={"request_id": request_id}
    ):
        try:
            embedding = await generate_embedding(
                text=cleaned_question,
                configuration=embedding_configuration,
            )
            logger.info(f"[GENERATE] request_id={request_id}, Embedding generated successfully, vector_dimension={len(embedding.vector) if embedding else 'None'}")

        except (
            EmbeddingException,
            EmbeddingConfigurationException,
            EmbeddingAPIException,
            EmbeddingTimeOutException,
        ) as e:
            logger.error(f"[GENERATE ERROR] request_id={request_id}, Embedding error: {e}")
            raise e
        except Exception as e:
            logger.error(f"[GENERATE ERROR] request_id={request_id}, Unexpected embedding error: {str(e)}", exc_info=True)
            raise EmbeddingException(f"Unexpected embedding error: {str(e)}")

        if embedding is None:
            logger.error(f"[GENERATE ERROR] request_id={request_id}, Embedding is None")
            raise EmbeddingException(
                f"Could not generate embedding for {user_question}"
            )

    # Retrieve matching documents with metrics logging
    vector: list[float] = embedding.vector
    data: list[DocumentModel] = []
    
    # Get collection details for logging
    collection_name = connection.collection.name if hasattr(connection, 'collection') else 'unknown'
    index_name = connection.index if hasattr(connection, 'index') else 'unknown'
    vector_path = connection.vector_path if hasattr(connection, 'vector_path') else 'unknown'
    
    logger.info(
        f"[GENERATE] request_id={request_id}, Starting document retrieval, "
        f"collection='{collection_name}', index='{index_name}', vector_path='{vector_path}', "
        f"vector_dimension={len(vector)}, name_spaces={name_spaces}"
    )
    
    async with metrics_connection.timed(
        metric_type="RETRIEVAL", data={"request_id": request_id}
    ):
        try:
            data = await connection.retrieve(
                embedded_data=vector, name_spaces=name_spaces
            )
            logger.info(
                f"[GENERATE] request_id={request_id}, Retrieved {len(data)} documents from database, "
                f"collection='{collection_name}', index='{index_name}'"
            )
            if data:
                logger.info(
                    f"[GENERATE] request_id={request_id}, Top document score: {data[0].score:.4f}, "
                    f"slug: {data[0].sanity_data.slug}, text_preview: {data[0].text[:100]}..."
                )
        except DataBaseException as e:
            logger.error(
                f"[GENERATE ERROR] request_id={request_id}, Database exception: {e}, "
                f"collection='{collection_name}', index='{index_name}', vector_path='{vector_path}', "
                f"name_spaces={name_spaces}"
            )
            raise
        except Exception as e:
            logger.error(
                f"[GENERATE ERROR] request_id={request_id}, Unexpected retrieval error: {str(e)}, "
                f"collection='{collection_name}', index='{index_name}', vector_path='{vector_path}'",
                exc_info=True
            )
            raise DataBaseException(f"Database retrieval failed: {str(e)}")

    logger.info(f"[GENERATE] request_id={request_id}, Generating prompt with {len(data)} documents, prompt_id={prompt_id}")
    
    try:
        prompt = generate_prompt(cleaned_question, data, prompt_id=prompt_id)
        logger.info(f"[GENERATE] request_id={request_id}, Prompt generated, length={len(prompt.value)} chars, prompt_id={prompt.id}")
    except LLMBaseException as e:
        logger.error(f"[GENERATE ERROR] request_id={request_id}, LLM exception: {e}")
        raise e
    except Exception as e:
        logger.error(f"[GENERATE ERROR] request_id={request_id}, Unexpected prompt generation error: {str(e)}", exc_info=True)
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

    logger.info(f"[GENERATE END] request_id={request_id}, Returning prompt and {len(transcript_data)} transcript_data items")
    return prompt, transcript_data
