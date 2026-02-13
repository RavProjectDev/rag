import asyncio
import json
import logging
import uuid
import httpx

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette import status

from rag.app.core.config import get_settings
from rag.app.db.connections import EmbeddingConnection, MetricsConnection
from rag.app.dependencies import (
    get_embedding_conn,
    get_metrics_conn,
    get_embedding_configuration,
    get_llm_configuration,
    get_redis_conn,
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
from rag.app.schemas.requests import (
    ChatRequest,
    RetrieveDocumentsRequest,
    TypeOfRequest,
)
from rag.app.schemas.response import (
    ChatResponse,
    RetrieveDocumentsResponse,
    TranscriptData,
    ErrorResponse,
)
from rag.app.services.embedding import generate_embedding
from rag.app.services.llm import (
    stream_llm_response,
    generate_prompt,
    get_llm_response,
    get_chat_response_json_schema,
)
from rag.app.services.preprocess.user_input import pre_process_user_query
from rag.app.services.prompts import PromptType
from rag.app.services.auth import verify_jwt_token
from rag.app.middleware.rate_limit import rate_limit_middleware, user_rate_limit_middleware
from rag.app.db.redis_connection import RedisConnection

router = APIRouter()
logger = logging.getLogger(__name__)


async def submit_to_supabase(
    question: str,
    response: str,
    thread_id: uuid.UUID | None,
    user_id: str,
) -> str | None:
    """
    Submit query and response to Supabase RPC (blocking call).
    
    Args:
        question: The user's question
        response: The generated response (main_text or full JSON string)
        thread_id: Optional thread ID to append to
        user_id: The authenticated user's ID
    
    Returns:
        str: Thread ID (UUID as string) from Supabase RPC, or None if Supabase is not configured
    
    Raises:
        httpx.HTTPStatusError: If the RPC call fails
    """
    settings = get_settings()
    
    # Skip if Supabase is not configured
    if not settings.supabase_url or not settings.supabase_service_role_key:
        logger.warning("Supabase URL or service role key not configured, skipping query submission")
        return None
    
    try:
        async with httpx.AsyncClient() as client:
            response_data = await client.post(
                f"{settings.supabase_url}/rest/v1/rpc/submit_user_query",
                json={
                    "question_arg": question,
                    "response_arg": response,
                    "thread_id_arg": str(thread_id) if thread_id else None,
                    "thread_name_arg": "New Thread",
                    "user_id_arg": user_id,
                },
                headers={
                    "Content-Type": "application/json",
                    "apikey": settings.supabase_service_role_key,
                    "Authorization": f"Bearer {settings.supabase_service_role_key}",
                },
                timeout=10.0,  # 10 second timeout for Supabase call
            )
            response_data.raise_for_status()
            logger.info(
                f"Successfully submitted query to Supabase. "
                f"thread_id={thread_id}, question_length={len(question)}, "
                f"response_length={len(response)}"
            )
            return response_data.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Supabase RPC call failed with status {e.response.status_code}: {e.response.text}"
        )
        raise
    except Exception as e:
        logger.error(f"Unexpected error submitting to Supabase: {e}", exc_info=True)
        raise


@router.post(
    "/documents",
    response_model=RetrieveDocumentsResponse,
    summary="Retrieve top-k documents",
    description=(
        "Runs the preprocessing, embedding, and vector search pipeline "
        "to return the top-k matching documents."
    ),
)
async def retrieve_documents_handler(
    request: RetrieveDocumentsRequest,
    user_id: str = Depends(verify_jwt_token),
    embedding_conn: EmbeddingConnection = Depends(get_embedding_conn),
    metrics_conn: MetricsConnection = Depends(get_metrics_conn),
    embedding_configuration: EmbeddingConfiguration = Depends(
        get_embedding_configuration
    ),
) -> RetrieveDocumentsResponse:
    """
    Retrieve top-k relevant documents for a question without invoking the LLM.
    """
    settings = get_settings()
    request_id = uuid.uuid4().hex

    try:
        cleaned_question, documents, transcript_data = await asyncio.wait_for(
            retrieve_relevant_documents(
                user_question=request.question,
                embedding_configuration=embedding_configuration,
                connection=embedding_conn,
                metrics_connection=metrics_conn,
                name_spaces=request.name_spaces,
                top_k=request.top_k,
                request_id=request_id,
                pinecone_index=request.pinecone_index,
                pinecone_namespace=request.pinecone_namespace,
            ),
            timeout=settings.external_api_timeout,
        )

        return RetrieveDocumentsResponse(
            request_id=request_id,
            cleaned_question=cleaned_question,
            requested_top_k=request.top_k,
            documents=documents,
            transcript_data=transcript_data,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Timeout while retrieving documents",
        )
    except NoDocumentFoundException as e:
        cleaned_question = pre_process_user_query(request.question)
        return RetrieveDocumentsResponse(
            request_id=request_id,
            cleaned_question=cleaned_question,
            requested_top_k=request.top_k,
            documents=[],
            transcript_data=[],
            message=e.message_to_ui,
        )
    except BaseAppException as e:
        raise HTTPException(
            status_code=e.status_code, detail={"code": e.code, "error": e.message}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"code": "internal_server_error", "message": str(e)}
        )


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
    request: Request,
    chat_request: ChatRequest,
    user_id: str = Depends(verify_jwt_token),
    embedding_conn: EmbeddingConnection = Depends(get_embedding_conn),
    metrics_conn: MetricsConnection = Depends(get_metrics_conn),
    embedding_configuration: EmbeddingConfiguration = Depends(
        get_embedding_configuration
    ),
    llm_configuration: LLMModel = Depends(get_llm_configuration),
    redis_conn: RedisConnection | None = Depends(get_redis_conn),
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
    
    # Apply rate limiting if Redis is available
    if redis_conn:
        # Global rate limiting (per endpoint)
        await rate_limit_middleware(
            request=request,
            redis_conn=redis_conn,
            limit=settings.rate_limit_max_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        
        # User-based rate limiting (monthly reset)
        await user_rate_limit_middleware(
            user_id=user_id,
            redis_conn=redis_conn,
            limit=settings.user_rate_limit_max_requests_per_month,
            request=request,
        )
    
    # Flag to track if we should decrement rate limit on error
    should_decrement_on_error = redis_conn is not None

    try:
        # Generate prompt and metadata
        transcript_data: list[TranscriptData]
        retrieved_docs: list[DocumentModel]
        source_list: list[dict]

        prompt, transcript_data, retrieved_docs, source_list = await asyncio.wait_for(
            generate(
                user_question=chat_request.question,
                embedding_configuration=embedding_configuration,
                connection=embedding_conn,
                metrics_connection=metrics_conn,
                name_spaces=chat_request.name_spaces,
                prompt_id=chat_request.prompt_type,
                pinecone_index=chat_request.pinecone_index,
                pinecone_namespace=chat_request.pinecone_namespace,
            ),
            timeout=settings.external_api_timeout,
        )
        if chat_request.type_of_request == TypeOfRequest.STREAM:
            # Use schema enforcement for STRUCTURED_JSON prompts
            # Check if any documents have timestamps to determine if timestamp should be required
            has_timestamps = any(
                item.metadata.time_start is not None or item.metadata.time_end is not None
                for item in transcript_data
            )
            response_format = (
                get_chat_response_json_schema(require_timestamp=has_timestamps)
                if prompt.id == PromptType.STRUCTURED_JSON.value
                else None
            )

            async def event_generator():
                """Asynchronous generator for Server-Sent Events (SSE)."""
                yield f"data: {json.dumps({'transcript_data': [item.to_dict() for item in transcript_data]})}\n\n"

                async for chunk in stream_llm_response(
                    metrics_connection=metrics_conn,
                    prompt=prompt.value,
                    model=llm_configuration.value,
                    response_format=response_format,
                ):
                    yield f"data: {json.dumps({'data': chunk})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        else:
            # Generate full response
            async def full_response():
                import logging
                logger = logging.getLogger(__name__)
                
                # Use schema enforcement for STRUCTURED_JSON prompts
                # Check if any documents have timestamps to determine if timestamp should be required
                has_timestamps = any(
                    item.metadata.time_start is not None or item.metadata.time_end is not None
                    for item in transcript_data
                )
                response_format = (
                    get_chat_response_json_schema(require_timestamp=has_timestamps)
                    if prompt.id == PromptType.STRUCTURED_JSON.value
                    else None
                )
                
                llm_response = await get_llm_response(
                    metrics_connection=metrics_conn,
                    prompt=prompt.value,
                    model=llm_configuration,
                    response_format=response_format,
                )
                # If we used the structured JSON prompt, validate JSON and return it
                if prompt.id == PromptType.STRUCTURED_JSON.value:
                    try:
                        parsed = json.loads(llm_response)
                        main_text = parsed.get("main_text", "")
                        source_numbers = parsed.get("source_numbers", [])
                        
                        logger.info(f"[FULL RESPONSE] LLM returned {len(source_numbers)} source numbers: {source_numbers}")
                        
                        # Build source lookup by number
                        source_by_number = {src["number"]: src for src in source_list}
                        
                        # Group sources by text_id (parent document)
                        sources_by_doc = {}
                        for num in source_numbers:
                            source = source_by_number.get(num)
                            if source is None:
                                logger.warning(
                                    f"[FULL RESPONSE] LLM referenced source number {num} but it doesn't exist. "
                                    f"Available numbers: {list(source_by_number.keys())[:10]}"
                                )
                                continue
                            
                            text_id = source["text_id"]
                            if text_id not in sources_by_doc:
                                sources_by_doc[text_id] = []
                            sources_by_doc[text_id].append(source)
                        
                        # Build sources with full context and bolded used quotes
                        sources = []
                        slug_counts = {}  # Track quotes per slug
                        
                        for text_id, used_sources in sources_by_doc.items():
                            # Find the original document from retrieved_docs
                            original_doc = next((d for d in retrieved_docs if d.id == text_id), None)
                            if not original_doc:
                                logger.warning(f"[FULL RESPONSE] Could not find original document for text_id={text_id}")
                                continue
                            
                            # Get all text segments from the document
                            all_segments = original_doc.text if isinstance(original_doc.text, list) else [(original_doc.text, None)]
                            
                            # Build set of used text for quick lookup
                            used_texts = {s["text"] for s in used_sources}
                            
                            # Reconstruct full text with bolded used segments
                            reconstructed_text_parts = []
                            min_time = None
                            max_time = None
                            
                            for segment in all_segments:
                                if isinstance(segment, (list, tuple)) and len(segment) >= 1:
                                    text_content = segment[0]
                                    
                                    # Track timestamp range
                                    if len(segment) >= 2 and isinstance(segment[1], (list, tuple)) and len(segment[1]) >= 2:
                                        time_start = segment[1][0]
                                        time_end = segment[1][1]
                                        if time_start:
                                            if min_time is None or time_start < min_time:
                                                min_time = time_start
                                        if time_end:
                                            if max_time is None or time_end > max_time:
                                                max_time = time_end
                                    
                                    # Check if this segment was used and bolden it
                                    if text_content in used_texts:
                                        reconstructed_text_parts.append(f"**{text_content}**")
                                    else:
                                        reconstructed_text_parts.append(text_content)
                                else:
                                    reconstructed_text_parts.append(str(segment))
                            
                            # Join all parts
                            full_text_with_highlights = " ".join(reconstructed_text_parts)
                            
                            # Create timestamp range
                            timestamp_range = None
                            if min_time and max_time:
                                timestamp_range = f"{min_time}-{max_time}"
                            elif min_time:
                                timestamp_range = min_time
                            elif max_time:
                                timestamp_range = max_time
                            
                            # Create list of used quotes
                            used_quotes = [
                                {
                                    "number": s["number"],
                                    "text": s["text"],
                                    "timestamp": s["timestamp"]
                                }
                                for s in used_sources
                            ]
                            
                            # Create source entry with full context
                            source_entry = {
                                "slug": original_doc.sanity_data.slug,
                                "text_id": text_id,
                                "full_text": full_text_with_highlights,
                                "used_quotes": used_quotes,
                                "timestamp_range": timestamp_range,
                            }
                            sources.append(source_entry)
                            
                            # Count quotes per slug
                            slug = original_doc.sanity_data.slug
                            slug_counts[slug] = slug_counts.get(slug, 0) + len(used_sources)
                            
                            # Log source details
                            logger.info(
                                f"[FULL RESPONSE] Document text_id={text_id}, slug={slug}, "
                                f"used_quotes={len(used_sources)}, full_text_length={len(full_text_with_highlights)} chars"
                            )
                        
                        # Log distribution of quotes across sources
                        unique_slugs = len(slug_counts)
                        logger.info(
                            f"[FULL RESPONSE] Returning {len(sources)} documents with {sum(slug_counts.values())} total quotes from {unique_slugs} unique transcript(s)"
                        )
                        for slug, count in slug_counts.items():
                            logger.info(f"[FULL RESPONSE] Transcript '{slug}': {count} quote(s)")
                        
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

            # Generate the response
            response = await full_response()
            
            # Submit to Supabase if requested
            returned_thread_id = None
            if chat_request.submit_query:
                try:
                    # Prepare response string for Supabase
                    # Use Pydantic's model_dump_json() to properly serialize all nested models
                    response_str = response.model_dump_json()
                    
                    result = await submit_to_supabase(
                        question=chat_request.question,
                        response=response_str,
                        thread_id=chat_request.thread_id,
                        user_id=user_id,
                    )
                    # Extract thread_id from Supabase response
                    # Supabase returns the UUID as a string, convert to UUID object
                    if result:
                        returned_thread_id = uuid.UUID(result) if isinstance(result, str) else result
                    
                    logger.info(
                        f"Query submitted to Supabase successfully. "
                        f"thread_id={returned_thread_id}, "
                        f"was_new_thread={chat_request.thread_id is None}"
                    )
                except Exception as e:
                    # Log error but don't fail the request - user still gets their response
                    logger.error(
                        f"Failed to submit query to Supabase: {e}. "
                        f"Continuing with response. question='{chat_request.question[:50]}...'"
                    )
            
            # Add thread_id to response
            response.thread_id = returned_thread_id
            
            return response
    except asyncio.TimeoutError:
        # Decrement rate limit on timeout error
        if should_decrement_on_error:
            try:
                await redis_conn.decrement_user_rate_limit(user_id)
            except Exception as decr_err:
                logger.error(f"Failed to decrement rate limit for user_id={user_id}: {decr_err}")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Timeout while waiting for chat request",
        )

    except NoDocumentFoundException as e:
        # Decrement rate limit when no documents found (generic message case)
        if should_decrement_on_error:
            try:
                await redis_conn.decrement_user_rate_limit(user_id)
            except Exception as decr_err:
                logger.error(f"Failed to decrement rate limit for user_id={user_id}: {decr_err}")
        return ChatResponse(main_text=e.message_to_ui, sources=[])
    except BaseAppException as e:
        # Decrement rate limit on application exceptions
        if should_decrement_on_error:
            try:
                await redis_conn.decrement_user_rate_limit(user_id)
            except Exception as decr_err:
                logger.error(f"Failed to decrement rate limit for user_id={user_id}: {decr_err}")
        raise HTTPException(
            status_code=e.status_code, detail={"code": e.code, "error": e.message}
        )
    except Exception as e:
        # Decrement rate limit on unexpected exceptions
        if should_decrement_on_error:
            try:
                await redis_conn.decrement_user_rate_limit(user_id)
            except Exception as decr_err:
                logger.error(f"Failed to decrement rate limit for user_id={user_id}: {decr_err}")
        raise HTTPException(
            status_code=500, detail={"code": "internal_server_error", "message": str(e)}
        )


async def retrieve_relevant_documents(
    user_question: str,
    embedding_configuration: EmbeddingConfiguration,
    connection: EmbeddingConnection,
    metrics_connection: MetricsConnection,
    name_spaces: list[str] | None = None,
    request_id: str | None = None,
    top_k: int | None = None,
    pinecone_index: str | None = None,
    pinecone_namespace: str | None = None,
) -> tuple[str, list[DocumentModel], list[TranscriptData]]:
    """
    Shared pipeline that performs preprocessing, embedding generation,
    and vector retrieval to produce context documents.
    """
    if request_id is None:
        request_id = uuid.uuid4().hex

    logger.info(
        f"[GENERATE START] request_id={request_id}, question='{user_question}', "
        f"name_spaces={name_spaces}"
    )

    cleaned_question = pre_process_user_query(user_question)
    logger.info(
        f"[GENERATE] request_id={request_id}, cleaned_question='{cleaned_question}'"
    )

    embedding = None
    logger.info(
        f"[GENERATE] request_id={request_id}, Starting embedding generation with "
        f"config={embedding_configuration}"
    )

    async with metrics_connection.timed(
        metric_type="EMBEDDING", data={"request_id": request_id}
    ):
        try:
            embedding = await generate_embedding(
                text=cleaned_question,
                configuration=embedding_configuration,
                task_type="RETRIEVAL_DOCUMENT",  # Changed to RETRIEVAL_DOCUMENT for Gemini
            )
            logger.info(
                f"[GENERATE] request_id={request_id}, Embedding generated successfully, "
                f"vector_dimension={len(embedding.vector) if embedding else 'None'}"
            )
        except (
            EmbeddingException,
            EmbeddingConfigurationException,
            EmbeddingAPIException,
            EmbeddingTimeOutException,
        ) as e:
            logger.error(f"[GENERATE ERROR] request_id={request_id}, Embedding error: {e}")
            raise e
        except Exception as e:
            logger.error(
                f"[GENERATE ERROR] request_id={request_id}, Unexpected embedding error: {str(e)}",
                exc_info=True,
            )
            raise EmbeddingException(f"Unexpected embedding error: {str(e)}")

        if embedding is None:
            logger.error(f"[GENERATE ERROR] request_id={request_id}, Embedding is None")
            raise EmbeddingException(f"Could not generate embedding for {user_question}")

    vector: list[float] = embedding.vector
    data: list[DocumentModel] = []

    collection_name = (
        connection.collection.name if hasattr(connection, "collection") else "unknown"
    )
    index_name = connection.index if hasattr(connection, "index") else "unknown"
    vector_path = (
        connection.vector_path if hasattr(connection, "vector_path") else "unknown"
    )

    logger.info(
        f"[GENERATE] request_id={request_id}, Starting document retrieval, "
        f"collection='{collection_name}', index='{index_name}', vector_path='{vector_path}', "
        f"vector_dimension={len(vector)}, name_spaces={name_spaces}, top_k={top_k}"
    )

    retrieve_kwargs = {}
    if top_k is not None:
        retrieve_kwargs["k"] = top_k
    if pinecone_index is not None:
        retrieve_kwargs["index_override"] = pinecone_index
    if pinecone_namespace is not None:
        retrieve_kwargs["namespace_override"] = pinecone_namespace

    async with metrics_connection.timed(
        metric_type="RETRIEVAL", data={"request_id": request_id}
    ):
        try:
            data = await connection.retrieve(
                embedded_data=vector, name_spaces=name_spaces, **retrieve_kwargs,
            )
            logger.info(
                f"[GENERATE] request_id={request_id}, Retrieved {len(data)} documents from database, "
                f"collection='{collection_name}', index='{index_name}'"
            )
            if data:
                preview_source = (
                    " ".join(part[0] for part in data[0].text)[:100]
                    if isinstance(data[0].text, list)
                    else str(data[0].text)[:100]
                )
                logger.info(
                    f"[GENERATE] request_id={request_id}, Top document score: {data[0].score:.4f}, "
                    f"slug: {data[0].sanity_data.slug}, text_preview: {preview_source}..."
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
                exc_info=True,
            )
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

    logger.info(
        f"[GENERATE END] request_id={request_id}, Retrieved {len(transcript_data)} transcript_data items"
    )
    return cleaned_question, data, transcript_data


async def generate(
    user_question: str,
    embedding_configuration: EmbeddingConfiguration,
    connection: EmbeddingConnection,
    metrics_connection: MetricsConnection,
    name_spaces: list[str] = None,
    prompt_id: PromptType = PromptType.LIGHT,
    pinecone_index: str | None = None,
    pinecone_namespace: str | None = None,
) -> tuple[Prompt, list[TranscriptData], list[DocumentModel], list[dict]]:
    """
    Generate an LLM prompt and retrieve relevant context.

    Args:
        user_question: The question inputted by user.
        embedding_configuration: Configuration for generating embeddings.
        connection: Database connection for retrieving documents.
        metrics_connection: Connection for logging metrics.
        name_spaces: Name spaces optional.
        pinecone_index: Optional Pinecone index name override.
        pinecone_namespace: Optional Pinecone namespace override.

    Returns:
        Tuple[str, List[dict]]: The generated prompt and list of metadata.

    Raises:
        InputValidationError: If the user question is empty.
        EmbeddingException: For embedding-related errors.
        DataBaseException: For database retrieval errors.
        LLMException: For prompt generation errors.
    """
    request_id = uuid.uuid4().hex
    logger.info(f"[GENERATE START] request_id={request_id}, question='{user_question}', name_spaces={name_spaces}, prompt_id={prompt_id}, pinecone_index={pinecone_index}, pinecone_namespace={pinecone_namespace}")
    cleaned_question, data, transcript_data = await retrieve_relevant_documents(
        user_question=user_question,
        embedding_configuration=embedding_configuration,
        connection=connection,
        metrics_connection=metrics_connection,
        name_spaces=name_spaces,
        request_id=request_id,
        pinecone_index=pinecone_index,
        pinecone_namespace=pinecone_namespace,
    )

    logger.info(f"[GENERATE] request_id={request_id}, Generating prompt with {len(data)} documents, prompt_id={prompt_id}")
    
    try:
        prompt, source_list = generate_prompt(cleaned_question, data, prompt_id=prompt_id, request_id=request_id)
        logger.info(f"[GENERATE] request_id={request_id}, Prompt generated, length={len(prompt.value)} chars, prompt_id={prompt.id}, sources={len(source_list)}")
    except LLMBaseException as e:
        logger.error(f"[GENERATE ERROR] request_id={request_id}, LLM exception: {e}")
        raise e
    except Exception as e:
        logger.error(f"[GENERATE ERROR] request_id={request_id}, Unexpected prompt generation error: {str(e)}", exc_info=True)
        raise LLMBaseException(str(e))
    
    logger.info(f"[GENERATE END] request_id={request_id}, Returning prompt and {len(transcript_data)} transcript_data items")
    return prompt, transcript_data, data, source_list
