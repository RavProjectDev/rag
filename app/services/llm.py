import asyncio
import logging
from contextlib import nullcontext
from functools import lru_cache
from typing import Union, Optional, Dict, Any

from openai import (
    AsyncOpenAI,
    APIError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    OpenAIError,
)
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
)

from rag.app.core.config import get_settings
from rag.app.db.connections import MetricsConnection
from rag.app.exceptions.llm import (
    LLMBaseException,
    LLMTimeoutException,
    LLMConnectionException,
)
from rag.app.models.data import DocumentModel, Prompt
from rag.app.schemas.data import LLMModel
from rag.app.services.prompts import (
    PROMPTS,
    PromptType,
    resolve_prompt_key,
)


@lru_cache()
def get_openai_client() -> AsyncOpenAI:
    try:
        settings = get_settings()
    except (AttributeError, KeyError, TypeError) as e:
        raise LLMBaseException("Failed to get data from settings")
    try:
        return AsyncOpenAI(api_key=settings.openai_api_key)
    except (OpenAIError, AuthenticationError) as e:
        raise LLMConnectionException("OpenAI API connection failed: {}".format(e))


def get_chat_response_json_schema(require_timestamp: bool = False) -> Dict[str, Any]:
    """
    Returns the JSON schema for structured chat responses.
    This schema enforces the structure expected by the chat endpoint.
    
    Args:
        require_timestamp: Not used anymore, kept for backwards compatibility.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "chat_response",
            "schema": {
                "type": "object",
                "properties": {
                    "main_text": {
                        "type": "string",
                        "description": "A comprehensive response that directly answers the user's question"
                    },
                    "source_numbers": {
                        "type": "array",
                        "description": "Array of source numbers (integers) referenced in the main_text",
                        "items": {
                            "type": "integer",
                            "description": "The number of a source from the context (e.g., 1, 2, 3)"
                        }
                    }
                },
                "required": ["main_text", "source_numbers"],
                "additionalProperties": False
            },
            "strict": False
        }
    }


# ---------------------------------------------------------------
# Single-response LLM call
# ---------------------------------------------------------------


async def get_llm_response(
    prompt: str,
    model: LLMModel = LLMModel.GPT_4,
    metrics_connection: MetricsConnection = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Fetches a synchronous completion from the LLM.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The LLM model to use
        metrics_connection: Optional metrics connection for tracking
        response_format: Optional response format for structured outputs (e.g., JSON schema)
    """
    data = {}
    cm = (
        metrics_connection.timed(metric_type="LLM", data=data)
        if metrics_connection
        else nullcontext()
    )

    async with cm:
        if model == LLMModel.GPT_4:
            try:
                response, metrics = await get_gpt_response(
                    prompt=prompt, model=model.value, response_format=response_format
                )
            except Exception as e:
                logging.error(f"Error in get_gpt_response: {e}")
                raise
        elif model == LLMModel.MOCK:
            response, metrics = get_mock_response()
        else:
            raise ValueError(f"Unsupported model: {model}")

    data.update(metrics or {})
    return response


# ---------------------------------------------------------------
# GPT-4 call
# ---------------------------------------------------------------


async def get_gpt_response(
    prompt: str,
    model: str,
    response_format: Optional[Dict[str, Any]] = None,
) -> tuple[str, dict | None]:
    """
    Fetches a completion from GPT.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model name to use
        response_format: Optional response format for structured outputs (e.g., JSON schema)
    """
    try:
        client = get_openai_client()
        messages: list[
            Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]
        ] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="You are a helpful assistant knowledgeable in Rav Soloveitchik's teachings.",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=prompt,
            ),
        ]
        
        create_kwargs = {
            "model": model,
            "messages": messages,
        }
        
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        
        try:
            response = await client.chat.completions.create(**create_kwargs)
        except OpenAIError as e:
            raise LLMBaseException(f"Failed to get data from OpenAI: {e}")

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        used_model = response.model

        result = response.choices[0].message.content
        if not result:
            return "Error: Received null response from OpenAI", None

        metrics = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_model": model,
            "model": used_model,
        }
        return result, metrics

    except Exception as e:
        logging.error(f"Error in get_gpt_response: {e}")
        raise


# ---------------------------------------------------------------
# Mock LLM response
# ---------------------------------------------------------------


def get_mock_response() -> tuple[str, dict]:
    """
    Returns a fixed lorem ipsum mock response and empty metrics.
    """
    return (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        {},
    )


# ---------------------------------------------------------------
# Streaming LLM call
# ---------------------------------------------------------------


async def stream_llm_response(
    metrics_connection: MetricsConnection,
    prompt: str,
    model: str = "gpt-4",
    response_format: Optional[Dict[str, Any]] = None,
):
    """
    Streams completion from the LLM as text chunks.
    
    Args:
        metrics_connection: Metrics connection for tracking
        prompt: The prompt to send to the LLM
        model: The model name to use
        response_format: Optional response format for structured outputs (e.g., JSON schema)
    """
    client = get_openai_client()
    messages = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful assistant knowledgeable in Rav Soloveitchik's teachings.",
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=prompt,
        ),
    ]
    settings = get_settings()

    try:
        create_kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        
        # Make the async streaming API call with timeout
        response = await asyncio.wait_for(
            client.chat.completions.create(**create_kwargs),
            timeout=settings.external_api_timeout,
        )

        async def _consume_stream():
            async for chunk in response:
                try:
                    if not chunk.choices or not chunk.choices[0].delta:
                        yield "Error: Invalid chunk structure"
                        return
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                except (AttributeError, IndexError):
                    yield "Error: Invalid chunk structure"
                    return
            yield "[DONE]"

        # Stream with metrics and per-token timeout
        async with metrics_connection.timed(metric_type="LLM_STREAM", data={}):
            stream = _consume_stream()
            while True:
                try:
                    token = await asyncio.wait_for(
                        stream.__anext__(), timeout=settings.external_api_timeout
                    )
                    yield token
                except StopAsyncIteration:
                    break

    except asyncio.TimeoutError:
        raise LLMTimeoutException(
            f"LLM call timed out after {settings.external_api_timeout} seconds"
        )
    except AuthenticationError:
        raise LLMConnectionException("Invalid OpenAI API key")
    except RateLimitError:
        raise LLMConnectionException("OpenAI rate limit exceeded")
    except APIConnectionError:
        raise LLMConnectionException("Failed to connect to OpenAI API")
    except APIError as e:
        raise LLMConnectionException(f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise LLMBaseException(f"Unexpected error: {str(e)}")


# ---------------------------------------------------------------
# Embedding Generation (incl. mock logic)
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Prompt Generation
# ---------------------------------------------------------------


def _write_prompt_to_dev_outputs(
    filled_prompt: str,
    user_question: str,
    prompt_id: str,
    request_id: str | None,
    num_docs: int,
    source_list: list[dict],
) -> None:
    """
    Write the filled prompt to dev_outputs/ directory for debugging.
    
    Args:
        filled_prompt: The complete prompt with context
        user_question: The original user question
        prompt_id: The prompt template ID used
        request_id: Optional request ID for the filename
        num_docs: Number of documents in the context
        source_list: List of source references
    """
    import json
    from pathlib import Path
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create dev_outputs directory if it doesn't exist
        dev_outputs_dir = Path(__file__).parent.parent.parent / "dev_outputs"
        dev_outputs_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp and request_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        req_id_part = f"_{request_id[:8]}" if request_id else ""
        filename = f"prompt_{timestamp}{req_id_part}.txt"
        filepath = dev_outputs_dir / filename
        
        # Prepare metadata
        metadata = {
            "timestamp": timestamp,
            "request_id": request_id,
            "user_question": user_question,
            "prompt_id": prompt_id,
            "num_documents": num_docs,
            "num_sources": len(source_list),
            "prompt_length": len(filled_prompt),
        }
        
        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("DEBUG OUTPUT - LLM PROMPT WITH CONTEXT\n")
            f.write("=" * 80 + "\n\n")
            f.write("METADATA:\n")
            f.write(json.dumps(metadata, indent=2))
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("FULL PROMPT:\n")
            f.write("=" * 80 + "\n\n")
            f.write(filled_prompt)
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("SOURCE LIST:\n")
            f.write("=" * 80 + "\n\n")
            f.write(json.dumps(source_list, indent=2, ensure_ascii=False))
            f.write("\n")
        
        logger.info(f"[DEV_OUTPUTS] Wrote prompt to {filepath}")
    except Exception as e:
        logger.error(f"[DEV_OUTPUTS] Failed to write prompt to dev_outputs: {e}")


def generate_prompt(
    user_question: str,
    data: list[DocumentModel],
    prompt_id: PromptType = PromptType.LIGHT,
    request_id: str | None = None,
) -> tuple[Prompt, list[dict]]:
    """
    Constructs a prompt including retrieved context snippets.
    
    Args:
        user_question: The user's question
        data: List of retrieved documents
        prompt_id: Type of prompt to generate
        request_id: Optional request ID for logging and debug output filenames
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"[PROMPT GENERATION] Starting prompt generation, num_docs={len(data)}, prompt_id={prompt_id}"
    )

    resolved_prompt_id = resolve_prompt_key(prompt_id)

    def estimate_tokens(text: str) -> int:
        return len(text) // 4  # very rough estimate

    context_parts = []
    token_count = 0
    source_list = []  # Store numbered sources for STRUCTURED_JSON
    source_number = 1  # Start numbering at 1

    def _normalize_text(text):
        if isinstance(text, list):
            return " ".join(segment[0] for segment in text).strip()
        return (text or "").strip()

    for doc in data:
        # For STRUCTURED_JSON prompts, handle list of tuples differently
        if resolved_prompt_id == PromptType.STRUCTURED_JSON.value and isinstance(doc.text, list):
            # Each tuple in the list becomes a separate numbered source
            slug = doc.sanity_data.slug
            
            for segment in doc.text:
                if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                    text_content = segment[0]
                    timestamps = segment[1]
                    
                    # Extract start and end timestamps from the tuple
                    if isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
                        time_start = timestamps[0]
                        time_end = timestamps[1]
                        if time_start and time_end:
                            timestamp = f"{time_start}-{time_end}"
                        elif time_start:
                            timestamp = f"{time_start}"
                        elif time_end:
                            timestamp = f"{time_end}"
                        else:
                            timestamp = None
                    else:
                        timestamp = None
                    
                    # Store source info for later mapping
                    source_list.append({
                        "number": source_number,
                        "slug": slug,
                        "timestamp": timestamp,
                        "text": text_content,
                        "text_id": doc.id,
                    })
                    
                    # Create numbered entry for context
                    entry = f"[{source_number}] {text_content}\n(Source: {slug}, Timestamp: {timestamp})"
                    
                    logger.debug(
                        f"[PROMPT GENERATION] Added source [{source_number}]: slug={slug}, "
                        f"timestamp={timestamp}, text_length={len(text_content)} chars"
                    )
                    
                    tokens = estimate_tokens(entry)
                    context_parts.append(entry)
                    token_count += tokens
                    source_number += 1
        else:
            # For non-STRUCTURED_JSON prompts or plain text, use existing logic
            quote = _normalize_text(doc.text)

            # Build timestamp string from metadata if available
            time_start = doc.metadata.time_start if hasattr(doc.metadata, "time_start") else None
            time_end = doc.metadata.time_end if hasattr(doc.metadata, "time_end") else None
            if time_start and time_end:
                timestamp = f"{time_start}-{time_end}"
            elif time_start:
                timestamp = f"{time_start}"
            elif time_end:
                timestamp = f"{time_end}"
            else:
                timestamp = None

            # Choose context entry format based on prompt_id
            if resolved_prompt_id == PromptType.STRUCTURED_JSON.value:
                # Numbered source format
                slug = doc.sanity_data.slug
                
                # Store source info for later mapping
                source_list.append({
                    "number": source_number,
                    "slug": slug,
                    "timestamp": timestamp,
                    "text": quote,
                    "text_id": doc.id,
                })
                
                # Create numbered entry for context
                entry = f"[{source_number}] {quote}\n(Source: {slug}, Timestamp: {timestamp})"
                
                # Log the source chunk being added
                logger.debug(
                    f"[PROMPT GENERATION] Added source [{source_number}]: slug={slug}, "
                    f"timestamp={timestamp}, text_length={len(quote)} chars"
                )
                source_number += 1
            else:
                # Original human-readable entry with metadata dump
                # Exclude text_hash and full_text_id from prompt (internal fields)
                metadata_dict = doc.metadata.model_dump()
                filtered_metadata = {
                    k: v for k, v in metadata_dict.items() 
                    if k not in ['text_hash', 'full_text_id', 'text_to_embed', 'full_text']
                }
                metadata_str = ", ".join(
                    f"{k}: {v}" for k, v in filtered_metadata.items()
                )
                # Add slug explicitly to source line
                entry = f'"{quote}"\n(Source: slug: {doc.sanity_data.slug}, {metadata_str})'

            tokens = estimate_tokens(entry)

            context_parts.append(entry)
            token_count += tokens

    context = "\n\n".join(context_parts)
    logger.info(f"[PROMPT GENERATION] Context built with {len(context_parts)} documents, ~{token_count} estimated tokens")

    prompt_template = get_prompt_template(prompt_id)
    logger.info(
        f"[PROMPT GENERATION] Using prompt template: {resolved_prompt_id}"
    )

    filled_prompt = prompt_template.format(
        context=context,
        user_question=user_question,
    )

    logger.info(f"[PROMPT GENERATION] Final prompt length: {len(filled_prompt)} characters")
    logger.info(f"[PROMPT GENERATION] Created {len(source_list)} numbered sources for STRUCTURED_JSON")
    logger.debug(f"[PROMPT GENERATION] Prompt preview (first 500 chars): {filled_prompt[:500]}...")
    
    # Write prompt to dev_outputs if enabled
    settings = get_settings()
    if settings.dev_outputs:
        _write_prompt_to_dev_outputs(
            filled_prompt=filled_prompt,
            user_question=user_question,
            prompt_id=resolved_prompt_id,
            request_id=request_id,
            num_docs=len(data),
            source_list=source_list,
        )
    
    return (
        Prompt(
            value=filled_prompt,
            id=resolved_prompt_id,
        ),
        source_list,
    )


def get_prompt_template(prompt: PromptType = PromptType.LIGHT) -> str:
    resolved_prompt_id = resolve_prompt_key(prompt)
    return PROMPTS[resolved_prompt_id]
