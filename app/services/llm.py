import asyncio
import logging
import os
from datetime import datetime
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
        require_timestamp: If True, timestamp will be required in the schema.
                          If False, timestamp will be optional.
    """
    source_required = ["slug", "text"]
    if require_timestamp:
        source_required.append("timestamp")
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "chat_response",
            "schema": {
                "type": "object",
                "properties": {
                    "main_text": {
                        "type": "string",
                        "description": "A concise summary synthesizing the relevant extracted quotes"
                    },
                    "sources": {
                        "type": "array",
                        "description": "Array of relevant quoted sources from the context",
                        "items": {
                            "type": "object",
                            "properties": {
                                "slug": {
                                    "type": "string",
                                    "description": "The source document slug"
                                },
                                "timestamp": {
                                    "type": ["string", "null"],
                                    "description": "Timestamp in format 'start-end' or single value, or null"
                                },
                                "text": {
                                    "type": "string",
                                    "description": "A relevant quote or excerpt from the source document"
                                }
                            },
                            "required": source_required,
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["main_text", "sources"],
                "additionalProperties": False
            },
            "strict": False
        }
    }


def get_numbered_sources_json_schema() -> Dict[str, Any]:
    """
    Returns the JSON schema for numbered sources responses.
    This schema enforces the structure with main_summary and sources array of integers.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "numbered_sources_response",
            "schema": {
                "type": "object",
                "properties": {
                    "main_summary": {
                        "type": "string",
                        "description": "A comprehensive summary that answers the user's question"
                    },
                    "sources": {
                        "type": "array",
                        "description": "Array of document numbers (integers) that are relevant to answering the question",
                        "items": {
                            "type": "integer",
                            "description": "Document number from the context (e.g., 1, 2, 3)"
                        }
                    }
                },
                "required": ["main_summary", "sources"],
                "additionalProperties": False
            },
            "strict": False
        }
    }


def save_prompt_to_dev_outputs(
    prompt: str, 
    request_id: Optional[str] = None, 
    model: Optional[str] = None,
    response: Optional[str] = None
) -> str:
    """
    Saves the LLM prompt (and optionally response) to dev_outputs folder when dev_outputs mode is enabled.
    
    Args:
        prompt: The prompt text to save
        request_id: Optional request ID for filename
        model: Optional model name for filename
        response: Optional response text to save
    
    Returns:
        The filepath where the data was saved (or empty string if not saved)
    """
    settings = get_settings()
    if not settings.dev_outputs:
        return ""
    
    # Create dev_outputs directory if it doesn't exist (relative to project root)
    dev_outputs_dir = "dev_outputs"
    try:
        os.makedirs(dev_outputs_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create dev_outputs directory: {e}")
        return ""
    
    # Generate filename with timestamp and request_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    request_id_short = request_id[:8] if request_id else "unknown"
    
    if model:
        # Sanitize model name for filename (remove special characters)
        model_safe = model.replace("/", "_").replace(":", "_")
        filename = f"{timestamp}_{model_safe}_{request_id_short}.txt"
    else:
        filename = f"{timestamp}_{request_id_short}.txt"
    
    filepath = os.path.join(dev_outputs_dir, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Request ID: {request_id or 'N/A'}\n")
            f.write(f"Model: {model or 'N/A'}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
            f.write("PROMPT:\n")
            f.write(f"{'='*80}\n")
            f.write(prompt)
            f.write(f"\n\n{'='*80}\n")
            if response is not None:
                f.write("RESPONSE:\n")
                f.write(f"{'='*80}\n")
                f.write(response)
                f.write(f"\n{'='*80}\n")
        logging.info(f"[DEV_OUTPUTS] Saved prompt{' and response' if response else ''} to {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"[DEV_OUTPUTS] Failed to save to {filepath}: {e}")
        return ""


# ---------------------------------------------------------------
# Single-response LLM call
# ---------------------------------------------------------------


async def get_llm_response(
    prompt: str,
    model: LLMModel = LLMModel.GPT_4,
    metrics_connection: MetricsConnection = None,
    response_format: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> str:
    """
    Fetches a synchronous completion from the LLM.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The LLM model to use
        metrics_connection: Optional metrics connection for tracking
        response_format: Optional response format for structured outputs (e.g., JSON schema)
        request_id: Optional request ID for dev_outputs tracking
    """
    # Save prompt to dev_outputs if enabled (will be updated with response later)
    model_name = model.value if hasattr(model, 'value') else str(model) if model else None
    filepath = save_prompt_to_dev_outputs(prompt, request_id=request_id, model=model_name)
    
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
    
    # Update dev_outputs file with response if enabled
    if filepath:
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"\n\n{'='*80}\n")
                f.write("RESPONSE:\n")
                f.write(f"{'='*80}\n")
                f.write(response)
                f.write(f"\n{'='*80}\n")
        except Exception as e:
            logging.error(f"[DEV_OUTPUTS] Failed to append response to {filepath}: {e}")
    
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
    request_id: Optional[str] = None,
):
    """
    Streams completion from the LLM as text chunks.
    
    Args:
        metrics_connection: Metrics connection for tracking
        prompt: The prompt to send to the LLM
        model: The model name to use
        response_format: Optional response format for structured outputs (e.g., JSON schema)
        request_id: Optional request ID for dev_outputs tracking
    """
    # Save prompt to dev_outputs if enabled (will be updated with response later)
    filepath = save_prompt_to_dev_outputs(prompt, request_id=request_id, model=model)
    
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

        # Collect all chunks for dev_outputs
        collected_response = []

        async def _consume_stream():
            async for chunk in response:
                try:
                    if not chunk.choices or not chunk.choices[0].delta:
                        yield "Error: Invalid chunk structure"
                        return
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content = delta.content
                        collected_response.append(content)
                        yield content
                except (AttributeError, IndexError):
                    yield "Error: Invalid chunk structure"
                    return
            
            # Save complete response to dev_outputs if enabled
            if filepath:
                try:
                    full_response = "".join(collected_response)
                    with open(filepath, "a", encoding="utf-8") as f:
                        f.write(f"\n\n{'='*80}\n")
                        f.write("RESPONSE:\n")
                        f.write(f"{'='*80}\n")
                        f.write(full_response)
                        f.write(f"\n{'='*80}\n")
                except Exception as e:
                    logging.error(f"[DEV_OUTPUTS] Failed to append response to {filepath}: {e}")
            
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


def generate_prompt(
    user_question: str,
    data: list[DocumentModel],
    prompt_id: PromptType = PromptType.LIGHT,
) -> Prompt:
    """
    Constructs a prompt including retrieved context snippets.
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

    for idx, doc in enumerate(data, start=1):
        quote = doc.text.strip()

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
            # JSON-line style entry with explicit fields for easier association
            # Ensure double quotes and nulls where appropriate
            safe_text = quote.replace("\\", "\\\\").replace("\"", "\\\"")
            safe_slug = doc.sanity_data.slug.replace("\\", "\\\\").replace("\"", "\\\"")
            ts_value = f'"{timestamp}"' if timestamp else "null"
            entry = (
                "{"
                f"\"slug\": \"{safe_slug}\", \"timestamp\": {ts_value}, \"text\": \"{safe_text}\""
                "}"
            )
            # Log the source chunk being added
            logger.debug(
                f"[PROMPT GENERATION] Added source: slug={doc.sanity_data.slug}, "
                f"timestamp={timestamp}, text_length={len(quote)} chars, "
                f"text_preview={quote[:100]}..."
            )
        elif resolved_prompt_id == PromptType.NUMBERED_SOURCES.value:
            # Numbered format for NUMBERED_SOURCES prompt type
            metadata_str = ", ".join(
                f"{k}: {v}" for k, v in doc.metadata.model_dump().items()
            )
            entry = f"[{idx}]\n\"{quote}\"\n(Source: slug: {doc.sanity_data.slug}, {metadata_str})"
            logger.debug(
                f"[PROMPT GENERATION] Added numbered source [{idx}]: slug={doc.sanity_data.slug}, "
                f"timestamp={timestamp}, text_length={len(quote)} chars"
            )
        else:
            # Original human-readable entry with metadata dump
            metadata_str = ", ".join(
                f"{k}: {v}" for k, v in doc.metadata.model_dump().items()
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
    logger.debug(f"[PROMPT GENERATION] Prompt preview (first 500 chars): {filled_prompt[:500]}...")
    
    return Prompt(
        value=filled_prompt,
        id=resolved_prompt_id,
    )


def get_prompt_template(prompt: PromptType = PromptType.LIGHT) -> str:
    resolved_prompt_id = resolve_prompt_key(prompt)
    return PROMPTS[resolved_prompt_id]
