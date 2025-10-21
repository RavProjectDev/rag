import asyncio
import logging
from contextlib import nullcontext
from functools import lru_cache
from typing import Union

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
from rag.app.services.prompts import PROMPTS


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


# ---------------------------------------------------------------
# Single-response LLM call
# ---------------------------------------------------------------


async def get_llm_response(
    prompt: str,
    model: LLMModel = LLMModel.GPT_4,
    metrics_connection: MetricsConnection = None,
) -> str:
    """
    Fetches a synchronous completion from the LLM.
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
                    prompt=prompt, model=model.value
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
) -> tuple[str, dict | None]:

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
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
            )
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
):
    """
    Streams completion from the LLM as text chunks.
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
        # Make the async streaming API call with timeout
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            ),
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


def generate_prompt(
    user_question: str,
    data: list[DocumentModel],
    max_tokens: int = 1500,
    prompt_id=None,
) -> Prompt:
    """
    Constructs a prompt including retrieved context snippets.
    """

    def estimate_tokens(text: str) -> int:
        return len(text) // 4  # very rough estimate

    context_parts = []
    token_count = 0

    for doc in data:
        quote = doc.text.strip()
        metadata_str = ", ".join(
            f"{k}: {v}" for k, v in doc.metadata.model_dump().items()
        )

        entry = f'"{quote}"\n(Source: {metadata_str})'
        tokens = estimate_tokens(entry)

        if token_count + tokens > max_tokens:
            break

        context_parts.append(entry)
        token_count += tokens

    context = "\n\n".join(context_parts)

    prompt_template = get_prompt_template(prompt_id)

    filled_prompt = prompt_template.format(
        context=context,
        user_question=user_question,
    )

    return Prompt(
        value=filled_prompt,
        id=str(prompt_id),
    )


def get_prompt_template(prompt: int = None) -> str:
    if not prompt:
        return PROMPTS["1"]
    else:
        return PROMPTS[str(prompt)]
