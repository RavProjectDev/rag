import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from rag.app.db.connections import MetricsConnection
from rag.app.exceptions.llm import LLMTimeoutException
from rag.app.models.data import DocumentModel, Metadata, SanityData
from rag.app.schemas.data import LLMModel
from rag.app.services.llm import (
    get_llm_response,
    stream_llm_response,
    generate_prompt,
    get_mock_response,
    get_gpt_response,
)


class MockMetricsConnection(MetricsConnection):
    async def log(self, metric_type: str, data: Dict[str, Any]):
        return

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class MockChatCompletion:
    def __init__(self, slow=False, invalid_chunk=False):
        self.slow = slow
        self.invalid_chunk = invalid_chunk

    async def create(self, model, messages, stream):

        if not stream:
            return {"mock": "response"}  # for non-streaming calls

        async def async_gen():
            if self.slow:
                await asyncio.sleep(5)

            if self.invalid_chunk:
                yield {"mock": "response"}

            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "))])
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="World"))])
            # Final yield with empty content to signal completion
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=None))])

        return async_gen()  # Return async generator


class MockChat:
    def __init__(self, slow=False, invalid_chunk=False):
        self.completions = MockChatCompletion(slow, invalid_chunk)


class MockOpenAIClient:
    def __init__(self, slow=False, invalid_chunk=False):
        self.chat = MockChat(slow, invalid_chunk)


# Helper to create a valid ChatCompletion object
def create_chat_completion(
    content, prompt_tokens=10, completion_tokens=20, total_tokens=30
):
    return ChatCompletion(
        id="test_id",
        choices=[
            {
                "index": 0,
                "message": ChatCompletionMessage(
                    content=content,
                    role="assistant",
                    tool_calls=None,
                ),
                "finish_reason": "stop",
            }
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        system_fingerprint=None,
    )


@pytest.mark.asyncio
async def test_get_llm_response_gpt4_success():
    mock_metrics = MockMetricsConnection()
    prompt = "What is Rav Soloveitchik's view on modernity?"
    mock_response = "Rav Soloveitchik embraced modernity thoughtfully."
    mock_metrics_data = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "input_model": "gpt-4",
        "model": "gpt-4",
    }

    with patch(
        "rag.app.services.llm.get_gpt_response",
        return_value=(mock_response, mock_metrics_data),
    ):
        response = await get_llm_response(prompt, LLMModel.GPT_4, mock_metrics)
        assert response == mock_response


@pytest.mark.asyncio
async def test_get_llm_response_mock():
    prompt = "Test prompt"
    mock_response, mock_metrics = get_mock_response()
    mock_metrics = MockMetricsConnection()

    response = await get_llm_response(prompt, LLMModel.MOCK, mock_metrics)
    assert response == mock_response
    assert response.startswith("Lorem ipsum dolor sit amet")


@pytest.mark.asyncio
async def test_get_llm_response_unsupported_model():
    mock_metrics = MagicMock(spec=MetricsConnection)
    with pytest.raises(ValueError, match="Unsupported model"):
        await get_llm_response("Test prompt", "invalid_model", mock_metrics)


@pytest.mark.asyncio
@patch("rag.app.services.llm.get_openai_client")
async def test_get_gpt_response_success(mock_get_client):
    mock_client = AsyncMock()
    mock_response = create_chat_completion("Test response")
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    response, metrics = await get_gpt_response("Test prompt", "gpt-4")
    assert response == "Test response"
    assert metrics == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "input_model": "gpt-4",
        "model": "gpt-4",
    }


@pytest.mark.asyncio
@patch("rag.app.services.llm.get_openai_client")
async def test_get_gpt_response_null_response(mock_get_client):
    mock_client = AsyncMock()
    mock_response = create_chat_completion(
        None, prompt_tokens=10, completion_tokens=0, total_tokens=10
    )
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    response, metrics = await get_gpt_response("Test prompt", "gpt-4")
    assert response == "Error: Received null response from OpenAI"
    assert metrics is None


@pytest.mark.asyncio
@patch("rag.app.services.llm.get_openai_client")
async def test_stream_llm_response_success(mock_get_client):
    mock_metrics = MockMetricsConnection()
    mock_get_client.return_value = MockOpenAIClient()

    with patch(
        "rag.app.services.llm.get_settings",
        return_value=MagicMock(external_api_timeout=10),
    ):
        chunks = [
            chunk
            async for chunk in stream_llm_response(mock_metrics, "Test prompt", "gpt-4")
        ]
        assert chunks == ["Hello ", "World", "[DONE]"]


@pytest.mark.asyncio
@patch("rag.app.services.llm.get_openai_client")
async def test_stream_llm_response_timeout(mock_get_client):
    mock_metrics = MockMetricsConnection()
    mock_get_client.return_value = MockOpenAIClient(slow=True)

    with patch(
        "rag.app.services.llm.get_settings",
        return_value=MagicMock(external_api_timeout=1),
    ):
        with pytest.raises(LLMTimeoutException):
            async for _ in stream_llm_response(mock_metrics, "Test prompt", "gpt-4"):
                pass


@pytest.mark.asyncio
@patch("rag.app.services.llm.get_openai_client")
async def test_stream_llm_response_timeout_and_invalid_chunk_structure(mock_get_client):
    mock_metrics = MockMetricsConnection()
    mock_get_client.return_value = MockOpenAIClient(slow=True)

    with patch(
        "rag.app.services.llm.get_settings",
        return_value=MagicMock(external_api_timeout=1),
    ):
        with pytest.raises(LLMTimeoutException):
            async for _ in stream_llm_response(mock_metrics, "Test prompt", "gpt-4"):
                pass


@pytest.mark.asyncio
@patch("rag.app.services.llm.get_openai_client")
async def test_stream_llm_response_invalid_chunk(mock_get_client):
    mock_metrics = MockMetricsConnection()
    mock_get_client.return_value = MockOpenAIClient(invalid_chunk=True)

    with patch(
        "rag.app.services.llm.get_settings",
        return_value=MagicMock(external_api_timeout=10),
    ):
        chunks = [
            chunk
            async for chunk in stream_llm_response(mock_metrics, "Test prompt", "gpt-4")
        ]

        assert chunks == ["Error: Invalid chunk structure"]


def test_generate_prompt_no_context():
    user_question = "What is Rav Soloveitchik's view on modernity?"
    data = []
    prompt = generate_prompt(user_question, data)
    assert "You are a Rav Soloveitchik expert" in prompt.value
    assert user_question in prompt.value
    assert "# Context" in prompt.value
    assert not any("Source:" in prompt.value for _ in data)


def test_generate_prompt_with_context():
    user_question = "What is Rav Soloveitchik's view on modernity?"
    data = [
        DocumentModel(
            _id="687c65e061b769c8ff78779f",
            text="Modernity must be approached with critical engagement.",
            metadata=Metadata(
                chunk_size=100,
                time_start="00:00",
                time_end="01:00",
                name_space="lecture",
            ),
            sanity_data=SanityData(
                id="sanity1",
                slug="lonely-man",
                title="The Lonely Man of Faith",
                transcriptURL="https://example.com/transcript1",
                hash="abc123",
            ),
            score=0.9,
        ),
        DocumentModel(
            _id="687c65e061b769c8ff78780f",
            text="Faith and reason are complementary.",
            metadata=Metadata(
                chunk_size=150, time_start="01:00", time_end="02:00", name_space="book"
            ),
            sanity_data=SanityData(
                id="sanity2",
                slug="halakhic-man",
                title="Halakhic Man",
                transcriptURL="https://example.com/transcript2",
                hash="def456",
            ),
            score=0.8,
        ),
    ]
    prompt = generate_prompt(user_question, data, max_tokens=1000)
    assert (
        '"Modernity must be approached with critical engagement."\n(Source: chunk_size: 100, time_start: 00:00, time_end: 01:00, name_space: lecture)'
        in prompt.value
    )
    assert (
        '"Faith and reason are complementary."\n(Source: chunk_size: 150, time_start: 01:00, time_end: 02:00, name_space: book)'
        in prompt.value
    )
    assert user_question in prompt.value


def test_generate_prompt_token_limit():
    user_question = "Test question"
    data = [
        DocumentModel(
            _id="687c65e061b769c8ff78779f",
            text="A" * 50,
            metadata=Metadata(
                chunk_size=100,
                time_start="00:00",
                time_end="01:00",
                name_space="lecture",
            ),
            sanity_data=SanityData(
                id="sanity1",
                slug="test-source",
                title="Test Source",
                transcriptURL="https://example.com/transcript1",
                hash="abc123",
            ),
            score=0.9,
        ),
        DocumentModel(
            _id="687c65e061b769c8ff78780f",
            text="B" * 1000,
            metadata=Metadata(
                chunk_size=150, time_start="01:00", time_end="02:00", name_space="book"
            ),
            sanity_data=SanityData(
                id="sanity2",
                slug="test-source-2",
                title="Test Source 2",
                transcriptURL="https://example.com/transcript2",
                hash="def456",
            ),
            score=0.8,
        ),
    ]
    prompt = generate_prompt(user_question, data, max_tokens=100)
    assert f'{"A" * 50}' in prompt.value
    assert (
        f'{"B" * 1000}\n(Source: chunk_size: 100, time_start: 00:00, time_end: 01:00, name_space: lecture)'
        not in prompt.value
    )
