import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Callable

import httpx
from fastapi import FastAPI

from rag.app.core.config import SharedSettings
from rag.app.core.lifespan import create_lifespan
from rag.app.schemas.data import EmbeddingConfiguration, ChunkingStrategy

logger = logging.getLogger(__name__)


class WebhookSettings(SharedSettings):
    """
    Settings for the Sync Webhook API.

    Only requires DB + embedding config from SharedSettings.
    Does NOT require OpenAI, Supabase, Redis, or rate-limit vars.

    Start with:
        uvicorn rag.app.webhook:app
    """

    # URL of the live RAG API — used at startup to fetch the active
    # embedding model and chunking strategy so this service always
    # mirrors the RAG API's configuration.
    rag_api_url: str = "https://api.theravlegacy.org"

    # HMAC secret used to verify incoming Sanity webhook signatures.
    webhook_secret: str | None = None


@lru_cache()
def get_webhook_settings() -> WebhookSettings:
    return WebhookSettings()


def create_webhook_lifespan(get_settings_fn: Callable[[], WebhookSettings]):
    """
    Webhook-specific lifespan factory.

    Wraps the shared lifespan (DB + embedding connection setup) and then
    fetches the live embedding_model + chunking_strategy from the RAG API's
    /api/v1/info/ endpoint, storing them on app.state so every request uses
    the same configuration as the live RAG API.

    Falls back to the values in WebhookSettings env vars if the RAG API is
    unreachable at startup.
    """
    shared_lifespan = create_lifespan(get_settings_fn)

    @asynccontextmanager
    async def webhook_lifespan(app: FastAPI):
        async with shared_lifespan(app):
            settings = get_settings_fn()
            info_url = f"{settings.rag_api_url.rstrip('/')}/api/v1/info/"
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.get(info_url)
                    response.raise_for_status()
                    config = response.json()

                embedding_configuration = EmbeddingConfiguration(config["embedding_model"])
                chunking_strategy = ChunkingStrategy(config["chunking_strategy"])

                logger.info(
                    f"[WEBHOOK] Live config from RAG API ({info_url}): "
                    f"embedding={embedding_configuration.value}, "
                    f"chunking={chunking_strategy.value}"
                )
            except Exception as e:
                logger.warning(
                    f"[WEBHOOK] Could not fetch live config from RAG API ({info_url}): {e}. "
                    f"Falling back to env var defaults: "
                    f"embedding={settings.embedding_configuration.value}, "
                    f"chunking={settings.chunking_strategy.value}"
                )
                embedding_configuration = settings.embedding_configuration
                chunking_strategy = settings.chunking_strategy

            app.state.embedding_configuration = embedding_configuration
            app.state.chunking_strategy = chunking_strategy

            yield

    return webhook_lifespan
