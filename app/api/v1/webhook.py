"""
Webhook endpoints for document sync triggered by Sanity CMS (or any external CMS).

All routes live under /api/v1/webhook (registered in app/webhook.py).

Each handler returns 200 immediately and processes in the background so the
webhook sender (Sanity) doesn't time out waiting for the embed pipeline.

embedding_configuration and chunking_strategy are resolved dynamically on every
request via the `get_live_config` dependency, which calls the RAG API's
/api/v1/info/ endpoint. This ensures the webhook always mirrors the live RAG
API config even after a redeployment that changes the embedding model or chunking
strategy. Falls back to the cached startup values if the RAG API is unreachable.
"""

import logging
import json
from typing import NamedTuple, Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Request, Body

from rag.app.schemas.data import EmbeddingConfiguration, ChunkingStrategy
from rag.app.schemas.requests import SanityWebhookPayload
from rag.app.services.sync_service.sync_db import sync_document

logger = logging.getLogger(__name__)

router = APIRouter()


class LiveConfig(NamedTuple):
    embedding_configuration: EmbeddingConfiguration
    chunking_strategy: ChunkingStrategy


async def get_live_config(request: Request) -> LiveConfig:
    """
    Fetch the current embedding model and chunking strategy from the RAG API
    before every request. Falls back to the values cached at startup if the
    RAG API is unreachable.
    """
    settings = request.app.state.settings
    info_url = f"{settings.rag_api_url.rstrip('/')}/api/v1/info/"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(info_url)
            response.raise_for_status()
            config = response.json()

        embedding_configuration = EmbeddingConfiguration(config["embedding_model"])
        chunking_strategy = ChunkingStrategy(config["chunking_strategy"])

        logger.debug(
            f"[WEBHOOK] Live config: embedding={embedding_configuration.value}, "
            f"chunking={chunking_strategy.value}"
        )
        return LiveConfig(embedding_configuration, chunking_strategy)

    except Exception as e:
        logger.warning(
            f"[WEBHOOK] Could not reach RAG API ({info_url}): {e}. "
            f"Using cached startup config."
        )
        return LiveConfig(
            request.app.state.embedding_configuration,
            request.app.state.chunking_strategy,
        )


@router.get("/info")
async def webhook_info(config: LiveConfig = Depends(get_live_config)):
    """Returns the currently active config as reported by the live RAG API."""
    return {
        "embedding_model": config.embedding_configuration.value,
        "chunking_strategy": config.chunking_strategy.value,
    }


@router.post("/create")
async def webhook_create(
    payload: SanityWebhookPayload,
    background_tasks: BackgroundTasks,
    request: Request,
    config: LiveConfig = Depends(get_live_config),
):
    conn = request.app.state.embedding_conn
    sanity_data = payload.to_sanity_data()
    logger.info(f"[WEBHOOK] create received: slug={sanity_data.slug} id={sanity_data.id}")
    background_tasks.add_task(
        sync_document,
        sanity_data,
        conn,
        config.embedding_configuration,
        config.chunking_strategy,
        conn.chunks_collection,
    )
    return {"status": "accepted"}


@router.patch("/update")
async def webhook_update(
    payload: SanityWebhookPayload,
    background_tasks: BackgroundTasks,
    request: Request,
    config: LiveConfig = Depends(get_live_config),
):
    conn = request.app.state.embedding_conn
    sanity_data = payload.to_sanity_data()
    logger.info(f"[WEBHOOK] update received: slug={sanity_data.slug} id={sanity_data.id}")
    background_tasks.add_task(
        sync_document,
        sanity_data,
        conn,
        config.embedding_configuration,
        config.chunking_strategy,
        conn.chunks_collection,
    )
    return {"status": "accepted"}


@router.delete("/delete")
async def webhook_delete(
    payload: SanityWebhookPayload,
    background_tasks: BackgroundTasks,
    request: Request,
):
    conn = request.app.state.embedding_conn
    sanity_data = payload.to_sanity_data()
    logger.info(f"[WEBHOOK] delete received: slug={sanity_data.slug} id={sanity_data.id}")
    background_tasks.add_task(conn.delete_document, sanity_data.id)
    return {"status": "accepted"}


@router.post("/mock")
async def webhook_mock(payload: Any = Body(...)):
    """
    Mock endpoint for debugging webhook payloads.
    Accepts any JSON and logs it without validation or processing.
    """
    logger.info("[WEBHOOK MOCK] Received request:")
    logger.info(f"[WEBHOOK MOCK] Payload type: {type(payload)}")
    logger.info(f"[WEBHOOK MOCK] Payload: {json.dumps(payload, indent=2)}")
    
    print("\n" + "="*80)
    print("WEBHOOK MOCK - Request Body:")
    print("="*80)
    print(json.dumps(payload, indent=2))
    print("="*80 + "\n")
    
    return {
        "status": "received",
        "message": "Payload logged successfully",
        "payload": payload
    }
