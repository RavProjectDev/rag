"""
Webhook endpoints for document sync triggered by Sanity CMS (or any external CMS).

All routes live under /api/v1/webhook (registered in app/webhook.py).

Each handler returns 200 immediately and processes in the background so the
webhook sender (Sanity) doesn't time out waiting for the embed pipeline.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Request

from rag.app.models.data import SanityData
from rag.app.services.sync_service.sync_db import sync_document

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/create")
async def webhook_create(
    payload: SanityData,
    background_tasks: BackgroundTasks,
    request: Request,
):
    conn = request.app.state.embedding_conn
    settings = request.app.state.settings
    logger.info(f"[WEBHOOK] create received: slug={payload.slug} id={payload.id}")
    background_tasks.add_task(
        sync_document,
        payload,
        conn,
        settings.embedding_configuration,
        settings.chunking_strategy,
        conn.chunks_collection,
    )
    return {"status": "accepted"}


@router.patch("/update")
async def webhook_update(
    payload: SanityData,
    background_tasks: BackgroundTasks,
    request: Request,
):
    conn = request.app.state.embedding_conn
    settings = request.app.state.settings
    logger.info(f"[WEBHOOK] update received: slug={payload.slug} id={payload.id}")
    background_tasks.add_task(
        sync_document,
        payload,
        conn,
        settings.embedding_configuration,
        settings.chunking_strategy,
        conn.chunks_collection,
    )
    return {"status": "accepted"}


@router.delete("/delete")
async def webhook_delete(
    payload: SanityData,
    background_tasks: BackgroundTasks,
    request: Request,
):
    conn = request.app.state.embedding_conn
    logger.info(f"[WEBHOOK] delete received: slug={payload.slug} id={payload.id}")
    background_tasks.add_task(conn.delete_document, payload.id)
    return {"status": "accepted"}
