"""
Webhook endpoints for document sync triggered by Sanity CMS (or any external CMS).

All routes live under /api/v1/webhook (registered in app/webhook.py).
"""

from fastapi import APIRouter
from rag.app.models.data import SanityData

router = APIRouter()


@router.post("/create")
async def webhook_create(payload: SanityData):
    pass


@router.patch("/update")
async def webhook_update(payload: SanityData):
    pass


@router.delete("/delete")
async def webhook_delete(payload: SanityData):
    pass
