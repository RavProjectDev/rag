"""
Sync Webhook API — receives CMS events and triggers document sync.

Start with:
    uvicorn rag.app.webhook:app
"""

import logging

from fastapi import FastAPI

from rag.app.api.v1.health import router as health_router
from rag.app.api.v1.webhook import router as webhook_router
from rag.app.core.webhook_config import get_webhook_settings, create_webhook_lifespan
from rag.app.core.lifespan import register_middleware

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RAV Sync Webhook",
    version="1.0.0",
    description=(
        "Receives CMS webhook events (create / update / delete) "
        "and keeps the vector store in sync."
    ),
    lifespan=create_webhook_lifespan(get_webhook_settings),
)

register_middleware(app)

app.include_router(webhook_router, prefix="/api/v1/webhook", tags=["webhook"])
app.include_router(health_router,  prefix="/api/v1/health",  tags=["health"])
