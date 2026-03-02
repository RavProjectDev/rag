"""
RAG API — chat, retrieval, and supporting endpoints.

Start with:
    uvicorn rag.app.main:app
"""

import logging

from fastapi import FastAPI

from rag.app.api.v1.chat import router as chat_router
from rag.app.api.v1.health import router as health_router
from rag.app.api.v1.docs import router as docs_router
from rag.app.api.v1.info import router as info_router
from rag.app.api.v1.mock import router as mock_router
from rag.app.api.v1.form import router as form_router
from rag.app.api.v1.prompt import router as prompt_router
from rag.app.api.v1.user import router as user_router
from rag.app.api.v1.config import router as config_router
from rag.app.core.config import get_settings
from rag.app.core.lifespan import create_lifespan, register_middleware

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RAV RAG API",
    version="1.0.0",
    description=(
        "Production-grade API for RAG chat and evaluation.\n\n"
        "Provides chat completion with retrieval augmentation "
        "and endpoints for evaluation data collection."
    ),
    lifespan=create_lifespan(get_settings),
)

register_middleware(app)

app.include_router(chat_router,   prefix="/api/v1/chat",    tags=["chat"])
app.include_router(health_router, prefix="/api/v1/health",  tags=["health"])
app.include_router(info_router,   prefix="/api/v1/info",    tags=["info"])
app.include_router(mock_router,   prefix="/api/v1/test",    tags=["mock"])
app.include_router(prompt_router, prefix="/api/v1/prompt",  tags=["prompt"])
app.include_router(user_router,   prefix="/api/v1/user",    tags=["user"])
app.include_router(config_router, prefix="/api/v1/config",  tags=["config"])
app.include_router(docs_router,   prefix="",                tags=["docs"])
app.include_router(form_router,   prefix="/form",           tags=["form"])
