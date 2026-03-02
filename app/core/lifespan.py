"""
Shared lifespan factory and middleware registration for all FastAPI app instances.

Usage:
    # RAG API
    from rag.app.core.lifespan import create_lifespan, register_middleware
    from rag.app.core.config import get_settings
    lifespan = create_lifespan(get_settings)

    # Webhook API
    from rag.app.core.webhook_config import get_webhook_settings
    lifespan = create_lifespan(get_webhook_settings)
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable

import certifi
from fastapi import FastAPI, Request
from fastapi import HTTPException as FastAPIHTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient

from rag.app.core.config import SharedSettings, Environment
from rag.app.db.connections import MetricsConnection, ExceptionsLogger
from rag.app.db.mongodb_connection import (
    MongoEmbeddingStore,
    MongoMetricsConnection,
    MongoExceptionsLogger,
)
from rag.app.db.pinecone_connection import PineconeEmbeddingStore
from rag.app.db.redis_connection import RedisConnection
from rag.app.schemas.data import DataBaseConfiguration
from rag.app.schemas.response import ErrorResponse

logger = logging.getLogger(__name__)


def create_lifespan(get_settings_fn: Callable[[], SharedSettings]):
    """
    Factory that returns a lifespan context manager bound to the given settings getter.
    Each app passes its own get_settings function so only its required env vars are loaded.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings_fn()
        client = AsyncIOMotorClient(
            settings.mongodb_uri, tlsCAFile=certifi.where(), maxPoolSize=50
        )
        try:
            db = client[settings.mongodb_db_name]
            metrics_collection = db[settings.metrics_collection]
            exceptions_collection = db[settings.exceptions_collection]

            if settings.database_configuration == DataBaseConfiguration.PINECONE:
                required_pinecone = {"pinecone_api_key": settings.pinecone_api_key}
                missing = [k for k, v in required_pinecone.items() if not v]
                if missing:
                    raise ValueError(
                        f"Missing required Pinecone settings: {', '.join(missing)}"
                    )

                index_name = settings.pinecone_index_name or settings.embedding_configuration.value
                namespace = settings.pinecone_namespace or settings.chunking_strategy.value

                logger.info(
                    f"[PINECONE CONFIG] index={index_name} "
                    f"(embedding_model={settings.embedding_configuration.value}), "
                    f"namespace={namespace} (chunking_strategy={settings.chunking_strategy.value})"
                )

                chunks_collection = db["chunks"]
                logger.info("MongoDB chunks collection: chunks (for Pinecone chunk tracking)")

                embedding_connection = PineconeEmbeddingStore(
                    api_key=settings.pinecone_api_key,
                    index_name=index_name,
                    environment=settings.pinecone_environment,
                    namespace=namespace,
                    host=settings.pinecone_host,
                    chunks_collection=chunks_collection,
                )
            else:
                collection_name = settings.mongodb_vector_collection
                if settings.chunking_strategy.value not in collection_name:
                    collection_name = f"{collection_name}_{settings.chunking_strategy.value}"

                logger.info(
                    f"MongoDB collection: {collection_name} "
                    f"(based on chunking_strategy: {settings.chunking_strategy.value})"
                )
                vector_embedding_collection = db[collection_name]
                chunks_collection = db["chunks"]
                logger.info("MongoDB chunks collection: chunks")
                embedding_connection = MongoEmbeddingStore(
                    collection=vector_embedding_collection,
                    index=settings.collection_index,
                    vector_path=settings.vector_path,
                    chunks_collection=chunks_collection,
                )

            metrics_connection = MongoMetricsConnection(collection=metrics_collection)
            exceptions_logger = MongoExceptionsLogger(collection=exceptions_collection)

            # Redis is optional and only present on Settings (RAG). Use getattr so
            # WebhookSettings (which doesn't declare these fields) gracefully skips it.
            redis_url = getattr(settings, "upstash_redis_rest_url", None)
            redis_token = getattr(settings, "upstash_redis_rest_token", None)
            redis_conn = None
            if redis_url and redis_token:
                try:
                    redis_conn = RedisConnection(url=redis_url, token=redis_token)
                    logger.info("Redis connection initialized successfully")
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize Redis connection: {e}. Rate limiting will be disabled."
                    )
            else:
                logger.info("Redis credentials not provided. Rate limiting will be disabled.")

            app.state.settings = settings
            app.state.embedding_conn = embedding_connection
            app.state.mongo_conn = embedding_connection  # backwards compat
            app.state.metrics_connection = metrics_connection
            app.state.exceptions_logger = exceptions_logger
            app.state.redis_conn = redis_conn
            app.state.db_client = db

            logger.info("Sync scheduler disabled (not running automatic sync jobs)")

            yield

        except Exception as e:
            logger.error(f"Failed to initialize application resources: {e}")
            raise
        finally:
            client.close()

    return lifespan


def register_middleware(app: FastAPI) -> None:
    """
    Attach CORS, request-logging, and exception handlers to a FastAPI instance.
    Reads environment from app.state.settings (set during lifespan startup).
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Enrich each request with a correlation id and log timings.

        In PRD also sends timing metrics to the metrics sink.
        """
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration = time.perf_counter() - start
            data = {
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": getattr(locals().get("response", None), "status_code", None),
                "duration": duration,
                "request_id": request_id,
            }
            logger.info(
                f"{request.method} {request.url.path} "
                f"completed in {duration:.4f}s [request_id={request_id}]"
            )
            app_settings: SharedSettings | None = getattr(request.app.state, "settings", None)
            if (
                app_settings is not None
                and app_settings.environment == Environment.PRD
                and hasattr(request.app.state, "metrics_connection")
            ):
                try:
                    metrics_connection: MetricsConnection = request.app.state.metrics_connection
                    await metrics_connection.log(metric_type="endpoint_timing", data=data)
                except Exception:
                    pass

        if "response" in locals():
            response.headers["X-Request-ID"] = request_id

            if hasattr(request.state, "user_rate_limit_limit"):
                response.headers["X-RateLimit-Limit"] = str(request.state.user_rate_limit_limit)
                response.headers["X-RateLimit-Remaining"] = str(
                    request.state.user_rate_limit_remaining
                )
                response.headers["X-RateLimit-Reset"] = str(request.state.user_rate_limit_reset)

            return response

        payload = ErrorResponse(
            code="internal_server_error", message="Unhandled error", request_id=request_id
        ).model_dump()
        return JSONResponse(status_code=500, content=payload)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        try:
            exceptions_logger: ExceptionsLogger = request.app.state.exceptions_logger
            await exceptions_logger.log(
                exception_code=getattr(exc, "code", None),
                data={
                    "endpoint": request.url.path,
                    "method": request.method,
                    "error": str(exc),
                    "request_id": request_id,
                },
            )
        except Exception:
            pass
        payload = ErrorResponse(
            code=getattr(exc, "code", "internal_server_error"),
            message=str(exc),
            request_id=request_id,
        ).model_dump()
        return JSONResponse(status_code=500, content=payload)

    @app.exception_handler(FastAPIHTTPException)
    async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
        request_id = getattr(request.state, "request_id", None)
        detail: Any = exc.detail
        if isinstance(detail, dict):
            code = detail.get("code", "http_error")
            message = detail.get("message") or detail.get("error") or str(detail)
            extra = {
                k: v for k, v in detail.items() if k not in {"code", "message", "error"}
            }
        else:
            code = "http_error"
            message = str(detail)
            extra = None
        try:
            exceptions_logger: ExceptionsLogger = request.app.state.exceptions_logger
            await exceptions_logger.log(
                exception_code=code,
                data={
                    "endpoint": request.url.path,
                    "method": request.method,
                    "error": message,
                    "request_id": request_id,
                    "status_code": exc.status_code,
                },
            )
        except Exception:
            pass
        payload = ErrorResponse(
            code=code, message=message, request_id=request_id, details=extra
        ).model_dump()
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", None)
        details = {"errors": exc.errors()}
        payload = ErrorResponse(
            code="validation_error",
            message="Invalid request",
            request_id=request_id,
            details=details,
        ).model_dump()
        return JSONResponse(status_code=422, content=payload)
