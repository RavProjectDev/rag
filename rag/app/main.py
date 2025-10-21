import time
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, Request
from fastapi import HTTPException as FastAPIHTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
from fastapi.middleware.cors import CORSMiddleware


from rag.app.api.v1.chat import router as chat_router
from rag.app.api.v1.data_management import router as upload_router
from rag.app.api.v1.health import router as health_router
from rag.app.api.v1.docs import router as docs_router
from rag.app.api.v1.mock import router as mock_router
from rag.app.api.v1.form import router as form_router
from rag.app.db.connections import MetricsConnection, ExceptionsLogger

from rag.app.db.mongodb_connection import (
    MongoEmbeddingStore,
    MongoMetricsConnection,
    MongoExceptionsLogger,
)
from rag.app.core.config import get_settings, Environment
from rag.app.core.scheduler import start_scheduler
from rag.app.schemas.response import ErrorResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = AsyncIOMotorClient(
        settings.mongodb_uri, tlsCAFile=certifi.where(), maxPoolSize=50
    )
    try:
        db = client[settings.mongodb_db_name]
        vector_embedding_collection = db[settings.mongodb_vector_collection]
        metrics_collection = db[settings.metrics_collection]
        exceptions_collection = db[settings.exceptions_collection]

        mongo_connection = MongoEmbeddingStore(
            collection=vector_embedding_collection,
            index=settings.collection_index,
            vector_path=settings.vector_path,
        )
        metrics_connection = MongoMetricsConnection(
            collection=metrics_collection,
        )
        exceptions_logger = MongoExceptionsLogger(
            collection=exceptions_collection,
        )

        app.state.mongo_conn = mongo_connection
        app.state.metrics_connection = metrics_connection
        app.state.exceptions_logger = exceptions_logger
        app.state.db_client = db
        start_scheduler(
            connection=mongo_connection,
            embedding_configuration=settings.embedding_configuration,
        )
        yield

    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}")
        raise
    finally:
        client.close()


app = FastAPI(
    title="RAV RAG API",
    version="1.0.0",
    description=(
        "Production-grade API for RAG chat, data upload and evaluation.\n\n"
        "This service provides chat completion with retrieval augmentation,"
        " background sync, and endpoints for evaluation data collection."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production by env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to enrich each request with a correlation id and log timings.

    In PRD, it also sends timing metrics to the metrics sink and exceptions to the
    exceptions logger. Correlation id is propagated via X-Request-ID header.
    """
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id

    settings = get_settings()

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
            f"{request.method} {request.url.path} completed in {duration:.4f}s [request_id={request_id}]"
        )
        if settings.environment == Environment.PRD and hasattr(
            request.app.state, "metrics_connection"
        ):
            try:
                metrics_connection: MetricsConnection = (
                    request.app.state.metrics_connection
                )
                await metrics_connection.log(metric_type="endpoint_timing", data=data)
            except Exception:
                # avoid cascading failures
                pass

    if "response" in locals():
        response.headers["X-Request-ID"] = request_id
        return response
    # If response was never created due to an exception, return generic 500 here.
    payload = ErrorResponse(
        code="internal_server_error", message="Unhandled error", request_id=request_id
    ).model_dump()
    return JSONResponse(status_code=500, content=payload)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler to ensure consistent error responses."""
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
    """Normalize HTTPException into ErrorResponse shape."""
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
    """Return validation errors in a consistent envelope."""
    request_id = getattr(request.state, "request_id", None)
    details = {"errors": exc.errors()}
    payload = ErrorResponse(
        code="validation_error",
        message="Invalid request",
        request_id=request_id,
        details=details,
    ).model_dump()
    return JSONResponse(status_code=422, content=payload)


app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(upload_router, prefix="/api/v1/upload", tags=["data-management"])
app.include_router(health_router, prefix="/api/v1/health", tags=["health"])
app.include_router(mock_router, prefix="/api/v1/test", tags=["mock"])
app.include_router(
    docs_router, prefix="", tags=["docs"]
)  # only path-level tags applied within router

app.include_router(form_router, prefix="/form", tags=["form"])
