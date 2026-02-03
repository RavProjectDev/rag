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
from rag.app.api.v1.info import router as info_router
from rag.app.api.v1.mock import router as mock_router
from rag.app.api.v1.form import router as form_router
from rag.app.api.v1.prompt import router as prompt_router
from rag.app.api.v1.user import router as user_router
from rag.app.db.connections import MetricsConnection, ExceptionsLogger

from rag.app.db.mongodb_connection import (
    MongoEmbeddingStore,
    MongoMetricsConnection,
    MongoExceptionsLogger,
)
from rag.app.db.pinecone_connection import PineconeEmbeddingStore
from rag.app.db.redis_connection import RedisConnection
from rag.app.core.config import get_settings, Environment
from rag.app.core.scheduler import start_scheduler
from rag.app.schemas.response import ErrorResponse
from rag.app.schemas.data import DataBaseConfiguration

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
        metrics_collection = db[settings.metrics_collection]
        exceptions_collection = db[settings.exceptions_collection]

        # Embedding connection selection
        if settings.database_configuration == DataBaseConfiguration.PINECONE:
            required_pinecone = {
                "pinecone_api_key": settings.pinecone_api_key,
            }
            missing = [k for k, v in required_pinecone.items() if not v]
            if missing:
                raise ValueError(
                    f"Missing required Pinecone settings: {', '.join(missing)}"
                )
            
            # Index name is based on embedding model
            # Use configured index name if provided, otherwise derive from embedding model
            index_name = settings.pinecone_index_name or settings.embedding_configuration.value
            
            # Namespace is based on chunking strategy
            # Use configured namespace if provided, otherwise use chunking strategy
            namespace = settings.pinecone_namespace or settings.chunking_strategy.value
            
            logger.info(
                f"[PINECONE CONFIG] index={index_name} (embedding_model={settings.embedding_configuration.value}), "
                f"namespace={namespace} (chunking_strategy={settings.chunking_strategy.value})"
            )
            
            # Get chunks collection for tracking (even when using Pinecone)
            chunks_collection = db["chunks"]
            logger.info(f"MongoDB chunks collection: chunks (for Pinecone chunk tracking)")
            
            embedding_connection = PineconeEmbeddingStore(
                api_key=settings.pinecone_api_key,
                index_name=index_name,
                environment=settings.pinecone_environment,
                namespace=namespace,
                host=settings.pinecone_host,
                chunks_collection=chunks_collection,
            )
        else:
            # For MongoDB, append chunking strategy to collection name if not using default collection
            collection_name = settings.mongodb_vector_collection
            # If collection doesn't already include strategy, append it
            if settings.chunking_strategy.value not in collection_name:
                collection_name = f"{collection_name}_{settings.chunking_strategy.value}"
            
            logger.info(f"MongoDB collection: {collection_name} (based on chunking_strategy: {settings.chunking_strategy.value})")
            vector_embedding_collection = db[collection_name]
            chunks_collection = db["chunks"]
            logger.info(f"MongoDB chunks collection: chunks")
            embedding_connection = MongoEmbeddingStore(
                collection=vector_embedding_collection,
                index=settings.collection_index,
                vector_path=settings.vector_path,
                chunks_collection=chunks_collection,
            )
        metrics_connection = MongoMetricsConnection(
            collection=metrics_collection,
        )
        exceptions_logger = MongoExceptionsLogger(
            collection=exceptions_collection,
        )
        
        # Initialize Redis connection if credentials are available
        redis_conn = None
        if settings.upstash_redis_rest_url and settings.upstash_redis_rest_token:
            try:
                redis_conn = RedisConnection(
                    url=settings.upstash_redis_rest_url,
                    token=settings.upstash_redis_rest_token,
                )
                logger.info("Redis connection initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis connection: {e}. Rate limiting will be disabled.")
        else:
            logger.info("Redis credentials not provided. Rate limiting will be disabled.")

        app.state.embedding_conn = embedding_connection
        # Backwards compatibility for code still accessing mongo_conn
        app.state.mongo_conn = embedding_connection
        app.state.metrics_connection = metrics_connection
        app.state.exceptions_logger = exceptions_logger
        app.state.redis_conn = redis_conn
        app.state.db_client = db
        
        # Sync scheduler disabled - keeping sync scripts but not running them automatically
        # if (
        #     settings.environment == Environment.PRD
        #     and settings.database_configuration == DataBaseConfiguration.MONGO
        # ):
        #     logger.info("Starting sync scheduler (PRD environment)")
        #     start_scheduler(
        #         connection=embedding_connection,
        #         embedding_configuration=settings.embedding_configuration,
        #         chunking_strategy=settings.chunking_strategy,
        #     )
        # else:
        #     logger.info(
        #         "Skipping sync scheduler (environment=%s, db=%s)",
        #         settings.environment,
        #         settings.database_configuration.value,
        #     )
        logger.info("Sync scheduler disabled (not running automatic sync jobs)")
        
        yield

    except Exception as e:
        logger.error(f"Failed to initialize application resources: {e}")
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
        
        # Add user rate limit headers if available
        if hasattr(request.state, "user_rate_limit_limit"):
            response.headers["X-RateLimit-Limit"] = str(request.state.user_rate_limit_limit)
            response.headers["X-RateLimit-Remaining"] = str(request.state.user_rate_limit_remaining)
            response.headers["X-RateLimit-Reset"] = str(request.state.user_rate_limit_reset)
        
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
app.include_router(info_router, prefix="/api/v1/info", tags=["info"])
app.include_router(mock_router, prefix="/api/v1/test", tags=["mock"])
app.include_router(prompt_router, prefix="/api/v1/prompt", tags=["prompt"])
app.include_router(user_router, prefix="/api/v1/user", tags=["user"])
app.include_router(
    docs_router, prefix="", tags=["docs"]
)  # only path-level tags applied within router

app.include_router(form_router, prefix="/form", tags=["form"])
