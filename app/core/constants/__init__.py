from rag.app.core.constants.app import DATABASE_CONFIGURATION, DEV_OUTPUTS, ENVIRONMENT
from rag.app.core.constants.llm import (
    CHUNKING_STRATEGY,
    EMBEDDING_CONFIGURATION,
    EXTERNAL_API_TIMEOUT,
    LLM_CONFIGURATION,
)
from rag.app.core.constants.mongo import (
    COLLECTION_INDEX,
    COLLECTIONS,
    EXCEPTIONS_COLLECTION,
    MAX_RETRY_ATTEMPTS,
    METRICS_COLLECTION,
    MONGODB_DB_NAME,
    MONGODB_VECTOR_COLLECTION,
    RETRY_BACKOFF_MULTIPLIER,
    RETRY_DELAY_SECONDS,
    RETRIEVAL_TIMEOUT_MS,
    VECTOR_PATH,
)
from rag.app.core.constants.supabase import SUPABASE_URL
from rag.app.core.constants.upstash import (
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    UPSTASH_REDIS_REST_URL,
    USER_RATE_LIMIT_MAX_REQUESTS_PER_MONTH,
)
from rag.app.core.constants.vertex import GOOGLE_CLOUD_PROJECT_ID, VERTEX_REGION
from rag.app.schemas.data import DataBaseConfiguration

# Vector-store-specific constants — only the active backend is imported.
if DATABASE_CONFIGURATION == DataBaseConfiguration.PINECONE:
    from rag.app.core.constants.pinecone import (  # noqa: F401
        PINECONE_ENVIRONMENT,
        PINECONE_HOST,
        PINECONE_INDEX_NAME,
        PINECONE_NAMESPACE,
    )
