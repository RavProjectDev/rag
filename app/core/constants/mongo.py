MONGODB_DB_NAME = "rav_dev"
MONGODB_VECTOR_COLLECTION = "gemini_embeddings_v3"
COLLECTION_INDEX = "vector_index"
METRICS_COLLECTION = "metrics"
EXCEPTIONS_COLLECTION = "exceptions"
VECTOR_PATH = "vector"
COLLECTIONS = ["gemini_embeddings_v2", "chunk_embeddings_gemini_embedding_001"]

RETRIEVAL_TIMEOUT_MS = 2000
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0
RETRY_BACKOFF_MULTIPLIER = 2.0
