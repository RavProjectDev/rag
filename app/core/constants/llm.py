from rag.app.schemas.data import ChunkingStrategy, EmbeddingConfiguration, LLMModel

LLM_CONFIGURATION = LLMModel.GEMINI_FLASH
EMBEDDING_CONFIGURATION = EmbeddingConfiguration.GEMINI
CHUNKING_STRATEGY = ChunkingStrategy.FIXED_SIZE
EXTERNAL_API_TIMEOUT = 60
