EMBEDDING_TEXT_SIZE = 50
FULL_TEXT_SIZE = 200
CHUNKS_PER_SEGMENT = 4  # Number of chunks to divide each segment into

# Semantic chunking constants
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Threshold for making cuts
MIN_CHUNK_TOKENS = 50  # Minimum tokens per chunk to avoid tiny chunks
MAX_CHUNK_TOKENS = 500  # Maximum tokens per chunk as a safety limit (reduced from 1000)
SEMANTIC_ENCODING = "cl100k_base"  # Tokenizer encoding (for GPT-4, GPT-3.5)

# Sliding window constants
SLIDING_WINDOW_SIZE = 250  # Tokens per chunk (reduced from 500)
SLIDING_WINDOW_OVERLAP = 50  # Token overlap between consecutive chunks (reduced from 100)
SLIDING_WINDOW_ENCODING = "cl100k_base"  # Tokenizer encoding (for GPT-4, GPT-3.5)
SLIDING_WINDOW_SEGMENT_SIZE = 2000  # Process transcript in segments to avoid metadata limits
