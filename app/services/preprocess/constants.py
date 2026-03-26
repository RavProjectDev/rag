# FIXED_SIZE strategy
TOKENS_PER_CHUNK = 200  # Fixed size: number of tokens per chunk

# DIVIDED strategy
DIVIDED_CHUNK_SIZE = 800  # Total tokens per main chunk before division
DIVIDED_SUBDIVISIONS = 4  # Number of subdivisions per main chunk

# SENTENCE-AWARE strategies
SENTENCE_HARD_MAX_MULTIPLIER = 2.0  # Hard max = threshold * multiplier (prevents runaway chunks)

# AGENTIC strategy
AGENTIC_DEFAULT_MODEL = "gemini_flash"  # LLMModel enum value name to use for section detection
AGENTIC_MIN_SECTION_SEGMENTS = 2   # Minimum segments per LLM-identified section (guard against over-splitting)
AGENTIC_MAX_SECTION_SEGMENTS = 30  # Maximum segments per section (guard against under-splitting)

# AGENTIC_MULTI_CALL strategy
AGENTIC_MULTI_CALL_MAX_RETRIES = 5        # Total attempts (1 original + 4 retries)
AGENTIC_MULTI_CALL_RETRY_BASE_DELAY = 5.0 # Seconds; doubles on each retry (exponential backoff)

# Common settings
SRT_LINES_PER_SEGMENT = 6  # Number of SRT subtitle lines to merge per timestamp segment
TIKTOKEN_ENCODING = "cl100k_base"  # OpenAI's encoding for GPT-4, GPT-3.5-turbo
