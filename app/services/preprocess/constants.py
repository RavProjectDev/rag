# FIXED_SIZE strategy
TOKENS_PER_CHUNK = 200  # Fixed size: number of tokens per chunk

# DIVIDED strategy
DIVIDED_CHUNK_SIZE = 800  # Total tokens per main chunk before division
DIVIDED_SUBDIVISIONS = 4  # Number of subdivisions per main chunk

# SENTENCE-AWARE strategies
SENTENCE_HARD_MAX_MULTIPLIER = 2.0  # Hard max = threshold * multiplier (prevents runaway chunks)

# Common settings
SRT_LINES_PER_SEGMENT = 6  # Number of SRT subtitle lines to merge per timestamp segment
TIKTOKEN_ENCODING = "cl100k_base"  # OpenAI's encoding for GPT-4, GPT-3.5-turbo
