"""FIXED_SIZE chunking strategy implementation."""
import logging
import uuid
from rag.app.schemas.data import Chunk
from rag.app.services.preprocess.constants import TOKENS_PER_CHUNK, SRT_LINES_PER_SEGMENT

logger = logging.getLogger(__name__)


def build_chunks_fixed_size_srt(
    sub_entries: list[dict],
    name_space: str,
    build_chunk_lines_fn,
    create_chunk_fn,
    tokens_per_chunk: int = TOKENS_PER_CHUNK,
) -> list[Chunk]:
    """
    FIXED_SIZE chunking strategy for SRT: accumulate subtitles until reaching token limit.
    Uses a "soft limit" approach: once the token limit is exceeded, continues until
    completing the current 6-line segment to avoid creating small orphaned segments.
    
    Also enforces a minimum chunk size: if remaining entries at the end are below
    a threshold (e.g., 50% of tokens_per_chunk), they are merged into the previous chunk
    instead of creating a tiny final chunk.
    
    Each chunk is independent with its own UUID. The text is embedded as-is,
    while metadata stores fine-grained timestamps as tuples.

    :param sub_entries: List of flattened subtitle entries (from _flatten_subs).
    :param name_space: The name of the file or namespace for this chunk.
    :param build_chunk_lines_fn: Function to build chunk lines (timestamp segments).
    :param create_chunk_fn: Function to create a Chunk from entries.
    :param tokens_per_chunk: Target number of tokens per chunk (default: 200 from TOKENS_PER_CHUNK).
    :return: List of Chunk objects, each with independent UUID.
    """
    if not sub_entries:
        return []

    chunks = []
    
    current_entries = []
    current_tokens = 0
    
    # Minimum chunk size threshold (e.g., 50% of target, or absolute minimum like 50 tokens)
    MIN_CHUNK_SIZE = max(50, tokens_per_chunk // 2)  # At least 50 tokens or 50% of target
    
    # Track the last chunk's entries for potential merging
    last_chunk_entries = None
    
    for i, entry in enumerate(sub_entries):
        # Add entry to current batch
        current_entries.append(entry)
        current_tokens += entry["tokens"]
        
        # Check if we should create a chunk (soft limit approach):
        # 1. We've exceeded the token limit
        # 2. AND we've completed a 6-line segment (or we're at the end)
        exceeded_limit = current_tokens >= tokens_per_chunk
        segment_position = len(current_entries) % SRT_LINES_PER_SEGMENT
        at_segment_boundary = (segment_position == 0)
        at_end = (i == len(sub_entries) - 1)
        
        if exceeded_limit and (at_segment_boundary or at_end):
            # Create chunk from accumulated entries
            chunk = create_chunk_fn(current_entries, name_space)
            chunks.append(chunk)
            
            # Store entries for potential merging with small final chunk
            last_chunk_entries = current_entries.copy()
            
            # Reset for next chunk
            current_entries = []
            current_tokens = 0
    
    # Handle remaining entries with minimum size check
    if current_entries:
        remaining_tokens = sum(entry["tokens"] for entry in current_entries)
        
        # If remaining entries are too small, merge into previous chunk
        if remaining_tokens < MIN_CHUNK_SIZE and last_chunk_entries is not None:
            # Merge remaining entries into the last chunk
            merged_entries = last_chunk_entries + current_entries
            chunks[-1] = create_chunk_fn(merged_entries, name_space)
        else:
            # Remaining entries are substantial enough to be their own chunk
            chunk = create_chunk_fn(current_entries, name_space)
            chunks.append(chunk)
    
    return chunks


def build_chunks_fixed_size_txt(
    tokens: list[int],
    encoder,
    name_space: str,
    compute_text_hash_fn,
    tokens_per_chunk: int = TOKENS_PER_CHUNK,
) -> list[Chunk]:
    """
    FIXED_SIZE chunking strategy for TXT: Independent chunks of TOKENS_PER_CHUNK.
    
    :param tokens: Token list from tiktoken encoding.
    :param encoder: Tiktoken encoder instance.
    :param name_space: The name of the file.
    :param compute_text_hash_fn: Function to compute text hash.
    :param tokens_per_chunk: Number of tokens per chunk.
    :return: List of Chunk objects with independent UUIDs.
    """
    chunks: list[Chunk] = []
    total_tokens = len(tokens)
    
    for i in range(0, total_tokens, tokens_per_chunk):
        chunk_tokens = tokens[i : i + tokens_per_chunk]
        chunk_text = encoder.decode(chunk_tokens)
        chunk_token_count = len(chunk_tokens)
        
        # Compute SHA-256 hash of the text to embed
        text_hash = compute_text_hash_fn(chunk_text)

        chunk = Chunk(
            full_text_id=uuid.uuid4(),  # Independent UUID
            name_space=name_space,
            text_to_embed=chunk_text,
            chunk_size=chunk_token_count,
            time_start=None,
            time_end=None,
            full_text=chunk_text,  # Same as text_to_embed for TXT
            embed_size=chunk_token_count,
            text_hash=text_hash,
        )
        chunks.append(chunk)

    logger.debug(f"Created {len(chunks)} FIXED_SIZE chunks from {name_space}")
    return chunks
