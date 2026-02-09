"""DIVIDED chunking strategy implementation."""
import logging
import uuid
from rag.app.schemas.data import Chunk
from rag.app.services.preprocess.constants import (
    DIVIDED_CHUNK_SIZE,
    DIVIDED_SUBDIVISIONS,
    SRT_LINES_PER_SEGMENT,
)

logger = logging.getLogger(__name__)


def _subdivide_entries(
    entries: list[dict],
    name_space: str,
    build_chunk_lines_fn,
    compute_text_hash_fn,
    subdivisions: int,
) -> list[Chunk]:
    """
    Subdivide a list of subtitle entries into N chunks that share the same full_text.
    
    All resulting chunks will have the same full_text_id and full_text metadata,
    but each will embed a different portion (subdivision) of the text.
    """
    if not entries:
        return []
    
    # Generate ONE shared UUID for all subdivisions
    shared_text_id = uuid.uuid4()
    
    # Get full concatenated text and metadata for all entries
    full_text_str = " ".join(entry["text"] for entry in entries)
    chunk_lines = build_chunk_lines_fn(entries, lines_per_segment=SRT_LINES_PER_SEGMENT)
    time_start = chunk_lines[0][1][0] if chunk_lines else None
    time_end = chunk_lines[-1][1][1] if chunk_lines else None
    total_tokens = sum(entry["tokens"] for entry in entries)
    
    # Calculate tokens per subdivision
    tokens_per_subdivision = total_tokens // subdivisions
    
    chunks = []
    current_idx = 0
    
    for i in range(subdivisions):
        # For last subdivision, take all remaining entries
        if i == subdivisions - 1:
            subdivision_entries = entries[current_idx:]
        else:
            # Collect entries for this subdivision
            subdivision_entries = []
            subdivision_tokens = 0
            while current_idx < len(entries) and subdivision_tokens < tokens_per_subdivision:
                entry = entries[current_idx]
                subdivision_entries.append(entry)
                subdivision_tokens += entry["tokens"]
                current_idx += 1
        
        if not subdivision_entries:
            continue
            
        # Text to embed is just this subdivision's text
        text_to_embed = " ".join(entry["text"] for entry in subdivision_entries)
        embed_tokens = sum(entry["tokens"] for entry in subdivision_entries)
        
        # Compute SHA-256 hash of the text to embed
        text_hash = compute_text_hash_fn(text_to_embed)
        
        chunk = Chunk(
            full_text_id=shared_text_id,  # SHARED across all subdivisions
            time_start=time_start,  # Same start/end for all subdivisions
            time_end=time_end,
            full_text=chunk_lines,  # SHARED - full context with timestamps
            text_to_embed=text_to_embed,  # UNIQUE - just this subdivision
            chunk_size=total_tokens,  # SHARED - total tokens in main chunk
            embed_size=embed_tokens,  # UNIQUE - tokens in this subdivision
            name_space=name_space,
            text_hash=text_hash,
        )
        chunks.append(chunk)
    
    return chunks


def build_chunks_divided_srt(
    sub_entries: list[dict],
    name_space: str,
    build_chunk_lines_fn,
    compute_text_hash_fn,
    chunk_size: int = DIVIDED_CHUNK_SIZE,
    subdivisions: int = DIVIDED_SUBDIVISIONS,
) -> list[Chunk]:
    """
    DIVIDED chunking strategy for SRT: accumulate subtitles to chunk_size, then subdivide.
    Uses a "soft limit" approach: once the token limit is exceeded, continues until
    completing the current 6-line segment to avoid creating small orphaned segments.
    
    Also enforces a minimum chunk size: if remaining entries at the end are below
    a threshold, they are merged into the previous chunk instead of creating tiny final chunks.
    
    Creates a main chunk of chunk_size tokens, then divides it into N subdivisions.
    All subdivisions share the same full_text_id and full_text metadata, but each
    has its own text_to_embed portion for independent embedding.

    :param sub_entries: List of flattened subtitle entries (from _flatten_subs).
    :param name_space: The name of the file or namespace for this chunk.
    :param build_chunk_lines_fn: Function to build chunk lines (timestamp segments).
    :param compute_text_hash_fn: Function to compute text hash.
    :param chunk_size: Size of main chunk before division (default: 800 from DIVIDED_CHUNK_SIZE).
    :param subdivisions: Number of subdivisions per main chunk (default: 4 from DIVIDED_SUBDIVISIONS).
    :return: List of Chunk objects with shared full_text_id per main chunk.
    """
    if not sub_entries:
        return []

    chunks = []
    
    current_entries = []
    current_tokens = 0
    
    # Minimum chunk size threshold - for DIVIDED, we need enough tokens to meaningfully subdivide
    # Use a higher threshold since we're dividing further (e.g., 25% of chunk_size)
    MIN_CHUNK_SIZE = max(100, chunk_size // 4)  # At least 100 tokens or 25% of chunk_size
    
    # Track the last chunk's entries and number of subdivisions for potential merging
    last_chunk_entries = None
    last_chunk_subdivisions_count = 0
    
    for i, entry in enumerate(sub_entries):
        # Add entry to current batch
        current_entries.append(entry)
        current_tokens += entry["tokens"]
        
        # Check if we should create a chunk (soft limit approach):
        # 1. We've exceeded the token limit
        # 2. AND we've completed a 6-line segment (or we're at the end)
        exceeded_limit = current_tokens >= chunk_size
        segment_position = len(current_entries) % SRT_LINES_PER_SEGMENT
        at_segment_boundary = (segment_position == 0)
        at_end = (i == len(sub_entries) - 1)
        
        if exceeded_limit and (at_segment_boundary or at_end):
            # Create subdivided chunks from accumulated entries
            subdivided_chunks = _subdivide_entries(
                current_entries, name_space, build_chunk_lines_fn, compute_text_hash_fn, subdivisions
            )
            chunks.extend(subdivided_chunks)
            
            # Store entries and count for potential merging with small final chunk
            last_chunk_entries = current_entries.copy()
            last_chunk_subdivisions_count = len(subdivided_chunks)
            
            # Reset for next main chunk
            current_entries = []
            current_tokens = 0
    
    # Handle remaining entries with minimum size check
    if current_entries:
        remaining_tokens = sum(entry["tokens"] for entry in current_entries)
        
        # If remaining entries are too small, merge into previous chunk
        if remaining_tokens < MIN_CHUNK_SIZE and last_chunk_entries is not None:
            # Merge remaining entries into the last chunk and re-subdivide
            merged_entries = last_chunk_entries + current_entries
            # Remove the old subdivided chunks and replace with new ones
            # Remove the last N chunks (the subdivisions from the previous chunk)
            chunks = chunks[:-last_chunk_subdivisions_count]
            # Create new subdivided chunks from merged entries
            subdivided_chunks = _subdivide_entries(
                merged_entries, name_space, build_chunk_lines_fn, compute_text_hash_fn, subdivisions
            )
            chunks.extend(subdivided_chunks)
        else:
            # Remaining entries are substantial enough to be their own chunk
            subdivided_chunks = _subdivide_entries(
                current_entries, name_space, build_chunk_lines_fn, compute_text_hash_fn, subdivisions
            )
            chunks.extend(subdivided_chunks)
    
    return chunks


def build_chunks_divided_txt(
    tokens: list[int],
    encoder,
    name_space: str,
    compute_text_hash_fn,
    chunk_size: int = DIVIDED_CHUNK_SIZE,
    subdivisions: int = DIVIDED_SUBDIVISIONS,
) -> list[Chunk]:
    """
    DIVIDED strategy for plain text files.
    Similar to divided SRT strategy but without timestamp metadata.
    """
    chunks = []
    total_tokens = len(tokens)
    
    # Process text in chunk_size batches
    for i in range(0, total_tokens, chunk_size):
        main_chunk_tokens = tokens[i : i + chunk_size]
        main_chunk_text = encoder.decode(main_chunk_tokens)
        main_chunk_token_count = len(main_chunk_tokens)
        
        # Generate shared UUID for this main chunk's subdivisions
        shared_text_id = uuid.uuid4()
        
        # Subdivide the main chunk
        tokens_per_subdivision = main_chunk_token_count // subdivisions
        
        for j in range(subdivisions):
            # Calculate subdivision boundaries
            if j == subdivisions - 1:
                # Last subdivision gets remaining tokens
                sub_tokens = main_chunk_tokens[j * tokens_per_subdivision:]
            else:
                start_idx = j * tokens_per_subdivision
                end_idx = (j + 1) * tokens_per_subdivision
                sub_tokens = main_chunk_tokens[start_idx:end_idx]
            
            if not sub_tokens:
                continue
                
            text_to_embed = encoder.decode(sub_tokens)
            sub_token_count = len(sub_tokens)
            
            # Compute SHA-256 hash of the text to embed
            text_hash = compute_text_hash_fn(text_to_embed)
            
            chunk = Chunk(
                full_text_id=shared_text_id,  # SHARED across subdivisions
                name_space=name_space,
                text_to_embed=text_to_embed,  # UNIQUE - this subdivision
                chunk_size=main_chunk_token_count,  # SHARED - total tokens in main chunk
                time_start=None,
                time_end=None,
                full_text=main_chunk_text,  # SHARED - full main chunk text
                embed_size=sub_token_count,  # UNIQUE - tokens in this subdivision
                text_hash=text_hash,
            )
            chunks.append(chunk)
    
    return chunks
