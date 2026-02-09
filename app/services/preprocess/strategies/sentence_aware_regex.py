"""Sentence-aware chunking strategies using regex-based sentence detection."""
import logging
import re
import uuid
from rag.app.schemas.data import Chunk
from rag.app.services.preprocess.constants import (
    TOKENS_PER_CHUNK,
    DIVIDED_CHUNK_SIZE,
    DIVIDED_SUBDIVISIONS,
    SENTENCE_HARD_MAX_MULTIPLIER,
)

logger = logging.getLogger(__name__)


def ends_with_sentence_punctuation(text: str) -> bool:
    """
    Check if text ends with sentence-ending punctuation using regex.
    
    Handles common cases:
    - Basic punctuation: . ! ?
    - With quotes: ." !" ?"
    - Multiple punctuation: ... !? ?!
    - Smart quotes: ." !" ?"
    
    Args:
        text: The text to check
        
    Returns:
        True if text ends with sentence-ending punctuation
    """
    if not text:
        return False
    
    # Pattern: sentence-ending punctuation, optionally followed by quotes/whitespace
    # \s* handles trailing whitespace
    # ["']* handles quotes (both regular and smart quotes)
    pattern = r'[.!?…]+["\'""\']*\s*$'
    
    return bool(re.search(pattern, text.rstrip()))


def build_chunks_sentence_fixed_srt_regex(
    sub_entries: list[dict],
    name_space: str,
    build_chunk_lines_fn,
    create_chunk_fn,
    tokens_per_chunk: int = TOKENS_PER_CHUNK,
    hard_max_multiplier: float = SENTENCE_HARD_MAX_MULTIPLIER,
) -> list[Chunk]:
    """
    SENTENCE_FIXED_REGEX chunking strategy for SRT files.
    
    Sentence-aware chunking with deferred sentence boundary checking:
    1. Accumulate SRT lines until threshold (no sentence checking for efficiency)
    2. Once threshold exceeded, start checking for sentence-ending punctuation
    3. Look ahead line-by-line until sentence completes
    4. Create chunk, with next chunk overlapping from first line that exceeded threshold
    5. Hard max: stop lookahead even if sentence incomplete
    6. Merge tiny final chunks (like FIXED_SIZE)
    
    Key features:
    - Preserves complete SRT line integrity (no text splitting)
    - Creates variable-sized chunks that respect sentence boundaries
    - Overlap ensures context continuity between chunks
    
    :param sub_entries: List of flattened subtitle entries (from _flatten_subs).
    :param name_space: The name of the file or namespace for this chunk.
    :param build_chunk_lines_fn: Function to build chunk lines (timestamp segments).
    :param create_chunk_fn: Function to create a Chunk from entries.
    :param tokens_per_chunk: Target number of tokens per chunk (default: 200).
    :param hard_max_multiplier: Multiplier for hard max (default: 2.0, so 400 tokens).
    :return: List of Chunk objects with sentence-aware boundaries.
    """
    if not sub_entries:
        return []
    
    hard_max = int(tokens_per_chunk * hard_max_multiplier)
    min_chunk_size = max(50, tokens_per_chunk // 2)
    
    chunks = []
    last_chunk_entries = None
    i = 0
    
    while i < len(sub_entries):
        current_entries = []
        current_tokens = 0
        threshold_exceeded_at = None
        
        while i < len(sub_entries):
            entry = sub_entries[i]
            current_entries.append(entry)
            current_tokens += entry["tokens"]
            
            # Hard max check: immediate stop regardless of sentence boundary
            if current_tokens >= hard_max:
                logger.debug(
                    f"Hard max ({hard_max}) reached at {current_tokens} tokens. "
                    f"Chunking even if sentence incomplete."
                )
                chunk = create_chunk_fn(current_entries, name_space)
                chunks.append(chunk)
                last_chunk_entries = current_entries.copy()
                i += 1  # No overlap on hard max
                break
            
            # Check if threshold exceeded
            if current_tokens >= tokens_per_chunk:
                # Mark the first line where we exceeded threshold (for overlap)
                if threshold_exceeded_at is None:
                    threshold_exceeded_at = i
                    logger.debug(
                        f"Threshold ({tokens_per_chunk}) exceeded at line {i}, "
                        f"tokens={current_tokens}. Starting sentence boundary check."
                    )
                
                # NOW check for sentence ending (only after threshold exceeded)
                current_text = " ".join(e["text"] for e in current_entries)
                
                if ends_with_sentence_punctuation(current_text):
                    # Sentence complete: chunk here
                    logger.debug(
                        f"Sentence complete at {current_tokens} tokens. Creating chunk."
                    )
                    chunk = create_chunk_fn(current_entries, name_space)
                    chunks.append(chunk)
                    last_chunk_entries = current_entries.copy()
                    
                    # Next chunk starts from where threshold was exceeded (overlap)
                    i = threshold_exceeded_at
                    break
                else:
                    # Sentence incomplete: continue lookahead
                    logger.debug(
                        f"Sentence incomplete at {current_tokens} tokens. "
                        f"Looking ahead to next line..."
                    )
                    i += 1
                    continue
            else:
                # UNDER threshold: just accumulate (no sentence checking)
                i += 1
                continue
        
        # Handle remaining entries at end of document
        if i >= len(sub_entries) and current_entries:
            # Check if we already added this chunk
            if not chunks or current_entries != last_chunk_entries:
                remaining_tokens = sum(e["tokens"] for e in current_entries)
                
                # Merge tiny final chunks (like FIXED_SIZE)
                if remaining_tokens < min_chunk_size and last_chunk_entries is not None:
                    logger.debug(
                        f"Final chunk too small ({remaining_tokens} tokens). "
                        f"Merging with previous chunk."
                    )
                    merged_entries = last_chunk_entries + current_entries
                    chunks[-1] = create_chunk_fn(merged_entries, name_space)
                else:
                    chunk = create_chunk_fn(current_entries, name_space)
                    chunks.append(chunk)
            break
    
    return chunks


def build_chunks_sentence_divided_srt_regex(
    sub_entries: list[dict],
    name_space: str,
    build_chunk_lines_fn,
    compute_text_hash_fn,
    subdivide_entries_fn,
    chunk_size: int = DIVIDED_CHUNK_SIZE,
    subdivisions: int = DIVIDED_SUBDIVISIONS,
    hard_max_multiplier: float = SENTENCE_HARD_MAX_MULTIPLIER,
) -> list[Chunk]:
    """
    SENTENCE_DIVIDED_REGEX chunking strategy for SRT files.
    
    Combines sentence-aware chunking with subdivision:
    1. Create sentence-aware main chunks (like SENTENCE_FIXED_REGEX)
    2. Subdivide each main chunk into N subdivisions
    3. All subdivisions share full_text_id and full_text
    4. Each subdivision embeds a different portion
    
    :param sub_entries: List of flattened subtitle entries.
    :param name_space: The name of the file.
    :param build_chunk_lines_fn: Function to build chunk lines.
    :param compute_text_hash_fn: Function to compute text hash.
    :param subdivide_entries_fn: Function to subdivide entries.
    :param chunk_size: Size of main chunk before division (default: 800).
    :param subdivisions: Number of subdivisions per main chunk (default: 4).
    :param hard_max_multiplier: Multiplier for hard max.
    :return: List of subdivided Chunk objects with sentence-aware boundaries.
    """
    if not sub_entries:
        return []
    
    hard_max = int(chunk_size * hard_max_multiplier)
    min_chunk_size = max(100, chunk_size // 4)
    
    chunks = []
    last_chunk_entries = None
    last_chunk_subdivisions_count = 0
    i = 0
    
    while i < len(sub_entries):
        current_entries = []
        current_tokens = 0
        threshold_exceeded_at = None
        
        while i < len(sub_entries):
            entry = sub_entries[i]
            current_entries.append(entry)
            current_tokens += entry["tokens"]
            
            # Hard max check
            if current_tokens >= hard_max:
                logger.debug(
                    f"Hard max ({hard_max}) reached. Creating subdivided chunks."
                )
                subdivided_chunks = subdivide_entries_fn(
                    current_entries, name_space, build_chunk_lines_fn, 
                    compute_text_hash_fn, subdivisions
                )
                chunks.extend(subdivided_chunks)
                last_chunk_entries = current_entries.copy()
                last_chunk_subdivisions_count = len(subdivided_chunks)
                i += 1
                break
            
            # Check if threshold exceeded
            if current_tokens >= chunk_size:
                if threshold_exceeded_at is None:
                    threshold_exceeded_at = i
                    logger.debug(
                        f"Chunk size threshold ({chunk_size}) exceeded at {current_tokens} tokens."
                    )
                
                # Check for sentence ending
                current_text = " ".join(e["text"] for e in current_entries)
                
                if ends_with_sentence_punctuation(current_text):
                    # Create subdivided chunks
                    logger.debug(
                        f"Sentence complete. Creating {subdivisions} subdivisions."
                    )
                    subdivided_chunks = subdivide_entries_fn(
                        current_entries, name_space, build_chunk_lines_fn,
                        compute_text_hash_fn, subdivisions
                    )
                    chunks.extend(subdivided_chunks)
                    last_chunk_entries = current_entries.copy()
                    last_chunk_subdivisions_count = len(subdivided_chunks)
                    
                    # Overlap from threshold line
                    i = threshold_exceeded_at
                    break
                else:
                    # Sentence incomplete: continue lookahead
                    i += 1
                    continue
            else:
                # Under threshold: accumulate
                i += 1
                continue
        
        # Handle remaining entries
        if i >= len(sub_entries) and current_entries:
            if not chunks or current_entries != last_chunk_entries:
                remaining_tokens = sum(e["tokens"] for e in current_entries)
                
                # Merge tiny chunks
                if remaining_tokens < min_chunk_size and last_chunk_entries is not None:
                    logger.debug(
                        f"Merging small final chunk with previous. Re-subdividing."
                    )
                    merged_entries = last_chunk_entries + current_entries
                    chunks = chunks[:-last_chunk_subdivisions_count]
                    subdivided_chunks = subdivide_entries_fn(
                        merged_entries, name_space, build_chunk_lines_fn,
                        compute_text_hash_fn, subdivisions
                    )
                    chunks.extend(subdivided_chunks)
                else:
                    subdivided_chunks = subdivide_entries_fn(
                        current_entries, name_space, build_chunk_lines_fn,
                        compute_text_hash_fn, subdivisions
                    )
                    chunks.extend(subdivided_chunks)
            break
    
    return chunks


def build_chunks_sentence_fixed_txt_regex(
    tokens: list[int],
    encoder,
    name_space: str,
    compute_text_hash_fn,
    tokens_per_chunk: int = TOKENS_PER_CHUNK,
    hard_max_multiplier: float = SENTENCE_HARD_MAX_MULTIPLIER,
) -> list[Chunk]:
    """
    SENTENCE_FIXED_REGEX chunking strategy for plain text files.
    
    Similar to SRT strategy but for plain text:
    1. Accumulate tokens until threshold
    2. Check for sentence-ending punctuation
    3. Continue until sentence completes
    4. No overlap (TXT doesn't have line-level structure like SRT)
    
    :param tokens: Token list from tiktoken encoding.
    :param encoder: Tiktoken encoder instance.
    :param name_space: The name of the file.
    :param compute_text_hash_fn: Function to compute text hash.
    :param tokens_per_chunk: Target tokens per chunk.
    :param hard_max_multiplier: Multiplier for hard max.
    :return: List of sentence-aware Chunk objects.
    """
    chunks = []
    total_tokens = len(tokens)
    hard_max = int(tokens_per_chunk * hard_max_multiplier)
    min_chunk_size = max(50, tokens_per_chunk // 2)
    
    i = 0
    
    while i < total_tokens:
        # Accumulate until threshold
        lookahead_idx = min(tokens_per_chunk, total_tokens - i)
        chunk_tokens = tokens[i:i + lookahead_idx]
        chunk_text = encoder.decode(chunk_tokens)
        
        # If we have more tokens ahead and at/over threshold, check sentence boundary
        if i + lookahead_idx < total_tokens and lookahead_idx >= tokens_per_chunk:
            # Look ahead token-by-token until sentence completes or hard max
            max_lookahead = min(hard_max, total_tokens - i)
            
            while lookahead_idx < max_lookahead:
                chunk_tokens = tokens[i:i + lookahead_idx]
                chunk_text = encoder.decode(chunk_tokens)
                
                if ends_with_sentence_punctuation(chunk_text):
                    break
                
                lookahead_idx += 1
            
            chunk_tokens = tokens[i:i + lookahead_idx]
            chunk_text = encoder.decode(chunk_tokens)
        
        current_token_count = len(chunk_tokens)
        
        # Create chunk
        text_hash = compute_text_hash_fn(chunk_text)
        
        chunk = Chunk(
            full_text_id=uuid.uuid4(),
            name_space=name_space,
            text_to_embed=chunk_text,
            chunk_size=current_token_count,
            time_start=None,
            time_end=None,
            full_text=chunk_text,
            embed_size=current_token_count,
            text_hash=text_hash,
        )
        chunks.append(chunk)
        
        # Move forward
        i += current_token_count
    
    # Handle small final chunk (merge with previous)
    if chunks and len(chunks) > 1:
        last_chunk = chunks[-1]
        if last_chunk.chunk_size < min_chunk_size:
            merged_text = chunks[-2].text_to_embed + " " + last_chunk.text_to_embed
            merged_tokens = len(encoder.encode(merged_text))
            text_hash = compute_text_hash_fn(merged_text)
            
            chunks[-2] = Chunk(
                full_text_id=uuid.uuid4(),
                name_space=name_space,
                text_to_embed=merged_text,
                chunk_size=merged_tokens,
                time_start=None,
                time_end=None,
                full_text=merged_text,
                embed_size=merged_tokens,
                text_hash=text_hash,
            )
            chunks.pop()
    
    logger.debug(f"Created {len(chunks)} sentence-aware chunks from {name_space}")
    return chunks


def build_chunks_sentence_divided_txt_regex(
    tokens: list[int],
    encoder,
    name_space: str,
    compute_text_hash_fn,
    chunk_size: int = DIVIDED_CHUNK_SIZE,
    subdivisions: int = DIVIDED_SUBDIVISIONS,
    hard_max_multiplier: float = SENTENCE_HARD_MAX_MULTIPLIER,
) -> list[Chunk]:
    """
    SENTENCE_DIVIDED_REGEX chunking strategy for plain text files.
    
    Creates sentence-aware main chunks, then subdivides them.
    
    :param tokens: Token list from tiktoken encoding.
    :param encoder: Tiktoken encoder instance.
    :param name_space: The name of the file.
    :param compute_text_hash_fn: Function to compute text hash.
    :param chunk_size: Size of main chunk before division.
    :param subdivisions: Number of subdivisions per main chunk.
    :param hard_max_multiplier: Multiplier for hard max.
    :return: List of subdivided sentence-aware Chunk objects.
    """
    chunks = []
    total_tokens = len(tokens)
    hard_max = int(chunk_size * hard_max_multiplier)
    
    i = 0
    
    while i < total_tokens:
        # Accumulate until chunk_size
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoder.decode(chunk_tokens)
        current_token_count = len(chunk_tokens)
        
        # Look ahead for sentence boundary if more tokens available
        if i + chunk_size < total_tokens:
            lookahead_idx = chunk_size
            
            while i + lookahead_idx < total_tokens and lookahead_idx < hard_max:
                chunk_tokens = tokens[i:i + lookahead_idx]
                chunk_text = encoder.decode(chunk_tokens)
                
                if ends_with_sentence_punctuation(chunk_text):
                    break
                
                lookahead_idx += min(20, chunk_size // 20)
            
            chunk_tokens = tokens[i:i + lookahead_idx]
            chunk_text = encoder.decode(chunk_tokens)
            current_token_count = len(chunk_tokens)
        
        # Subdivide this main chunk
        shared_text_id = uuid.uuid4()
        tokens_per_subdivision = current_token_count // subdivisions
        
        for j in range(subdivisions):
            if j == subdivisions - 1:
                sub_tokens = chunk_tokens[j * tokens_per_subdivision:]
            else:
                start_idx = j * tokens_per_subdivision
                end_idx = (j + 1) * tokens_per_subdivision
                sub_tokens = chunk_tokens[start_idx:end_idx]
            
            if not sub_tokens:
                continue
            
            text_to_embed = encoder.decode(sub_tokens)
            sub_token_count = len(sub_tokens)
            text_hash = compute_text_hash_fn(text_to_embed)
            
            chunk = Chunk(
                full_text_id=shared_text_id,
                name_space=name_space,
                text_to_embed=text_to_embed,
                chunk_size=current_token_count,
                time_start=None,
                time_end=None,
                full_text=chunk_text,
                embed_size=sub_token_count,
                text_hash=text_hash,
            )
            chunks.append(chunk)
        
        i += current_token_count
    
    return chunks
