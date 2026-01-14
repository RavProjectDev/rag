import json
import uuid
import pysrt
import tiktoken
from rag.app.schemas.data import Chunk, TypeOfFormat, ChunkingStrategy
import logging
from rag.app.services.preprocess.constants import (
    TOKENS_PER_CHUNK,
    DIVIDED_CHUNK_SIZE,
    DIVIDED_SUBDIVISIONS,
    SRT_LINES_PER_SEGMENT,
    TIKTOKEN_ENCODING,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tiktoken encoder once at module level
_encoder = None

def _get_encoder():
    """Get or initialize the tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return _encoder

def _count_tokens(text: str) -> int:
    """Count the number of tokens in text using tiktoken."""
    encoder = _get_encoder()
    return len(encoder.encode(text))


def _flatten_subs(subs):
    """
    Flattens subtitle items into entries with text, tokens, and timestamps.
    Each entry represents one subtitle line with its token count.
    """
    flattened = []
    for idx, sub in enumerate(subs):
        clean_text = sub.text.replace("\n", " ")
        start_str = str(sub.start)
        end_str = str(sub.end)
        token_count = _count_tokens(clean_text)
        flattened.append(
            {
                "text": clean_text,
                "tokens": token_count,
                "sub_idx": idx,
                "start": start_str,
                "end": end_str,
            }
        )
    return flattened


def _build_chunk_lines(sub_entries: list[dict], lines_per_segment: int = 6) -> list[tuple[str, tuple[str, str]]]:
    """
    Groups consecutive subtitle entries into merged segments.
    
    Args:
        sub_entries: List of subtitle dictionaries with text, tokens, sub_idx, start, end
        lines_per_segment: Number of subtitle lines to merge per timestamp segment (default: 6)
    
    Returns:
        List of (text, (start, end)) tuples where text is from multiple merged lines
    """
    if not sub_entries:
        return []

    # Group subtitle entries into segments of N lines
    merged_segments: list[tuple[str, tuple[str, str]]] = []
    for i in range(0, len(sub_entries), lines_per_segment):
        segment_entries = sub_entries[i:i + lines_per_segment]
        
        # Merge text from all entries in this segment
        merged_text = " ".join(entry["text"] for entry in segment_entries)
        
        # Use start time from first entry and end time from last entry
        segment_start = segment_entries[0]["start"]
        segment_end = segment_entries[-1]["end"]
        
        merged_segments.append((merged_text, (segment_start, segment_end)))

    return merged_segments


def build_chunks(subs, name_space, tokens_per_chunk=TOKENS_PER_CHUNK) -> list[Chunk]:
    """
    FIXED_SIZE chunking strategy: accumulate subtitles until reaching token limit.
    
    Each chunk is independent with its own UUID. The text is embedded as-is,
    while metadata stores fine-grained timestamps as tuples.

    :param subs: List of subtitle objects (pysrt.SubRipItem).
    :param name_space: The name of the file or namespace for this chunk.
    :param tokens_per_chunk: Fixed number of tokens per chunk (default: 200 from TOKENS_PER_CHUNK).
    :return: List of Chunk objects, each with independent UUID.
    """
    if not subs:
        return []

    sub_stream = _flatten_subs(subs)  # List of subtitle entries with tokens
    chunks = []
    
    current_entries = []
    current_tokens = 0
    
    for entry in sub_stream:
        # If adding this entry would exceed limit and we have content, create chunk
        if current_tokens + entry["tokens"] > tokens_per_chunk and current_entries:
            # Create chunk from accumulated entries
            chunk = _create_chunk_from_entries(current_entries, name_space)
            chunks.append(chunk)
            
            # Reset for next chunk
            current_entries = []
            current_tokens = 0
        
        # Add entry to current batch
        current_entries.append(entry)
        current_tokens += entry["tokens"]
    
    # Handle remaining entries
    if current_entries:
        chunk = _create_chunk_from_entries(current_entries, name_space)
        chunks.append(chunk)
    
    return chunks


def _create_chunk_from_entries(entries: list[dict], name_space: str) -> Chunk:
    """Create a single Chunk from a list of subtitle entries."""
    # Concatenate all text for embedding
    text_to_embed = " ".join(entry["text"] for entry in entries)
    
    # Build timestamp tuples (merge every N lines)
    chunk_lines = _build_chunk_lines(entries, lines_per_segment=SRT_LINES_PER_SEGMENT)
    
    # Get overall time boundaries
    time_start = chunk_lines[0][1][0] if chunk_lines else None
    time_end = chunk_lines[-1][1][1] if chunk_lines else None
    
    # Count tokens
    embed_tokens = sum(entry["tokens"] for entry in entries)
    
    return Chunk(
        full_text_id=uuid.uuid4(),  # Each chunk gets its own UUID
        time_start=time_start,
        time_end=time_end,
        full_text=chunk_lines,  # List of (text, (start, end)) tuples as metadata
        text_to_embed=text_to_embed,  # Actual concatenated text for embedding
        chunk_size=embed_tokens,  # Tokens in this chunk
        embed_size=embed_tokens,  # Same as chunk_size for fixed-size chunking
        name_space=name_space,
    )


def build_chunks_divided(subs, name_space, chunk_size=DIVIDED_CHUNK_SIZE, subdivisions=DIVIDED_SUBDIVISIONS) -> list[Chunk]:
    """
    DIVIDED chunking strategy: accumulate subtitles to chunk_size, then subdivide.
    
    Creates a main chunk of chunk_size tokens, then divides it into N subdivisions.
    All subdivisions share the same full_text_id and full_text metadata, but each
    has its own text_to_embed portion for independent embedding.

    :param subs: List of subtitle objects (pysrt.SubRipItem).
    :param name_space: The name of the file or namespace for this chunk.
    :param chunk_size: Size of main chunk before division (default: 800 from DIVIDED_CHUNK_SIZE).
    :param subdivisions: Number of subdivisions per main chunk (default: 4 from DIVIDED_SUBDIVISIONS).
    :return: List of Chunk objects with shared full_text_id per main chunk.
    """
    if not subs:
        return []

    sub_stream = _flatten_subs(subs)  # List of subtitle entries with tokens
    chunks = []
    
    current_entries = []
    current_tokens = 0
    
    for entry in sub_stream:
        # If adding this entry would exceed chunk_size and we have content, process chunk
        if current_tokens + entry["tokens"] > chunk_size and current_entries:
            # Create subdivided chunks from accumulated entries
            subdivided_chunks = _subdivide_entries(current_entries, name_space, subdivisions)
            chunks.extend(subdivided_chunks)
            
            # Reset for next main chunk
            current_entries = []
            current_tokens = 0
        
        # Add entry to current batch
        current_entries.append(entry)
        current_tokens += entry["tokens"]
    
    # Handle remaining entries
    if current_entries:
        subdivided_chunks = _subdivide_entries(current_entries, name_space, subdivisions)
        chunks.extend(subdivided_chunks)
    
    return chunks


def _subdivide_entries(entries: list[dict], name_space: str, subdivisions: int) -> list[Chunk]:
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
    chunk_lines = _build_chunk_lines(entries, lines_per_segment=SRT_LINES_PER_SEGMENT)
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
        
        chunk = Chunk(
            full_text_id=shared_text_id,  # SHARED across all subdivisions
            time_start=time_start,  # Same start/end for all subdivisions
            time_end=time_end,
            full_text=chunk_lines,  # SHARED - full context with timestamps
            text_to_embed=text_to_embed,  # UNIQUE - just this subdivision
            chunk_size=total_tokens,  # SHARED - total tokens in main chunk
            embed_size=embed_tokens,  # UNIQUE - tokens in this subdivision
            name_space=name_space,
        )
        chunks.append(chunk)
    
    return chunks


def chunk_srt(content: tuple[str, str], strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE) -> list[Chunk]:
    """
    SRT file chunking with strategy selection.
    Parse subtitles and apply the specified chunking strategy.
    """
    file_name, text = content
    logger.debug(f"Chunking SRT file: {file_name} with strategy: {strategy.value}")
    
    subs = pysrt.from_string(text)
    logger.debug(f"Parsed {len(subs)} subtitle segments from {file_name}")
    
    # Apply strategy
    if strategy == ChunkingStrategy.DIVIDED:
        chunks = build_chunks_divided(subs, file_name)
    elif strategy == ChunkingStrategy.FIXED_SIZE:
        chunks = build_chunks(subs, file_name)
    else:
        raise ValueError(f"Unsupported chunking strategy for SRT: {strategy.value}")
    
    logger.debug(f"Created {len(chunks)} chunks from {file_name}")
    return chunks


def build_chunks_divided_txt(text: str, name_space: str, chunk_size=DIVIDED_CHUNK_SIZE, subdivisions=DIVIDED_SUBDIVISIONS) -> list[Chunk]:
    """
    DIVIDED strategy for plain text files.
    Similar to divided SRT strategy but without timestamp metadata.
    """
    encoder = _get_encoder()
    tokens = encoder.encode(text)
    total_tokens = len(tokens)
    
    chunks = []
    
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
            
            chunk = Chunk(
                full_text_id=shared_text_id,  # SHARED across subdivisions
                name_space=name_space,
                text_to_embed=text_to_embed,  # UNIQUE - this subdivision
                chunk_size=main_chunk_token_count,  # SHARED - total tokens in main chunk
                time_start=None,
                time_end=None,
                full_text=main_chunk_text,  # SHARED - full main chunk text
                embed_size=sub_token_count,  # UNIQUE - tokens in this subdivision
            )
            chunks.append(chunk)
    
    return chunks


def chunk_txt(content: tuple[str, str], strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE) -> list[Chunk]:
    """
    Plain text file chunking with strategy selection.

    :param content: A tuple containing (filename, raw text content).
    :param strategy: Chunking strategy to use.
    :return: A list of Chunk objects.
    """
    file_name, text = content
    logger.debug(f"Chunking TXT file: {file_name} with strategy: {strategy.value}")
    
    encoder = _get_encoder()
    tokens = encoder.encode(text)
    total_tokens = len(tokens)
    logger.debug(f"Found {total_tokens} tokens in {file_name}")

    if strategy == ChunkingStrategy.DIVIDED:
        return build_chunks_divided_txt(text, file_name)
    elif strategy == ChunkingStrategy.FIXED_SIZE:
        # FIXED_SIZE: Independent chunks of TOKENS_PER_CHUNK
        chunks: list[Chunk] = []
        
        for i in range(0, total_tokens, TOKENS_PER_CHUNK):
            chunk_tokens = tokens[i : i + TOKENS_PER_CHUNK]
            chunk_text = encoder.decode(chunk_tokens)
            chunk_token_count = len(chunk_tokens)

            chunk = Chunk(
                full_text_id=uuid.uuid4(),  # Independent UUID
                name_space=file_name,
                text_to_embed=chunk_text,
                chunk_size=chunk_token_count,
                time_start=None,
                time_end=None,
                full_text=chunk_text,  # Same as text_to_embed for TXT
                embed_size=chunk_token_count,
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks from {file_name}")
        return chunks
    else:
        raise ValueError(f"Unsupported chunking strategy for TXT: {strategy.value}")


def preprocess_raw_transcripts(
    raw_transcripts: list[tuple[str, str]], 
    data_format: TypeOfFormat = TypeOfFormat.SRT,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
) -> list[Chunk]:
    """
    Processes raw transcripts by applying preprocessing steps, including:

    1. Chunking data into fixed-size token batches
    2. Adding metadata to each chunk

    :param raw_transcripts: List of (filename, content) tuples
    :param data_format: Format of the data (e.g., SRT or TXT)
    :param chunking_strategy: Chunking strategy to use (e.g., FIXED_SIZE)
    :return: List of Chunk objects
    """
    if data_format is None:
        raise RuntimeError(f"Unknown data_format")
    
    logger.info(
        f"Starting preprocessing of {len(raw_transcripts)} transcripts with "
        f"format: {data_format.name}, strategy: {chunking_strategy.value}"
    )
    cleaned_transcripts: list[Chunk] = []

    for i, raw_transcript in enumerate(raw_transcripts, 1):
        file_name = raw_transcript[0]
        logger.info(f"Processing transcript {i}/{len(raw_transcripts)}: {file_name}")

        if data_format.value == TypeOfFormat.SRT.value:
            chunks = chunk_srt(raw_transcript, strategy=chunking_strategy)
        elif data_format.value == TypeOfFormat.TXT.value:
            chunks = chunk_txt(raw_transcript, strategy=chunking_strategy)
        else:
            logger.error(f"Unsupported format: {data_format}")
            raise ValueError(f"Unsupported format: {data_format}")

        cleaned_transcripts.extend(chunks)
        logger.info(f"Completed processing {file_name}: {len(chunks)} chunks")

    logger.info(
        f"Preprocessing completed. Total chunks created: {len(cleaned_transcripts)}, "
        f"strategy: {chunking_strategy.value}"
    )
    return cleaned_transcripts


def translate_chunks(chunks: list[Chunk]) -> list[tuple[str, Chunk]]:
    logger.info(f"Starting translation of {len(chunks)} chunks")

    try:
        with open(
            "/Users/dothanbardichev/Desktop/RAV/RavProject/embed/data_embedder/data/translations/assigned.json",
            "r",
        ) as f:
            translations = json.load(f)
        logger.debug(f"Loaded {len(translations)} translation mappings")
    except Exception as e:
        logger.error(f"Failed to load translation file: {str(e)}")
        raise

    result: list[tuple[str, Chunk]] = []
    for i, chunk in enumerate(chunks, 1):
        if i % 100 == 0:
            logger.debug(f"Translated {i}/{len(chunks)} chunks")

        words = chunk.text_to_embed.lower().split()
        translated_words = [translations.get(word, word) for word in words]
        mapped_text = " ".join(translated_words)
        result.append((mapped_text, chunk))

    logger.info(f"Translation completed for {len(result)} chunks")
    return result
