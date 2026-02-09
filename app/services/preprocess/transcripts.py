import hashlib
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
from rag.app.services.preprocess.strategies.fixed_size import (
    build_chunks_fixed_size_srt,
    build_chunks_fixed_size_txt,
)
from rag.app.services.preprocess.strategies.divided import (
    build_chunks_divided_srt,
    build_chunks_divided_txt,
    _subdivide_entries,
)
from rag.app.services.preprocess.strategies.sentence_aware_regex import (
    build_chunks_sentence_fixed_srt_regex,
    build_chunks_sentence_divided_srt_regex,
    build_chunks_sentence_fixed_txt_regex,
    build_chunks_sentence_divided_txt_regex,
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


def _compute_text_hash(text: str) -> str:
    """
    Compute SHA-256 hash of the text.
    
    Args:
        text: The text to hash
    
    Returns:
        Hexadecimal string representation of the SHA-256 hash
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


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
    
    Wrapper for the modular FIXED_SIZE SRT strategy implementation.

    :param subs: List of subtitle objects (pysrt.SubRipItem).
    :param name_space: The name of the file or namespace for this chunk.
    :param tokens_per_chunk: Target number of tokens per chunk (default: 200 from TOKENS_PER_CHUNK).
    :return: List of Chunk objects, each with independent UUID.
    """
    if not subs:
        return []

    sub_stream = _flatten_subs(subs)  # List of subtitle entries with tokens
    return build_chunks_fixed_size_srt(
        sub_entries=sub_stream,
        name_space=name_space,
        build_chunk_lines_fn=_build_chunk_lines,
        create_chunk_fn=_create_chunk_from_entries,
        tokens_per_chunk=tokens_per_chunk,
    )


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
    
    # Compute SHA-256 hash of the text to embed
    text_hash = _compute_text_hash(text_to_embed)
    
    return Chunk(
        full_text_id=uuid.uuid4(),  # Each chunk gets its own UUID
        time_start=time_start,
        time_end=time_end,
        full_text=chunk_lines,  # List of (text, (start, end)) tuples as metadata
        text_to_embed=text_to_embed,  # Actual concatenated text for embedding
        chunk_size=embed_tokens,  # Tokens in this chunk
        embed_size=embed_tokens,  # Same as chunk_size for fixed-size chunking
        name_space=name_space,
        text_hash=text_hash,
    )


def build_chunks_divided(subs, name_space, chunk_size=DIVIDED_CHUNK_SIZE, subdivisions=DIVIDED_SUBDIVISIONS) -> list[Chunk]:
    """
    DIVIDED chunking strategy: accumulate subtitles to chunk_size, then subdivide.
    
    Wrapper for the modular DIVIDED SRT strategy implementation.

    :param subs: List of subtitle objects (pysrt.SubRipItem).
    :param name_space: The name of the file or namespace for this chunk.
    :param chunk_size: Size of main chunk before division (default: 800 from DIVIDED_CHUNK_SIZE).
    :param subdivisions: Number of subdivisions per main chunk (default: 4 from DIVIDED_SUBDIVISIONS).
    :return: List of Chunk objects with shared full_text_id per main chunk.
    """
    if not subs:
        return []

    sub_stream = _flatten_subs(subs)  # List of subtitle entries with tokens
    return build_chunks_divided_srt(
        sub_entries=sub_stream,
        name_space=name_space,
        build_chunk_lines_fn=_build_chunk_lines,
        compute_text_hash_fn=_compute_text_hash,
        chunk_size=chunk_size,
        subdivisions=subdivisions,
    )


def build_chunks_sentence_fixed_regex(subs, name_space, tokens_per_chunk=TOKENS_PER_CHUNK) -> list[Chunk]:
    """
    SENTENCE_FIXED_REGEX chunking strategy: sentence-aware chunking with regex.
    
    Wrapper for the modular sentence-aware SRT strategy implementation.

    :param subs: List of subtitle objects (pysrt.SubRipItem).
    :param name_space: The name of the file or namespace for this chunk.
    :param tokens_per_chunk: Target number of tokens per chunk (default: 200 from TOKENS_PER_CHUNK).
    :return: List of Chunk objects with sentence-aware boundaries.
    """
    if not subs:
        return []

    sub_stream = _flatten_subs(subs)  # List of subtitle entries with tokens
    return build_chunks_sentence_fixed_srt_regex(
        sub_entries=sub_stream,
        name_space=name_space,
        build_chunk_lines_fn=_build_chunk_lines,
        create_chunk_fn=_create_chunk_from_entries,
        tokens_per_chunk=tokens_per_chunk,
    )


def build_chunks_sentence_divided_regex(subs, name_space, chunk_size=DIVIDED_CHUNK_SIZE, subdivisions=DIVIDED_SUBDIVISIONS) -> list[Chunk]:
    """
    SENTENCE_DIVIDED_REGEX chunking strategy: sentence-aware divided chunking.
    
    Wrapper for the modular sentence-aware divided SRT strategy implementation.

    :param subs: List of subtitle objects (pysrt.SubRipItem).
    :param name_space: The name of the file or namespace for this chunk.
    :param chunk_size: Size of main chunk before division (default: 800 from DIVIDED_CHUNK_SIZE).
    :param subdivisions: Number of subdivisions per main chunk (default: 4 from DIVIDED_SUBDIVISIONS).
    :return: List of Chunk objects with sentence-aware boundaries and subdivisions.
    """
    if not subs:
        return []

    sub_stream = _flatten_subs(subs)  # List of subtitle entries with tokens
    return build_chunks_sentence_divided_srt_regex(
        sub_entries=sub_stream,
        name_space=name_space,
        build_chunk_lines_fn=_build_chunk_lines,
        compute_text_hash_fn=_compute_text_hash,
        subdivide_entries_fn=_subdivide_entries,
        chunk_size=chunk_size,
        subdivisions=subdivisions,
    )


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
    elif strategy == ChunkingStrategy.SENTENCE_FIXED_REGEX:
        chunks = build_chunks_sentence_fixed_regex(subs, file_name)
    elif strategy == ChunkingStrategy.SENTENCE_DIVIDED_REGEX:
        chunks = build_chunks_sentence_divided_regex(subs, file_name)
    else:
        raise ValueError(f"Unsupported chunking strategy for SRT: {strategy.value}")
    
    logger.debug(f"Created {len(chunks)} chunks from {file_name}")
    return chunks


def build_chunks_divided_txt_wrapper(text: str, name_space: str, chunk_size=DIVIDED_CHUNK_SIZE, subdivisions=DIVIDED_SUBDIVISIONS) -> list[Chunk]:
    """
    DIVIDED strategy for plain text files.
    
    Wrapper for the modular DIVIDED TXT strategy implementation.
    """
    encoder = _get_encoder()
    tokens = encoder.encode(text)
    
    return build_chunks_divided_txt(
        tokens=tokens,
        encoder=encoder,
        name_space=name_space,
        compute_text_hash_fn=_compute_text_hash,
        chunk_size=chunk_size,
        subdivisions=subdivisions,
    )


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
        return build_chunks_divided_txt_wrapper(text, file_name)
    elif strategy == ChunkingStrategy.FIXED_SIZE:
        return build_chunks_fixed_size_txt(
            tokens=tokens,
            encoder=encoder,
            name_space=file_name,
            compute_text_hash_fn=_compute_text_hash,
            tokens_per_chunk=TOKENS_PER_CHUNK,
        )
    elif strategy == ChunkingStrategy.SENTENCE_FIXED_REGEX:
        return build_chunks_sentence_fixed_txt_regex(
            tokens=tokens,
            encoder=encoder,
            name_space=file_name,
            compute_text_hash_fn=_compute_text_hash,
            tokens_per_chunk=TOKENS_PER_CHUNK,
        )
    elif strategy == ChunkingStrategy.SENTENCE_DIVIDED_REGEX:
        return build_chunks_sentence_divided_txt_regex(
            tokens=tokens,
            encoder=encoder,
            name_space=file_name,
            compute_text_hash_fn=_compute_text_hash,
            chunk_size=DIVIDED_CHUNK_SIZE,
            subdivisions=DIVIDED_SUBDIVISIONS,
        )
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
