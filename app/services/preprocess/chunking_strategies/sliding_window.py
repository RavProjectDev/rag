"""
Strategy C: Sliding Window Chunking

Sentence-aware sliding window approach with overlap.
- Splits text into sentences using spaCy
- Accumulates complete sentences until token threshold
- Overlaps by including sentences from previous chunk
- Never cuts sentences in the middle
- Default: 500 tokens per chunk, 100 token overlap
"""
import uuid
import pysrt
import spacy
import tiktoken
import logging
from typing import List, Tuple
from functools import lru_cache
from rag.app.schemas.data import Chunk, TypeOfFormat
from rag.app.services.preprocess.constants import (
    SLIDING_WINDOW_SIZE,
    SLIDING_WINDOW_OVERLAP,
    SLIDING_WINDOW_ENCODING,
)

logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def _get_spacy_model():
    """Get cached spaCy model for sentence splitting."""
    logger.info(f"[SLIDING WINDOW] Loading spaCy model: {SPACY_MODEL}")
    try:
        nlp = spacy.load(SPACY_MODEL)
        # Check if there's already a component that sets sentence boundaries
        if not nlp.has_pipe("sentencizer") and not nlp.has_pipe("parser"):
            # Add sentencizer if neither sentencizer nor parser exists
            nlp.add_pipe("sentencizer")
    except OSError:
        logger.warning(f"[SLIDING WINDOW] Model {SPACY_MODEL} not found, downloading...")
        from spacy.cli import download
        download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
        # Check if there's already a component that sets sentence boundaries
        if not nlp.has_pipe("sentencizer") and not nlp.has_pipe("parser"):
            # Add sentencizer if neither sentencizer nor parser exists
            nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=1)
def get_tokenizer(encoding_name: str = SLIDING_WINDOW_ENCODING):
    """Get tiktoken tokenizer for accurate token counting."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(
            f"[SLIDING WINDOW] Failed to load encoding {encoding_name}: {e}, "
            f"falling back to cl100k_base"
        )
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoding_name: str = SLIDING_WINDOW_ENCODING) -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    tokenizer = get_tokenizer(encoding_name)
    return len(tokenizer.encode(text))


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy."""
    if not text or not text.strip():
        return []
    nlp = _get_spacy_model()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences if sentences else [text]


def build_sliding_window_chunks(
    text: str,
    name_space: str,
    window_size: int = SLIDING_WINDOW_SIZE,
    overlap: int = SLIDING_WINDOW_OVERLAP,
    start_time: str = None,
    end_time: str = None,
    encoding_name: str = SLIDING_WINDOW_ENCODING,
) -> List[Chunk]:
    """
    Build chunks using sentence-aware sliding window with overlap.
    
    Logic:
    1. Split text into sentences using spaCy
    2. Accumulate complete sentences until window_size tokens is reached
    3. For next chunk, include sentences worth ~overlap tokens from previous chunk
    4. Never cuts sentences in the middle
    
    Args:
        text: The text to chunk
        name_space: Namespace/filename
        window_size: Target tokens per chunk (default: 500)
        overlap: Target tokens to overlap between chunks (default: 100)
        start_time: Start time for SRT files
        end_time: End time for SRT files
        encoding_name: Tiktoken encoding to use
        
    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []
    
    logger.info(
        f"[SLIDING WINDOW] Starting sentence-based sliding window chunking, "
        f"window_size={window_size}, overlap={overlap}"
    )
    
    # Split into sentences
    sentences = _split_into_sentences(text)
    logger.info(f"[SLIDING WINDOW] Split into {len(sentences)} sentences")
    
    if not sentences:
        return []
    
    # Calculate token count for each sentence
    sentence_tokens = [_count_tokens(sent, encoding_name) for sent in sentences]
    
    chunks = []
    i = 0  # Current sentence index
    
    while i < len(sentences):
        # Build current chunk by accumulating sentences
        current_sentences = []
        current_tokens = 0
        
        # Accumulate sentences until we reach window_size
        while i < len(sentences) and current_tokens + sentence_tokens[i] <= window_size:
            current_sentences.append(sentences[i])
            current_tokens += sentence_tokens[i]
            i += 1
        
        # If we didn't add any sentences (single sentence > window_size), add it anyway
        if not current_sentences and i < len(sentences):
            current_sentences.append(sentences[i])
            current_tokens = sentence_tokens[i]
            logger.warning(
                f"[SLIDING WINDOW] Single sentence exceeds window_size: "
                f"{current_tokens} > {window_size} tokens"
            )
            i += 1
        
        # Create chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = Chunk(
                full_text_id=uuid.uuid4(),
                time_start=start_time,
                time_end=end_time,
                full_text=chunk_text,
                text_to_embed=chunk_text,
                chunk_size=current_tokens,
                embed_size=current_tokens,
                name_space=name_space,
            )
            chunks.append(chunk)
            logger.debug(f"[SLIDING WINDOW] Created chunk with {current_tokens} tokens")
        
        # Calculate overlap for next chunk
        # Move back to include sentences worth ~overlap tokens
        if i < len(sentences):
            overlap_tokens = 0
            sentences_to_overlap = 0
            
            # Count back from current position to find overlap
            j = i - 1
            while j >= 0 and overlap_tokens < overlap:
                overlap_tokens += sentence_tokens[j]
                sentences_to_overlap += 1
                j -= 1
            
            # Move back to start of overlap
            i = max(0, i - sentences_to_overlap)
            logger.debug(f"[SLIDING WINDOW] Overlapping {sentences_to_overlap} sentences (~{overlap_tokens} tokens)")
    
    logger.info(f"[SLIDING WINDOW] Created {len(chunks)} chunks")
    return chunks


def chunk_srt_sliding_window(
    content: Tuple[str, str],
    window_size: int = SLIDING_WINDOW_SIZE,
    overlap: int = SLIDING_WINDOW_OVERLAP,
    encoding_name: str = SLIDING_WINDOW_ENCODING,
) -> List[Chunk]:
    """
    Sentence-aware sliding window chunking for SRT files.
    
    Accumulates complete sentences until window_size, with overlap between chunks.
    Preserves timing information from SRT files.
    """
    file_name, text = content
    logger.info(f"[SLIDING WINDOW] Chunking SRT file: {file_name}")
    subs = pysrt.from_string(text)
    logger.info(f"[SLIDING WINDOW] Parsed {len(subs)} subtitle segments from {file_name}")
    
    # Combine all subtitle text for sentence-based chunking
    full_text = " ".join(sub.text.replace("\n", " ") for sub in subs)
    start_time = str(subs[0].start) if subs else None
    end_time = str(subs[-1].end) if subs else None
    
    chunks = build_sliding_window_chunks(
        text=full_text,
        name_space=file_name,
        window_size=window_size,
        overlap=overlap,
        start_time=start_time,
        end_time=end_time,
        encoding_name=encoding_name,
    )
    
    logger.info(f"[SLIDING WINDOW] Created {len(chunks)} chunks from {file_name}")
    return chunks


def chunk_txt_sliding_window(
    content: Tuple[str, str],
    window_size: int = SLIDING_WINDOW_SIZE,
    overlap: int = SLIDING_WINDOW_OVERLAP,
    encoding_name: str = SLIDING_WINDOW_ENCODING,
) -> List[Chunk]:
    """
    Sentence-aware sliding window chunking for TXT files.
    
    Accumulates complete sentences until window_size, with overlap between chunks.
    """
    file_name, text = content
    logger.info(f"[SLIDING WINDOW] Chunking TXT file: {file_name}")
    
    chunks = build_sliding_window_chunks(
        text=text,
        name_space=file_name,
        window_size=window_size,
        overlap=overlap,
        start_time=None,
        end_time=None,
        encoding_name=encoding_name,
    )
    
    logger.info(f"[SLIDING WINDOW] Created {len(chunks)} chunks from {file_name}")
    return chunks


def preprocess_raw_transcripts_sliding_window(
    raw_transcripts: List[Tuple[str, str]],
    data_format: TypeOfFormat,
    window_size: int = SLIDING_WINDOW_SIZE,
    overlap: int = SLIDING_WINDOW_OVERLAP,
    encoding_name: str = SLIDING_WINDOW_ENCODING,
) -> List[Chunk]:
    """
    Sliding window preprocessing pipeline.
    
    Uses fixed-size sliding windows with overlap - the industry baseline approach.
    """
    if data_format is None:
        raise RuntimeError(f"Unknown data_format")
    logger.info(
        f"[SLIDING WINDOW] Starting sliding window preprocessing of {len(raw_transcripts)} transcripts "
        f"with format: {data_format.name}, window_size: {window_size}, overlap: {overlap}"
    )
    cleaned_transcripts: List[Chunk] = []
    
    for i, raw_transcript in enumerate(raw_transcripts, 1):
        file_name = raw_transcript[0]
        logger.info(f"[SLIDING WINDOW] Processing transcript {i}/{len(raw_transcripts)}: {file_name}")
        
        if data_format.value == TypeOfFormat.SRT.value:
            chunks = chunk_srt_sliding_window(
                raw_transcript,
                window_size,
                overlap,
                encoding_name,
            )
        elif data_format.value == TypeOfFormat.TXT.value:
            chunks = chunk_txt_sliding_window(
                raw_transcript,
                window_size,
                overlap,
                encoding_name,
            )
        else:
            logger.error(f"[SLIDING WINDOW] Unsupported format: {data_format}")
            raise ValueError(f"Unsupported format: {data_format}")
        
        cleaned_transcripts.extend(chunks)
        logger.info(f"[SLIDING WINDOW] Completed processing {file_name}: {len(chunks)} chunks")
    
    logger.info(
        f"[SLIDING WINDOW] Sliding window preprocessing completed. "
        f"Total chunks created: {len(cleaned_transcripts)}"
    )
    return cleaned_transcripts

