"""
Strategy A: Fixed-Size Chunking

Sentence-aware token-based chunking with fixed sizes.
- Accumulates complete sentences until token threshold is reached
- Never cuts sentences in the middle
- Default: 500 tokens per chunk
- No overlap between chunks
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
    EMBEDDING_TEXT_SIZE,
    FULL_TEXT_SIZE,
    CHUNKS_PER_SEGMENT,
)

logger = logging.getLogger(__name__)

# Constants for fixed-size chunking
FIXED_CHUNK_SIZE_TOKENS = 250  # Target tokens per chunk (reduced from 500)
SPACY_MODEL = "en_core_web_sm"
TIKTOKEN_ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _get_spacy_model():
    """Get cached spaCy model for sentence splitting."""
    logger.info(f"[FIXED SIZE] Loading spaCy model: {SPACY_MODEL}")
    try:
        nlp = spacy.load(SPACY_MODEL)
        # Check if there's already a component that sets sentence boundaries
        if not nlp.has_pipe("sentencizer") and not nlp.has_pipe("parser"):
            # Add sentencizer if neither sentencizer nor parser exists
            nlp.add_pipe("sentencizer")
    except OSError:
        logger.warning(f"[FIXED SIZE] Model {SPACY_MODEL} not found, downloading...")
        from spacy.cli import download
        download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
        # Check if there's already a component that sets sentence boundaries
        if not nlp.has_pipe("sentencizer") and not nlp.has_pipe("parser"):
            # Add sentencizer if neither sentencizer nor parser exists
            nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=1)
def _get_tokenizer(encoding_name: str = TIKTOKEN_ENCODING):
    """Get cached tiktoken tokenizer."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"[FIXED SIZE] Failed to load encoding {encoding_name}: {e}, falling back to cl100k_base")
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text))


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy."""
    if not text or not text.strip():
        return []
    nlp = _get_spacy_model()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences if sentences else [text]


def build_chunks_sentence_based(
    text: str,
    name_space: str,
    chunk_size_tokens: int = FIXED_CHUNK_SIZE_TOKENS,
    start_time: str = None,
    end_time: str = None,
) -> List[Chunk]:
    """
    Build chunks by accumulating complete sentences until token threshold is reached.
    
    Logic:
    1. Split text into sentences using spaCy
    2. Accumulate sentences until adding the next one would exceed chunk_size_tokens
    3. Create chunk when threshold is reached
    4. Never cuts sentences in the middle
    
    Args:
        text: The text to chunk
        name_space: Namespace/filename
        chunk_size_tokens: Target tokens per chunk (default: 500)
        start_time: Start time for SRT files
        end_time: End time for SRT files
        
    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []
    
    logger.info(f"[FIXED SIZE] Building sentence-based chunks, target={chunk_size_tokens} tokens")
    
    sentences = _split_into_sentences(text)
    logger.info(f"[FIXED SIZE] Split into {len(sentences)} sentences")
    
    chunks = []
    current_sentences = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)
        
        # If adding this sentence would exceed threshold, create chunk
        if current_sentences and (current_tokens + sentence_tokens) > chunk_size_tokens:
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
            logger.debug(f"[FIXED SIZE] Created chunk with {current_tokens} tokens")
            
            # Start new chunk
            current_sentences = [sentence]
            current_tokens = sentence_tokens
        else:
            # Accumulate sentence
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
    
    # Handle remaining sentences
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
        logger.debug(f"[FIXED SIZE] Created final chunk with {current_tokens} tokens")
    
    logger.info(f"[FIXED SIZE] Created {len(chunks)} chunks")
    return chunks


def build_chunks(subs, name_space, embed_word_limit=EMBEDDING_TEXT_SIZE) -> List[Chunk]:
    """
    Builds a list of Chunk objects from subtitle segments, where each chunk contains
    a segment for embedding and the full text for reference.

    All chunks created from the same text segment share a single UUID (full_text_id)
    to enable deduplication during retrieval.

    If the segment exceeds FULL_TEXT_SIZE (200 words), it is divided evenly into
    CHUNKS_PER_SEGMENT (4) chunks. Otherwise, it uses the embed_word_limit (50 words)
    to create chunks.

    :param subs: List of subtitle objects (pysrt.SubRipItem).
    :param name_space: The name of the file or namespace for this chunk.
    :param embed_word_limit: Number of words per embedding segment (default: 50 from EMBEDDING_TEXT_SIZE).
    :return: List of Chunk objects with shared full_text_id.
    """
    if not subs:
        return []

    start_time = subs[0].start
    end_time = subs[-1].end

    full_text = " ".join(s.text.replace("\n", " ") for s in subs)

    # Get all words from all subtitles
    all_words = []
    for sub in subs:
        clean_text = sub.text.replace("\n", " ")
        words = clean_text.split()
        all_words.extend(words)

    # Generate ONE shared UUID for all chunks from this text segment
    shared_text_id = uuid.uuid4()

    chunks = []
    total_words = len(all_words)

    # If segment exceeds FULL_TEXT_SIZE, divide evenly into CHUNKS_PER_SEGMENT chunks
    if total_words > FULL_TEXT_SIZE:
        words_per_chunk = total_words // CHUNKS_PER_SEGMENT
        remainder = total_words % CHUNKS_PER_SEGMENT
        
        start_idx = 0
        for i in range(CHUNKS_PER_SEGMENT):
            chunk_size = words_per_chunk + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            
            text_to_embed = " ".join(all_words[start_idx:end_idx])
            
            chunk = Chunk(
                full_text_id=shared_text_id,
                time_start=str(start_time),
                time_end=str(end_time),
                full_text=full_text,
                text_to_embed=text_to_embed,
                chunk_size=total_words,
                embed_size=chunk_size,
                name_space=name_space,
            )
            chunks.append(chunk)
            start_idx = end_idx
    else:
        # Original logic for segments <= FULL_TEXT_SIZE
        for i in range(0, total_words, embed_word_limit):
            end_idx = min(i + embed_word_limit, total_words)
            text_to_embed = " ".join(all_words[i:end_idx])
            embed_size = end_idx - i

            chunk = Chunk(
                full_text_id=shared_text_id,
                time_start=str(start_time),
                time_end=str(end_time),
                full_text=full_text,
                text_to_embed=text_to_embed,
                chunk_size=total_words,
                embed_size=embed_size,
                name_space=name_space,
            )
            chunks.append(chunk)

    return chunks


def chunk_srt_fixed(content: Tuple[str, str]) -> List[Chunk]:
    """
    Fixed-size sentence-based chunking for SRT files.
    
    Uses spaCy for sentence splitting and accumulates sentences until token threshold.
    Preserves timing information from SRT files.
    """
    file_name, text = content
    logger.info(f"[FIXED SIZE] Chunking SRT file: {file_name}")
    subs = pysrt.from_string(text)
    logger.info(f"[FIXED SIZE] Parsed {len(subs)} subtitle segments from {file_name}")

    # Combine all subtitle text for sentence-based chunking
    full_text = " ".join(sub.text.replace("\n", " ") for sub in subs)
    start_time = str(subs[0].start) if subs else None
    end_time = str(subs[-1].end) if subs else None
    
    chunks = build_chunks_sentence_based(
        text=full_text,
        name_space=file_name,
        chunk_size_tokens=FIXED_CHUNK_SIZE_TOKENS,
        start_time=start_time,
        end_time=end_time,
    )

    logger.info(f"[FIXED SIZE] Created {len(chunks)} chunks from {file_name}")
    return chunks


def chunk_txt_fixed(content: Tuple[str, str]) -> List[Chunk]:
    """
    Fixed-size sentence-based chunking for TXT files.
    
    Uses spaCy for sentence splitting and accumulates sentences until token threshold.
    """
    file_name, text = content
    logger.info(f"[FIXED SIZE] Chunking TXT file: {file_name}")
    
    chunks = build_chunks_sentence_based(
        text=text,
        name_space=file_name,
        chunk_size_tokens=FIXED_CHUNK_SIZE_TOKENS,
        start_time=None,
        end_time=None,
    )
    
    logger.info(f"[FIXED SIZE] Created {len(chunks)} chunks from {file_name}")
    return chunks


def preprocess_raw_transcripts_fixed(
    raw_transcripts: List[Tuple[str, str]], 
    data_format: TypeOfFormat = TypeOfFormat.SRT
) -> List[Chunk]:
    """
    Fixed-size chunking preprocessing pipeline.
    
    Processes raw transcripts by applying fixed-size word-based chunking.
    """
    if data_format is None:
        raise RuntimeError(f"Unknown data_format")
    logger.info(
        f"[FIXED SIZE] Starting preprocessing of {len(raw_transcripts)} transcripts "
        f"with format: {data_format.name}"
    )
    cleaned_transcripts: List[Chunk] = []

    for i, raw_transcript in enumerate(raw_transcripts, 1):
        file_name = raw_transcript[0]
        logger.info(f"[FIXED SIZE] Processing transcript {i}/{len(raw_transcripts)}: {file_name}")

        if data_format.value == TypeOfFormat.SRT.value:
            chunks = chunk_srt_fixed(raw_transcript)
        elif data_format.value == TypeOfFormat.TXT.value:
            chunks = chunk_txt_fixed(raw_transcript)
        else:
            logger.error(f"[FIXED SIZE] Unsupported format: {data_format}")
            raise ValueError(f"Unsupported format: {data_format}")

        cleaned_transcripts.extend(chunks)
        logger.info(f"[FIXED SIZE] Completed processing {file_name}: {len(chunks)} chunks")

    logger.info(
        f"[FIXED SIZE] Preprocessing completed. Total chunks created: {len(cleaned_transcripts)}"
    )
    return cleaned_transcripts

