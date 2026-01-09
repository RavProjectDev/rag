"""
Strategy B: Semantic Chunking

Uses semantic similarity between sentences to determine chunk boundaries.
- Splits text into sentences using spaCy
- Generates embeddings for each sentence using all-MiniLM-L6-v2 (BERT)
- Computes cosine similarity between consecutive sentences
- Creates chunks when similarity drops below threshold
- Variable chunk sizes (50-1000 tokens)

Note: This strategy uses BERT (all-MiniLM-L6-v2) internally for similarity calculations,
independent of the embedding configuration used for final vector storage.
"""
import uuid
import pysrt
import numpy as np
import logging
import spacy
import tiktoken
from typing import List, Tuple
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from rag.app.schemas.data import Chunk, TypeOfFormat, EmbeddingConfiguration
from rag.app.services.preprocess.constants import (
    SEMANTIC_SIMILARITY_THRESHOLD,
    MIN_CHUNK_TOKENS,
    MAX_CHUNK_TOKENS,
    SEMANTIC_ENCODING,
    FULL_TEXT_SIZE,
)

logger = logging.getLogger(__name__)

# Internal constant for semantic chunking - not exposed in EmbeddingConfiguration
SEMANTIC_CHUNKING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def _get_spacy_model():
    """
    Get cached spaCy model for sentence splitting.
    Uses en_core_web_sm for efficient sentence boundary detection.
    """
    logger.info(f"[SEMANTIC] Loading spaCy model: {SPACY_MODEL}")
    try:
        nlp = spacy.load(SPACY_MODEL)
        # Check if there's already a component that sets sentence boundaries
        if not nlp.has_pipe("sentencizer") and not nlp.has_pipe("parser"):
            # Add sentencizer if neither sentencizer nor parser exists
            nlp.add_pipe("sentencizer")
    except OSError:
        logger.warning(f"[SEMANTIC] Model {SPACY_MODEL} not found, downloading...")
        from spacy.cli import download
        download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
        # Check if there's already a component that sets sentence boundaries
        if not nlp.has_pipe("sentencizer") and not nlp.has_pipe("parser"):
            # Add sentencizer if neither sentencizer nor parser exists
            nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=1)
def _get_semantic_chunking_model() -> SentenceTransformer:
    """
    Get cached SentenceTransformer model for semantic chunking.
    This is separate from the main embedding service and only used internally.
    """
    logger.info(f"[SEMANTIC] Loading model for chunking: {SEMANTIC_CHUNKING_MODEL}")
    return SentenceTransformer(SEMANTIC_CHUNKING_MODEL)


@lru_cache(maxsize=1)
def _get_tokenizer(encoding_name: str = SEMANTIC_ENCODING):
    """
    Get cached tiktoken tokenizer for accurate token counting.
    Uses cl100k_base encoding (GPT-4/GPT-3.5) by default.
    """
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(
            f"[SEMANTIC] Failed to load encoding {encoding_name}: {e}, "
            f"falling back to cl100k_base"
        )
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoding_name: str = SEMANTIC_ENCODING) -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text: The text to count tokens for
        encoding_name: The tiktoken encoding to use
        
    Returns:
        Number of tokens in the text
    """
    if not text:
        return 0
    tokenizer = _get_tokenizer(encoding_name)
    return len(tokenizer.encode(text))


def _generate_sentence_embedding_for_chunking(text: str) -> List[float]:
    """
    Generate embeddings for semantic chunking using BERT.
    This is an internal function, not exposed through the main embedding API.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        List of float values representing the embedding (384 dimensions)
    """
    try:
        model = _get_semantic_chunking_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"[SEMANTIC ERROR] Failed to generate chunking embedding: {e}")
        # Return zero vector as fallback
        return [0.0] * 384


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using spaCy's sentence segmentation.
    This provides more accurate sentence boundary detection compared to regex,
    especially for complex cases like abbreviations, decimals, and quotes.
    """
    if not text or not text.strip():
        return []
    
    nlp = _get_spacy_model()
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract sentences and clean them
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    return sentences if sentences else [text]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


async def build_semantic_chunks(
    text: str,
    name_space: str,
    embedding_configuration: EmbeddingConfiguration,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    start_time: str = None,
    end_time: str = None,
) -> List[Chunk]:
    """
    Build chunks using semantic similarity between sentences.
    
    Logic:
    1. Split text into sentences
    2. Generate embeddings for each sentence using BERT (all-MiniLM-L6-v2)
    3. Compute cosine similarity between consecutive sentences
    4. Make a cut when similarity drops below threshold
    5. Create variable-sized chunks based on "idea boundaries"
    
    Note: The embedding_configuration parameter is kept for API compatibility
    but is NOT used for chunking. Semantic chunking always uses BERT (all-MiniLM-L6-v2)
    internally for similarity calculations. The embedding_configuration is only used
    later for generating final storage embeddings.
    """
    if not text or not text.strip():
        return []
    
    logger.info(
        f"[SEMANTIC] Starting semantic chunking for '{name_space}', "
        f"text_length={len(text)} chars, threshold={similarity_threshold}, "
        f"using {SEMANTIC_CHUNKING_MODEL} for similarity"
    )
    
    sentences = split_into_sentences(text)
    logger.info(f"[SEMANTIC] Split into {len(sentences)} sentences")
    
    if len(sentences) <= 1:
        num_tokens = _count_tokens(text)
        shared_text_id = uuid.uuid4()
        return [
            Chunk(
                full_text_id=shared_text_id,
                time_start=start_time,
                time_end=end_time,
                full_text=text,
                text_to_embed=text,
                chunk_size=num_tokens,
                embed_size=num_tokens,
                name_space=name_space,
            )
        ]
    
    # Generate embeddings for all sentences using internal BERT function
    logger.info(f"[SEMANTIC] Generating embeddings for {len(sentences)} sentences using {SEMANTIC_CHUNKING_MODEL}...")
    sentence_embeddings = []
    
    for i, sentence in enumerate(sentences):
        # Use internal BERT function, not the main embedding service
        embedding_vector = _generate_sentence_embedding_for_chunking(sentence)
        sentence_embeddings.append(embedding_vector)
        
        if (i + 1) % 10 == 0:
            logger.debug(f"[SEMANTIC] Generated {i + 1}/{len(sentences)} embeddings")
    
    logger.info(f"[SEMANTIC] Generated {len(sentence_embeddings)} embeddings")
    
    # Compute similarities and find cut points
    cut_points = [0]
    similarities = []
    
    for i in range(len(sentences) - 1):
        similarity = cosine_similarity(
            sentence_embeddings[i],
            sentence_embeddings[i + 1]
        )
        similarities.append(similarity)
        
        if similarity < similarity_threshold:
            cut_points.append(i + 1)
            logger.debug(
                f"[SEMANTIC] Cut point at sentence {i + 1}, "
                f"similarity={similarity:.3f} < {similarity_threshold}"
            )
    
    cut_points.append(len(sentences))
    
    avg_similarity = np.mean(similarities) if similarities else 0
    logger.info(
        f"[SEMANTIC] Found {len(cut_points) - 1} semantic boundaries "
        f"(avg similarity: {avg_similarity:.3f})"
    )
    
    # Build chunks from cut points
    chunks = []
    shared_text_id = uuid.uuid4()
    
    for i in range(len(cut_points) - 1):
        start_idx = cut_points[i]
        end_idx = cut_points[i + 1]
        
        chunk_sentences = sentences[start_idx:end_idx]
        chunk_text = " ".join(chunk_sentences)
        num_tokens = _count_tokens(chunk_text)
        
        # Skip chunks that are too small
        if num_tokens < MIN_CHUNK_TOKENS and len(cut_points) > 2:
            logger.debug(
                f"[SEMANTIC] Skipping tiny chunk ({num_tokens} tokens) "
                f"at sentences {start_idx}-{end_idx}"
            )
            if chunks:
                prev_chunk = chunks[-1]
                merged_text = prev_chunk.text_to_embed + " " + chunk_text
                merged_tokens = _count_tokens(merged_text)
                
                chunks[-1] = Chunk(
                    full_text_id=shared_text_id,
                    time_start=start_time,
                    time_end=end_time,
                    full_text=text,
                    text_to_embed=merged_text,
                    chunk_size=merged_tokens,
                    embed_size=merged_tokens,
                    name_space=name_space,
                )
                continue
        
        # Enforce maximum chunk size
        if num_tokens > MAX_CHUNK_TOKENS:
            logger.warning(
                f"[SEMANTIC] Chunk exceeds MAX_CHUNK_TOKENS ({num_tokens} > {MAX_CHUNK_TOKENS})"
            )
            # Split into smaller chunks by dividing sentences
            sentences_per_subchunk = max(1, len(chunk_sentences) // ((num_tokens // MAX_CHUNK_TOKENS) + 1))
            for j in range(0, len(chunk_sentences), sentences_per_subchunk):
                sub_sentences = chunk_sentences[j:j + sentences_per_subchunk]
                sub_text = " ".join(sub_sentences)
                sub_tokens = _count_tokens(sub_text)
                
                chunk = Chunk(
                    full_text_id=shared_text_id,
                    time_start=start_time,
                    time_end=end_time,
                    full_text=text,
                    text_to_embed=sub_text,
                    chunk_size=sub_tokens,
                    embed_size=sub_tokens,
                    name_space=name_space,
                )
                chunks.append(chunk)
        else:
            chunk = Chunk(
                full_text_id=shared_text_id,
                time_start=start_time,
                time_end=end_time,
                full_text=text,
                text_to_embed=chunk_text,
                chunk_size=num_tokens,
                embed_size=num_tokens,
                name_space=name_space,
            )
            chunks.append(chunk)
    
    logger.info(
        f"[SEMANTIC] Created {len(chunks)} semantic chunks from {len(sentences)} sentences"
    )
    return chunks


async def chunk_srt_semantic(
    content: Tuple[str, str],
    embedding_configuration: EmbeddingConfiguration,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> List[Chunk]:
    """
    Semantic chunking for SRT files.
    
    Note: embedding_configuration is kept for API compatibility but not used.
    Semantic chunking always uses BERT (all-MiniLM-L6-v2) internally.
    """
    file_name, text = content
    logger.info(f"[SEMANTIC] Chunking SRT file: {file_name}")
    subs = pysrt.from_string(text)
    logger.info(f"[SEMANTIC] Parsed {len(subs)} subtitle segments from {file_name}")
    
    chunks = []
    current_chunk = []
    
    for sub in subs:
        current_chunk.append(sub)
        full_text = " ".join(s.text.replace("\n", " ") for s in current_chunk)
        words = full_text.split()
        
        if len(words) >= FULL_TEXT_SIZE or sub == subs[-1]:
            start_time = current_chunk[0].start
            end_time = current_chunk[-1].end
            segment_text = " ".join(s.text.replace("\n", " ") for s in current_chunk)
            
            semantic_chunks = await build_semantic_chunks(
                text=segment_text,
                name_space=file_name,
                embedding_configuration=embedding_configuration,
                similarity_threshold=similarity_threshold,
                start_time=str(start_time),
                end_time=str(end_time),
            )
            
            chunks.extend(semantic_chunks)
            current_chunk = []
    
    logger.info(f"[SEMANTIC] Created {len(chunks)} semantic chunks from {file_name}")
    return chunks


async def chunk_txt_semantic(
    content: Tuple[str, str],
    embedding_configuration: EmbeddingConfiguration,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> List[Chunk]:
    """
    Semantic chunking for TXT files.
    
    Note: embedding_configuration is kept for API compatibility but not used.
    Semantic chunking always uses BERT (all-MiniLM-L6-v2) internally.
    """
    file_name, text = content
    logger.info(f"[SEMANTIC] Chunking TXT file: {file_name}")
    
    return await build_semantic_chunks(
        text=text,
        name_space=file_name,
        embedding_configuration=embedding_configuration,
        similarity_threshold=similarity_threshold,
        start_time=None,
        end_time=None,
    )


async def preprocess_raw_transcripts_semantic(
    raw_transcripts: List[Tuple[str, str]],
    data_format: TypeOfFormat,
    embedding_configuration: EmbeddingConfiguration,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> List[Chunk]:
    """
    Semantic chunking preprocessing pipeline.
    
    Uses semantic similarity between sentences to determine chunk boundaries.
    
    Note: embedding_configuration is kept for API compatibility but not used for chunking.
    Semantic chunking always uses BERT (all-MiniLM-L6-v2) internally for similarity
    calculations. The embedding_configuration is only used later when generating final
    embeddings for storage.
    """
    if data_format is None:
        raise RuntimeError(f"Unknown data_format")
    logger.info(
        f"[SEMANTIC] Starting semantic preprocessing of {len(raw_transcripts)} transcripts "
        f"with format: {data_format.name}, threshold: {similarity_threshold}"
    )
    cleaned_transcripts: List[Chunk] = []
    
    for i, raw_transcript in enumerate(raw_transcripts, 1):
        file_name = raw_transcript[0]
        logger.info(f"[SEMANTIC] Processing transcript {i}/{len(raw_transcripts)}: {file_name}")
        
        if data_format.value == TypeOfFormat.SRT.value:
            chunks = await chunk_srt_semantic(
                raw_transcript,
                embedding_configuration,
                similarity_threshold,
            )
        elif data_format.value == TypeOfFormat.TXT.value:
            chunks = await chunk_txt_semantic(
                raw_transcript,
                embedding_configuration,
                similarity_threshold,
            )
        else:
            logger.error(f"[SEMANTIC] Unsupported format: {data_format}")
            raise ValueError(f"Unsupported format: {data_format}")
        
        cleaned_transcripts.extend(chunks)
        logger.info(f"[SEMANTIC] Completed processing {file_name}: {len(chunks)} chunks")
    
    logger.info(
        f"[SEMANTIC] Semantic preprocessing completed. Total chunks created: {len(cleaned_transcripts)}"
    )
    return cleaned_transcripts

