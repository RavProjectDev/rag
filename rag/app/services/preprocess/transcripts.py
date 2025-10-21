import json
import pysrt
from rag.app.schemas.data import Chunk, TypeOfFormat
import logging
from rag.app.services.preprocess.constants import EMBEDDING_TEXT_SIZE, FULL_TEXT_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_chunks(subs, name_space, embed_word_limit=EMBEDDING_TEXT_SIZE) -> list[Chunk]:
    """
    Builds a list of Chunk objects from subtitle segments, where each chunk contains
    a 40-word segment for embedding and the full text for reference.

    :param subs: List of subtitle objects (pysrt.SubRipItem).
    :param name_space: The name of the file or namespace for this chunk.
    :param embed_word_limit: Number of words per embedding segment (default: 40).
    :return: List of Chunk objects.

    Example:
    If subs contain 120 words total:
    - Chunk 1: text_to_embed = words 1-40, full_text = all 120 words
    - Chunk 2: text_to_embed = words 41-80, full_text = all 120 words
    - Chunk 3: text_to_embed = words 81-120, full_text = all 120 words
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

    # Create chunks with 40-word segments for embedding
    chunks = []
    total_words = len(all_words)

    for i in range(0, total_words, embed_word_limit):
        # Get the next 40 words (or remaining words if less than 40)
        end_idx = min(i + embed_word_limit, total_words)
        text_to_embed = " ".join(all_words[i:end_idx])
        embed_size = end_idx - i

        chunk = Chunk(
            time_start=str(start_time),
            time_end=str(end_time),
            full_text=full_text,  # Same complete text for all chunks
            text_to_embed=text_to_embed,  # Unique 40-word segment
            chunk_size=total_words,  # Total word count across all subs
            embed_size=embed_size,  # Words in this specific embedding segment
            name_space=name_space,
        )
        chunks.append(chunk)

    return chunks


def chunk_srt(content: tuple[str, str]) -> list[Chunk]:
    file_name, text = content
    logger.debug(f"Chunking SRT file: {file_name}")
    subs = pysrt.from_string(text)
    logger.debug(f"Parsed {len(subs)} subtitle segments from {file_name}")

    chunks = []
    current_chunk = []
    word_count = 0

    for sub in subs:
        words = sub.text.replace("\n", " ").split()
        current_chunk.append(sub)
        word_count += len(words)

        if word_count >= FULL_TEXT_SIZE:
            # Build chunks from current_chunk and add them to chunks list
            new_chunks = build_chunks(current_chunk, file_name)
            chunks.extend(new_chunks)
            current_chunk = []
            word_count = 0

    # Handle remaining subtitles if any
    if current_chunk:
        new_chunks = build_chunks(current_chunk, file_name)
        chunks.extend(new_chunks)

    logger.debug(f"Created {len(chunks)} chunks from {file_name}")
    return chunks


def chunk_txt(content: tuple[str, str]) -> list[Chunk]:
    """
    Splits plain text into word-based chunks without timing metadata.

    :param content: A tuple containing (filename, raw text content).
    :return: A list of Chunk objects, each with text data, chunk size, and file-level metadata.
    """
    file_name, text = content
    logger.debug(f"Chunking TXT file: {file_name}")
    words = text.split()
    logger.debug(f"Found {len(words)} words in {file_name}")

    chunks: list[Chunk] = []

    for i in range(0, len(words), EMBEDDING_TEXT_SIZE):
        chunk_words = words[i : i + EMBEDDING_TEXT_SIZE]
        chunk_text = " ".join(chunk_words)

        chunk = Chunk(
            name_space=file_name,
            text_to_embed=chunk_text,
            chunk_size=len(chunk_words),
            time_start=None,
            time_end=None,
            full_text=chunk_text,
            embed_size=len(chunk_words),
        )
        chunks.append(chunk)

    logger.debug(f"Created {len(chunks)} chunks from {file_name}")
    return chunks


def preprocess_raw_transcripts(
    raw_transcripts: list[tuple[str, str]], data_format: TypeOfFormat = TypeOfFormat.SRT
) -> list[Chunk]:
    """
    Processes raw transcripts by applying preprocessing steps, including:

    1. Chunking data into fixed-size word batches
    2. Adding metadata to each chunk

    :param raw_transcripts: List of (filename, content) tuples
    :param data_format: Format of the data (e.g., SRT or TXT)
    :return: List of Chunk objects
    """
    if data_format is None:
        raise RuntimeError(f"Unknown data_format")
    logger.info(
        f"Starting preprocessing of {len(raw_transcripts)} transcripts with format: {data_format.name}"
    )
    cleaned_transcripts: list[Chunk] = []

    for i, raw_transcript in enumerate(raw_transcripts, 1):
        file_name = raw_transcript[0]
        logger.info(f"Processing transcript {i}/{len(raw_transcripts)}: {file_name}")

        if data_format.value == TypeOfFormat.SRT.value:
            chunks = chunk_srt(raw_transcript)
        elif data_format.value == TypeOfFormat.TXT.value:
            chunks = chunk_txt(raw_transcript)
        else:
            logger.error(f"Unsupported format: {data_format}")
            raise ValueError(f"Unsupported format: {data_format}")

        cleaned_transcripts.extend(chunks)
        logger.info(f"Completed processing {file_name}: {len(chunks)} chunks")

    logger.info(
        f"Preprocessing completed. Total chunks created: {len(cleaned_transcripts)}"
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
