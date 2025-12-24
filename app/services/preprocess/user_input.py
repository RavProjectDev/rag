import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "whom",
    "whose",
    "is",
    "the",
    "a",
}


@lru_cache(maxsize=1)
def get_spacy_model():
    """Load and cache spaCy model for sentence segmentation."""
    try:
        import spacy
        # Try to load the model, if not available return None
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            return None
    except ImportError:
        logger.warning("spaCy not installed. Install with: pip install spacy")
        return None


def remove_last_sentence(text: str) -> str:
    """
    Remove the last sentence from text to avoid cut-off sentences.
    Uses spaCy for sentence segmentation if available, otherwise returns original text.
    
    Args:
        text: The text to process
        
    Returns:
        The text without the last sentence, or original text if processing fails
    """
    if not text or not text.strip():
        return text
    
    nlp = get_spacy_model()
    if nlp is None:
        logger.warning("spaCy model not available, returning original text")
        return text
    
    try:
        doc = nlp(text)
        sentences = list(doc.sents)
        
        # If there's only one sentence or no sentences, return empty or original
        if len(sentences) <= 1:
            logger.info(f"Text has {len(sentences)} sentence(s), returning original text")
            return text
        
        # Join all sentences except the last one
        result = " ".join(sent.text for sent in sentences[:-1])
        logger.debug(f"Removed last sentence. Original: {len(sentences)} sentences, Result: {len(sentences)-1} sentences")
        return result
    except Exception as e:
        logger.error(f"Error removing last sentence: {e}, returning original text")
        return text


def pre_process_user_query(user_query: str) -> str:
    user_query = user_query.strip()
    user_query = remove_question_words(user_query)
    return user_query


def remove_question_words(text: str) -> str:
    words = text.split()
    filtered = [word for word in words if word.lower() not in QUESTION_WORDS]
    result = " ".join(filtered)
    return result
