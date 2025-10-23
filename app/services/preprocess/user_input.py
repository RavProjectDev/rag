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


def pre_process_user_query(user_query: str) -> str:
    user_query = user_query.strip()
    user_query = remove_question_words(user_query)
    return user_query


def remove_question_words(text: str) -> str:
    words = text.split()
    filtered = [word for word in words if word.lower() not in QUESTION_WORDS]
    result = " ".join(filtered)
    return result
