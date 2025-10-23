from rag.app.exceptions.base import BaseAppException


class DataBaseException(BaseAppException):
    status_code: int = 500
    code: str = "database_error"
    description: str = "A general database error occurred."

    def __init__(self, message: str | None = None):
        self.message = message or self.description
        super().__init__(self.message)


class RetrievalException(DataBaseException):
    status_code = 502
    code = "retrieval_error"
    description: str = "Failed to retrieve data from the database."

    def __init__(self, message: str | None = None):
        self.message = message or self.description
        super().__init__(self.message)


class RetrievalTimeoutException(DataBaseException):
    status_code = 504
    code = "retrieval_timeout"
    description: str = "Database retrieval timed out."

    def __init__(self, message: str | None = None):
        self.message = message or self.description
        super().__init__(self.message)


class InsertException(DataBaseException):
    status_code = 500
    code = "insert_error"
    description: str = "Failed to insert data into the database."

    def __init__(self, message: str | None = None):
        self.message = message or self.description
        super().__init__(self.message)


class NoDocumentFoundException(DataBaseException):
    status_code = 404
    code = "no_document_found"
    description: str = "Document not found similar to the given query"
    message_to_ui: str = (
        "Thank you for your question. Unfortunately, we do not have transcripts or teachings from Rav Soloveitchik on this topic in our database at this time. I encourage you to consult other sources or scholars for further insight "
    )

    def __init__(self, message: str | None = None):
        self.message = message or self.description
        super().__init__(self.message)
