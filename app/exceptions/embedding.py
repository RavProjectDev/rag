from rag.app.exceptions.base import BaseAppException


class EmbeddingException(BaseAppException):
    status_code: int = 500
    code: str = "embedding_error"
    description: str = "General embedding error."

    def __init__(self, message: str | None = None):
        self.message = message or self.description
        super().__init__(self.message)


class EmbeddingTimeOutException(EmbeddingException):
    status_code = 408
    code = "embedding_timeout"
    description: str = "Embedding operation timed out."


class EmbeddingAPIException(EmbeddingException):
    status_code = 412
    code = "embedding_api_error"
    description: str = "Embedding API returned an error."


class EmbeddingConfigurationException(EmbeddingException):
    status_code = 422
    code = "embedding_config_error"
    description: str = "Invalid embedding configuration."


class VertexAIEmbeddingException(EmbeddingAPIException):
    status_code = 422
    code = "vertexai_error"
    description: str = "Issue connecting to VertexAI."
