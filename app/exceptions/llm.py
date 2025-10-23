from rag.app.exceptions.base import BaseAppException


class LLMBaseException(BaseAppException):
    status_code: int = 500
    code: str = "llm_error"
    description: str = "General LLM embedding error."

    def __init__(self, message: str | None = None):
        self.message = message or self.description
        super().__init__(self.message)


class LLMConnectionException(LLMBaseException):
    status_code: int = 500
    code: str = "llm_error_connection_exception"
    description: str = "Problem establishing connection with LLM Client API"


class LLMTimeoutException(LLMBaseException):
    status_code: int = 500
    code: str = "llm_timeout_error"
    description: str = "LLM request timed out."
