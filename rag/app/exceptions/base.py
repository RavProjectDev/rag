class BaseAppException(Exception):
    code: str = "app_error"
    description: str = "Base exception for all RAG app errors."
    status_code: int = 500

    def __init__(self, message: str | None = None):
        self.message = message or self.description
        super().__init__(self.message)
