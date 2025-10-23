from rag.app.exceptions import BaseAppException


class BaseUploadException(BaseAppException):
    status_code: int = 500
    code: str = "upload_error"
    description: str = "General Upload error."

    def __init__(self, message: str | None = None):
        self.message = message or self.description
        super().__init__(self.message)


class SRTFileNotFound(BaseUploadException):
    status_code: int = 404
    code: str = "file_not_found"
    description: str = "SRT file not found."
