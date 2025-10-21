from fastapi import APIRouter
from pydantic import BaseModel
from rag.app.exceptions import ALL_EXCEPTIONS, BaseAppException

router = APIRouter()


class ExceptionDoc(BaseModel):
    code: str
    description: str
    status_code: int


@router.get(
    "/exceptions",
    response_model=list[ExceptionDoc],
    include_in_schema=True,
    summary="List all application exception types",
    description="Returns the catalog of custom exceptions exposed by the API for documentation purposes.",
)
async def list_all_exceptions():
    docs = []
    for exc_cls in ALL_EXCEPTIONS:
        if issubclass(exc_cls, BaseAppException):
            docs.append(
                {
                    "code": getattr(exc_cls, "code"),
                    "status_code": getattr(exc_cls, "status_code"),
                    "description": getattr(exc_cls, "description"),
                }
            )
    return docs
