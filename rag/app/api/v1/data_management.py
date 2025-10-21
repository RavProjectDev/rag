from fastapi import APIRouter, Depends, HTTPException

from rag.app.db.connections import EmbeddingConnection
from rag.app.dependencies import (
    get_embedding_conn,
    get_embedding_configuration,
)
from rag.app.exceptions import BaseAppException
from rag.app.exceptions.upload import BaseUploadException
from rag.app.models.data import SanityData
from rag.app.schemas.data import (
    EmbeddingConfiguration,
)
from rag.app.schemas.response import UploadResponse, ErrorResponse, SuccessResponse
from rag.app.services.data_upload_service import (
    upload_document as upload_document_service,
    delete_document as delete_document_service,
    update_document as update_documents_service,
)

router = APIRouter()


@router.post(
    "/create",
    response_model=UploadResponse,
    summary="Upload a new transcript document",
    description="Creates a new document by fetching transcript and storing chunk embeddings.",
    responses={
        200: {"model": UploadResponse},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload_files(
    upload_request: SanityData,
    embedding_conn: EmbeddingConnection = Depends(get_embedding_conn),
    embedding_configuration: EmbeddingConfiguration = Depends(
        get_embedding_configuration
    ),
):
    """
      Upload endpoint for subtitle (.srt) files.

      Accepts multiple .srt files via multipart/form-data, processes
      their textual content into data chunks, generates vector embeddings
      for each chunk, and stores the embeddings in the database.

      Request:
      --------
    b'{"_id":"55806772-3246-4eaf-88a3-4448eb39846e","_updatedAt":"2025-07-15T20:31:24Z","slug":"kedusha-and-malchus","title":"Kedusha and Malchus","transcriptURL":"https://cdn.sanity.io/files/ybwh5ic4/primary/2fbb38de4c27f54dfe767841cde0dae92c4be543.srt"}'
      Response:
      ---------
      JSON object containing:
          {
              "results": [
                  {
                      "vector": [...],
                      "dimension": <int>,
                      "data": {
                          "text": <str>,
                          ...
                      }
                  },
                  ...
              ]
          }

      Raises:
      -------
      HTTPException (400):
          If any uploaded file is not an .srt file.

      HTTPException (500):
          For any unexpected server-side errors.
    """
    try:
        await upload_document_service(
            upload_request, embedding_conn, embedding_configuration
        )

    except BaseUploadException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "message": e.message,
                "code": e.code,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": BaseAppException.code,
                "message": str(e),
            },
        )

    return UploadResponse(message="success")


@router.patch(
    "/update",
    response_model=SuccessResponse,
    summary="Update an existing document",
    description="Re-embed and update an existing document from its transcript source.",
)
async def update_files(
    update_request: SanityData,
    embedding_conn: EmbeddingConnection = Depends(get_embedding_conn),
    embedding_configuration: EmbeddingConfiguration = Depends(
        get_embedding_configuration
    ),
):
    try:
        await update_documents_service(
            update_request, embedding_conn, embedding_configuration
        )
    except BaseUploadException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "message": e.message,
                "code": e.code,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": BaseAppException.code,
                "message": str(e),
            },
        )
    return SuccessResponse(success=True, message="updated")


@router.delete(
    "/delete",
    response_model=SuccessResponse,
    summary="Delete a document",
    description="Deletes a document and its embeddings by id.",
)
async def delete_files(
    delete_request: SanityData,
    embedding_conn: EmbeddingConnection = Depends(get_embedding_conn),
):
    try:
        await delete_document_service(delete_request.id, embedding_conn)
    except BaseUploadException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "message": e.message,
                "code": e.code,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": BaseAppException.code,
                "message": str(e),
            },
        )
    return SuccessResponse(success=True, message="deleted")
