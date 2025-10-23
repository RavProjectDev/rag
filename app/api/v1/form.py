import asyncio
import random

from fastapi import APIRouter, HTTPException, Depends
from fastapi import Request
from fastapi.params import Path
from motor.motor_asyncio import AsyncIOMotorClient
from starlette import status

from rag.app import form_data
from rag.app.core.config import COLLECTIONS
from rag.app.core.config import get_settings
from rag.app.db.connections import EmbeddingConnection, MetricsConnection
from rag.app.db.mongodb_connection import MongoEmbeddingStore
from rag.app.dependencies import (
    get_embedding_conn,
    get_metrics_conn,
    get_embedding_configuration,
    get_llm_configuration,
)
from rag.app.exceptions.embedding import (
    EmbeddingException,
)
from rag.app.form_data.data import QUESTIONS
from rag.app.models.data import DocumentModel, Prompt
from rag.app.schemas.data import EmbeddingConfiguration, LLMModel
from rag.app.schemas.form import (
    RatingsModel,
    UploadRatingsRequestCHUNK,
    UploadRatingsRequestFULL,
    FullResponseRankingModel,
    DataFullUpload,
    CommentModel,
    FormRequestType,
)
from rag.app.schemas.requests import TypeOfRequest
from rag.app.schemas.response import ChatResponse, TranscriptData, FormFullResponse
from rag.app.schemas.response import (
    FormGetChunksResponse,
    ErrorResponse,
    SuccessResponse,
)
from rag.app.services.embedding import generate_embedding
from rag.app.services.form import get_all_form_data
from rag.app.services.llm import get_llm_response, generate_prompt
from rag.app.services.preprocess.user_input import pre_process_user_query

SITE_EMBEDDING_EVAL_COLLECTION = "site_data_chunking_perfromance"
SITE_FULL_PERFORMANCE_EVAL_COLLECTION = "site_data_prompt_performance"


router = APIRouter()


@router.get(
    "/{question}",
    response_model=FormGetChunksResponse,
    summary="Retrieve nearest chunks for a question",
    description="Generates an embedding for the question and returns the most similar chunks from a random collection.",
    responses={
        200: {"model": FormGetChunksResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_chunks(
    app_state: Request,
    question: str = Path(...),
    embedding_configuration: EmbeddingConfiguration = Depends(
        get_embedding_configuration
    ),
    connection: EmbeddingConnection = Depends(get_embedding_conn),
) -> FormGetChunksResponse:
    # Preprocess question
    client: AsyncIOMotorClient = app_state.app.state.db_client
    collection_name = random.choice(COLLECTIONS)
    collection = client[collection_name]
    connection: EmbeddingConnection = MongoEmbeddingStore(
        collection=collection,
        index=get_settings().collection_index,
        vector_path=get_settings().vector_path,
    )
    cleaned_question = pre_process_user_query(question)
    # Generate embedding with metrics logging

    try:
        embedding = await generate_embedding(
            text=cleaned_question,
            configuration=embedding_configuration,
        )
        vector: list[float] = embedding.vector
        data: list[DocumentModel]
        data = await connection.retrieve(embedded_data=vector, threshold=0.85)

    except EmbeddingException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": str(e.message),
                "code": e.code,
                "description": e.description,
            },
        )
    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    return FormGetChunksResponse(
        documents=data,
        embedding_type=collection_name,
    )


@router.get(
    "/full/{question}",
    summary="Return stored full response for a question (if any)",
    description="Returns a pre-cached full response for the given question if present.",
)
async def get_full_response(
    app_state: Request,
    question: str = Path(...),
):
    return form_data.data.data.get(question, "")


@router.post(
    "/upload_ratings/chunk",
    response_model=SuccessResponse,
    summary="Upload chunk-level ratings",
    description="Store per-chunk ratings for a user question and embedding type.",
)
async def upload_ratings(request: UploadRatingsRequestCHUNK, app_state: Request):
    client: AsyncIOMotorClient = app_state.app.state.db_client
    collection = client[SITE_EMBEDDING_EVAL_COLLECTION]
    for data in request.data:
        model = RatingsModel(
            user_question=request.user_question,
            ratings=data,
            embedding_type=request.embedding_type,
        )
        await collection.insert_one(model.to_dict())
    comment_model = CommentModel(
        comments=request.comments,
        user_question=request.user_question,
        type_of_request=FormRequestType.CHUNK,
    )
    await collection.insert_one(comment_model.to_dict())
    return SuccessResponse(success=True, message="ratings uploaded")


@router.post(
    "/upload_ratings/full",
    response_model=SuccessResponse,
    summary="Upload full-response ranking",
    description="Stores ranking for full LLM responses associated with prompt ids.",
)
async def upload_ratings_full(request: UploadRatingsRequestFULL, app_state: Request):
    client: AsyncIOMotorClient = app_state.app.state.db_client
    collection = client[SITE_FULL_PERFORMANCE_EVAL_COLLECTION]
    for ranking in request.rankings:
        model = FullResponseRankingModel(
            user_question=request.user_question,
            ranking_data=DataFullUpload(
                prompt_id=ranking.prompt_id,
                rank=ranking.rank,
            ),
        )
        await collection.insert_one(model.to_dict())
    comment_model = CommentModel(
        comments=request.comments,
        user_question=request.user_question,
        type_of_request=FormRequestType.PROMPT,
    )
    collection.insert_one(comment_model.model_dump())
    return SuccessResponse(success=True, message="full rankings uploaded")


async def _generate(
    user_question: str,
    embedding_configuration: EmbeddingConfiguration,
    connection: EmbeddingConnection,
) -> tuple[list[Prompt], list[TranscriptData]]:
    cleaned_question = pre_process_user_query(user_question)
    try:
        embedding = await generate_embedding(
            text=cleaned_question,
            configuration=embedding_configuration,
        )

    except Exception as e:
        raise EmbeddingException(f"Unexpected embedding error: {str(e)}")

    if embedding is None:
        raise EmbeddingException(f"Could not generate embedding for {user_question}")

    vector: list[float] = embedding.vector
    try:
        data = await connection.retrieve(
            embedded_data=vector,
        )
    except Exception as e:
        raise e

    prompts: list[Prompt] = []
    try:
        for i in range(1, 4):
            prompts.append(generate_prompt(cleaned_question, data, prompt_id=i))
    except Exception as e:
        raise

    transcript_data: list[TranscriptData] = []
    for datum in data:
        transcript_data.append(
            TranscriptData(
                sanity_data=datum.sanity_data,
                metadata=datum.metadata,
                score=datum.score,
            )
        )

    return prompts, transcript_data


@router.get(
    "/data/get_all_questions",
    response_model=list[str],
    summary="List all evaluation questions",
)
async def get_questions(app_state: Request):

    return QUESTIONS
