import json
import os
import dotenv

dotenv.load_dotenv()

import logging
from pathlib import Path
from typing import Dict, List

import certifi
from motor.motor_asyncio import AsyncIOMotorClient

from rag.app.db.mongodb_connection import MongoEmbeddingStore
from rag.app.exceptions.embedding import EmbeddingException
from rag.app.form_data.data import QUESTIONS
from rag.app.models.data import Prompt
from rag.app.schemas.data import EmbeddingConfiguration, LLMModel
from rag.app.services.embedding import generate_embedding
from rag.app.services.llm import get_llm_response, generate_prompt
from rag.app.services.preprocess.user_input import pre_process_user_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_FILE_PATH = Path(
    "/Users/dothanbardichev/Desktop/RAV/RAG/rag/app/form_data/questions_llm_response_map.json"
)
MAX_PROMPTS = 3
MONGODB_POOL_SIZE = 50


async def setup_connections() -> (
    tuple[MongoEmbeddingStore, EmbeddingConfiguration, LLMModel]
):
    """Set up database connections and configurations."""
    required_env_vars = [
        "MONGODB_URI",
        "MONGODB_DB_NAME",
        "MONGODB_VECTOR_COLLECTION",
        "COLLECTION_INDEX",
        "VECTOR_PATH",
    ]

    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    client = AsyncIOMotorClient(
        os.environ["MONGODB_URI"],
        tlsCAFile=certifi.where(),
        maxPoolSize=MONGODB_POOL_SIZE,
    )

    db = client[os.environ["MONGODB_DB_NAME"]]
    vector_embedding_collection = db[os.environ["MONGODB_VECTOR_COLLECTION"]]

    mongo_connection = MongoEmbeddingStore(
        collection=vector_embedding_collection,
        index=os.environ["COLLECTION_INDEX"],
        vector_path=os.environ["VECTOR_PATH"],
    )

    embedding_configuration = EmbeddingConfiguration.GEMINI
    llm_configuration = LLMModel.GPT_4

    return mongo_connection, embedding_configuration, llm_configuration


async def generate_prompts(
    question: str,
    mongo_connection: MongoEmbeddingStore,
    embedding_config: EmbeddingConfiguration,
) -> List[Prompt]:
    """Generate prompts for a question."""
    cleaned_question = pre_process_user_query(question)

    # Generate embedding
    embedding = await generate_embedding(
        text=cleaned_question,
        configuration=embedding_config,
    )

    if embedding is None:
        raise EmbeddingException(f"Could not generate embedding for {question}")

    # Retrieve relevant data
    data = await mongo_connection.retrieve(embedded_data=embedding.vector)

    # Generate prompts
    prompts = []
    for i in range(1, MAX_PROMPTS + 1):
        prompt = generate_prompt(cleaned_question, data, prompt_id=i)
        prompts.append(prompt)

    return prompts


def load_existing_map() -> Dict[str, List[str]]:
    """Load existing question-response map from file."""
    if OUTPUT_FILE_PATH.exists():
        try:
            with open(OUTPUT_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing map: {e}. Starting fresh.")
    return {}


def save_map(question_response_map: Dict[str, List[str]]) -> None:
    """Save question-response map to file."""
    OUTPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(question_response_map, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved progress: {len(question_response_map)} questions completed")


async def build_question_response_map() -> None:
    """Build dictionary mapping questions to list of responses."""
    mongo_connection, embedding_config, llm_config = await setup_connections()

    # Load existing map to preserve previous progress
    question_response_map = load_existing_map()
    initial_count = len(question_response_map)

    logger.info(f"Loaded {initial_count} existing questions")
    logger.info(f"Processing {len(QUESTIONS)} total questions")

    for i, question in enumerate(QUESTIONS, 1):
        # Skip if already processed
        if question in question_response_map:
            logger.info(
                f"Skipping question {i}/{len(QUESTIONS)} (already processed): {question[:50]}..."
            )
            continue

        try:
            logger.info(f"Processing question {i}/{len(QUESTIONS)}: {question[:50]}...")

            # Generate prompts
            prompts = await generate_prompts(
                question, mongo_connection, embedding_config
            )

            # Get responses for each prompt
            responses = []
            for prompt in prompts:
                llm_response = await get_llm_response(
                    prompt=prompt.value,
                    model=llm_config,
                )
                responses.append(llm_response)

            question_response_map[question] = responses

        except Exception as e:
            logger.error(f"Failed to process question '{question}': {e}")
            question_response_map[question] = []

        # Save after each question
        save_map(question_response_map)

    successful_count = len([q for q, r in question_response_map.items() if r])
    logger.info(f"Final results: {successful_count} questions processed successfully")
    logger.info(f"Question-response map saved to {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(build_question_response_map())
