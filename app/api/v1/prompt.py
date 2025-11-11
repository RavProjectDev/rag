from fastapi import APIRouter
from rag.app.services.prompts import PromptType, PROMPTS

router = APIRouter()


@router.get(
    "/",
    summary="Get all available prompts",
    description="Returns all available prompt templates with their types and contents.",
)
def get_prompts() -> dict[str, dict[str, str]]:
    """
    Retrieve all prompt templates.
    
    Returns a dictionary mapping prompt types to their details including
    the prompt type name and the full prompt template content.
    """
    result = {}
    
    for prompt_type in PromptType:
        prompt_key = prompt_type.value
        result[prompt_type.name] = {
            "type": prompt_key,
            "content": PROMPTS.get(prompt_key, "")
        }
    
    return result


@router.get(
    "/{prompt_type}",
    summary="Get a specific prompt",
    description="Returns a specific prompt template by its type.",
)
def get_prompt(prompt_type: PromptType) -> dict[str, str]:
    """
    Retrieve a specific prompt template.
    
    Args:
        prompt_type: The type of prompt to retrieve (MINIMAL, LIGHT, MODERATE, COMPREHENSIVE, or STRUCTURED_JSON)
    
    Returns:
        A dictionary containing the prompt type and its content.
    """
    prompt_key = prompt_type.value
    return {
        "type": prompt_key,
        "content": PROMPTS.get(prompt_key, "")
    }

