import logging
from fastapi import APIRouter, Depends, HTTPException
from starlette import status

from rag.app.core.config import get_settings
from rag.app.db.redis_connection import RedisConnection
from rag.app.dependencies import get_redis_conn
from rag.app.schemas.response import RateLimitInfoResponse
from rag.app.services.auth import verify_jwt_token

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/rate-limit",
    response_model=RateLimitInfoResponse,
    summary="Get user rate limit information",
    description=(
        "Returns the current rate limit status for the authenticated user, "
        "including usage, remaining requests, and reset time."
    ),
)
async def get_user_rate_limit(
    user_id: str = Depends(verify_jwt_token),
    redis_conn: RedisConnection | None = Depends(get_redis_conn),
) -> RateLimitInfoResponse:
    """
    Get rate limit information for the authenticated user.
    
    Returns:
        RateLimitInfoResponse with current usage, remaining requests, and reset time.
        
    Raises:
        HTTPException: 503 if Redis is not available.
    """
    if redis_conn is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "rate_limiting_unavailable",
                "message": "Rate limiting is not enabled on this server.",
            },
        )
    
    settings = get_settings()
    
    # Get rate limit info from Redis
    rate_limit_info = await redis_conn.get_user_rate_limit_info(
        user_id=user_id,
        limit=settings.user_rate_limit_max_requests_per_month,
    )
    
    logger.info(
        f"Rate limit info requested: user_id={user_id}, "
        f"usage={rate_limit_info['current_usage']}/{rate_limit_info['limit']}"
    )
    
    return RateLimitInfoResponse(
        user_id=user_id,
        current_usage=rate_limit_info["current_usage"],
        remaining=rate_limit_info["remaining"],
        limit=rate_limit_info["limit"],
        reset_at=rate_limit_info["reset_at"],
        reset_in_seconds=rate_limit_info["reset_in_seconds"],
    )
