import logging
from fastapi import Request, HTTPException
from starlette import status
from rag.app.db.redis_connection import RedisConnection

logger = logging.getLogger(__name__)


async def rate_limit_middleware(
    request: Request,
    redis_conn: RedisConnection,
    limit: int,
    window_seconds: int,
):
    """
    Rate limiting middleware using Redis.
    
    Args:
        request: FastAPI request object
        redis_conn: Redis connection instance
        limit: Maximum number of requests allowed in the window
        window_seconds: Time window in seconds for rate limiting
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    # Use the endpoint path as the key
    endpoint_key = f"rate_limit:{request.url.path}"
    
    try:
        # Increment the counter
        current_count = await redis_conn.increment_rate_limit(
            key=endpoint_key,
            window_seconds=window_seconds
        )
        
        logger.info(
            f"Rate limit check: endpoint={request.url.path}, "
            f"count={current_count}/{limit}, window={window_seconds}s"
        )
        
        # Check if limit exceeded
        if current_count > limit:
            logger.warning(
                f"Rate limit exceeded: endpoint={request.url.path}, "
                f"count={current_count}, limit={limit}"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Maximum {limit} requests per {window_seconds} seconds.",
                    "current_count": current_count,
                    "limit": limit,
                }
            )
        
        # Add rate limit headers to response (will be handled by the endpoint)
        request.state.rate_limit_remaining = limit - current_count
        request.state.rate_limit_limit = limit
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error but don't block the request if Redis fails
        logger.error(
            f"Rate limiting error: {str(e)}. Allowing request to proceed.",
            exc_info=True
        )
