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


async def user_rate_limit_middleware(
    user_id: str,
    redis_conn: RedisConnection,
    limit: int,
    request: Request = None,
):
    """
    User-based rate limiting middleware using Redis with monthly reset.
    Rate limits reset on the first of each month at midnight ET.
    
    Args:
        user_id: User ID extracted from JWT token
        redis_conn: Redis connection instance
        limit: Maximum number of requests allowed per month
        request: Optional FastAPI request object to store rate limit info
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    try:
        # Increment the user's counter
        current_count = await redis_conn.increment_user_rate_limit(user_id)
        
        logger.info(
            f"User rate limit check: user_id={user_id}, "
            f"count={current_count}/{limit}"
        )
        
        # Check if limit exceeded
        if current_count > limit:
            logger.warning(
                f"User rate limit exceeded: user_id={user_id}, "
                f"count={current_count}, limit={limit}"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "user_rate_limit_exceeded",
                    "message": f"Monthly rate limit exceeded. Maximum {limit} requests per month. Resets on the 1st of next month.",
                    "current_count": current_count,
                    "limit": limit,
                }
            )
        
        # Store rate limit info in request state for response headers
        if request is not None:
            # Get full rate limit info including reset time
            rate_limit_info = await redis_conn.get_user_rate_limit_info(user_id, limit)
            request.state.user_rate_limit_current = current_count
            request.state.user_rate_limit_limit = limit
            request.state.user_rate_limit_remaining = rate_limit_info["remaining"]
            request.state.user_rate_limit_reset = rate_limit_info["reset_at"]
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error but don't block the request if Redis fails
        logger.error(
            f"User rate limiting error for user_id={user_id}: {str(e)}. Allowing request to proceed.",
            exc_info=True
        )
