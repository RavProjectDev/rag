import logging
from upstash_redis import Redis

logger = logging.getLogger(__name__)


class RedisConnection:
    """Redis connection wrapper for Upstash Redis."""
    
    def __init__(self, url: str, token: str):
        """
        Initialize Redis connection.
        
        Args:
            url: Upstash Redis REST URL
            token: Upstash Redis REST Token
        """
        self.redis = Redis(url=url, token=token)
        logger.info(f"Redis connection initialized with URL: {url}")
    
    async def increment_rate_limit(self, key: str, window_seconds: int = 3600) -> int:
        """
        Increment rate limit counter for a key.
        
        Args:
            key: The rate limit key (e.g., endpoint path)
            window_seconds: Time window in seconds for rate limiting
            
        Returns:
            Current count for the key
        """
        # Increment the key
        count = self.redis.incr(key)
        
        # Set expiration on first increment
        if count == 1:
            self.redis.expire(key, window_seconds)
        
        return count
    
    async def get_rate_limit(self, key: str) -> int:
        """
        Get current rate limit count for a key.
        
        Args:
            key: The rate limit key
            
        Returns:
            Current count for the key (0 if key doesn't exist)
        """
        count = self.redis.get(key)
        return int(count) if count else 0
    
    async def reset_rate_limit(self, key: str):
        """
        Reset rate limit counter for a key.
        
        Args:
            key: The rate limit key
        """
        self.redis.delete(key)
        logger.info(f"Rate limit reset for key: {key}")
