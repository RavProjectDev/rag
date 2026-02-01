import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Optional
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
    
    def _calculate_seconds_until_next_month_et(self) -> int:
        """
        Calculate seconds until the first of the next month at midnight ET.
        
        Returns:
            Number of seconds until first of next month at 00:00:00 ET
        """
        # Get current time in ET timezone
        et_tz = ZoneInfo("America/New_York")
        now_et = datetime.now(et_tz)
        
        # Calculate first day of next month
        if now_et.month == 12:
            next_month = now_et.replace(year=now_et.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            next_month = now_et.replace(month=now_et.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate difference in seconds
        seconds_until_reset = int((next_month - now_et).total_seconds())
        
        logger.info(
            f"Calculated TTL until next month: {seconds_until_reset}s "
            f"(current ET: {now_et.isoformat()}, reset at: {next_month.isoformat()})"
        )
        
        return seconds_until_reset
    
    async def increment_user_rate_limit(self, user_id: str) -> int:
        """
        Increment rate limit counter for a user with monthly reset.
        Creates the key with TTL until first of next month ET if it doesn't exist.
        
        Args:
            user_id: The user ID from JWT token
            
        Returns:
            Current count for the user
        """
        key = f"rate_limit:{user_id}"
        
        # Check if key exists
        exists = self.redis.exists(key)
        
        # Increment the key
        count = self.redis.incr(key)
        
        # Set expiration on first increment (key didn't exist before)
        if exists == 0:
            ttl_seconds = self._calculate_seconds_until_next_month_et()
            self.redis.expire(key, ttl_seconds)
            logger.info(
                f"Created new user rate limit key: {key} with TTL of {ttl_seconds}s "
                f"(resets first of next month ET)"
            )
        
        return count
    
    async def get_user_rate_limit(self, user_id: str) -> int:
        """
        Get current rate limit count for a user.
        
        Args:
            user_id: The user ID from JWT token
            
        Returns:
            Current count for the user (0 if key doesn't exist)
        """
        key = f"rate_limit:{user_id}"
        count = self.redis.get(key)
        return int(count) if count else 0
    
    async def decrement_user_rate_limit(self, user_id: str) -> int:
        """
        Decrement rate limit counter for a user.
        Used when a request fails and should not count against the user's limit.
        
        Args:
            user_id: The user ID from JWT token
            
        Returns:
            Current count for the user after decrement (0 if key doesn't exist)
        """
        key = f"rate_limit:{user_id}"
        
        # Check if key exists
        exists = self.redis.exists(key)
        
        if exists == 0:
            logger.warning(f"Attempted to decrement non-existent rate limit key: {key}")
            return 0
        
        # Decrement the key (won't go below 0)
        count = self.redis.decr(key)
        
        # Ensure count doesn't go negative
        if count < 0:
            self.redis.set(key, 0)
            count = 0
            logger.warning(f"Rate limit key {key} was negative, reset to 0")
        
        logger.info(f"Decremented user rate limit for user_id={user_id}, new count={count}")
        return count
    
    def _get_next_month_reset_time_et(self) -> datetime:
        """
        Get the exact datetime when rate limit resets (first of next month at midnight ET).
        
        Returns:
            datetime object for first of next month at 00:00:00 ET
        """
        et_tz = ZoneInfo("America/New_York")
        now_et = datetime.now(et_tz)
        
        # Calculate first day of next month
        if now_et.month == 12:
            next_month = now_et.replace(year=now_et.year + 1, month=1, day=1, 
                                         hour=0, minute=0, second=0, microsecond=0)
        else:
            next_month = now_et.replace(month=now_et.month + 1, day=1, 
                                         hour=0, minute=0, second=0, microsecond=0)
        
        return next_month
    
    async def get_user_rate_limit_info(self, user_id: str, limit: int) -> dict:
        """
        Get complete rate limit information for a user.
        
        Args:
            user_id: The user ID from JWT token
            limit: The configured monthly limit
            
        Returns:
            dict with current_usage, remaining, limit, reset_at, reset_in_seconds
        """
        key = f"rate_limit:{user_id}"
        
        # Get current count
        current_usage = await self.get_user_rate_limit(user_id)
        
        # Get TTL
        ttl = self.redis.ttl(key)
        
        if ttl <= 0:
            # Key doesn't exist or has no expiration - calculate next reset
            reset_time = self._get_next_month_reset_time_et()
            now_et = datetime.now(ZoneInfo("America/New_York"))
            ttl = int((reset_time - now_et).total_seconds())
        else:
            # Use actual TTL from Redis
            et_tz = ZoneInfo("America/New_York")
            now_et = datetime.now(et_tz)
            reset_time = now_et + timedelta(seconds=ttl)
        
        return {
            "current_usage": current_usage,
            "remaining": max(0, limit - current_usage),
            "limit": limit,
            "reset_at": reset_time.isoformat(),
            "reset_in_seconds": ttl,
        }
