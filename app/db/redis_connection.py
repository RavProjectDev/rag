import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional
from upstash_redis import Redis

logger = logging.getLogger(__name__)

_USER_RATE_LIMIT_TTL_SECONDS = 45 * 24 * 60 * 60  # 45 days — cleanup only, not reset logic
_ET_TZ = ZoneInfo("America/New_York")


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

    # ------------------------------------------------------------------
    # Endpoint-level rate limiting (fixed window, keyed by path)
    # ------------------------------------------------------------------

    async def increment_rate_limit(self, key: str, window_seconds: int = 3600) -> int:
        """
        Increment rate limit counter for a key.

        Args:
            key: The rate limit key (e.g., endpoint path)
            window_seconds: Time window in seconds for rate limiting

        Returns:
            Current count for the key
        """
        count = self.redis.incr(key)
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

    # ------------------------------------------------------------------
    # User-level rate limiting (monthly, keyed by YYYY-MM + user_id)
    # ------------------------------------------------------------------

    def _get_user_rate_limit_key(self, user_id: str) -> str:
        """
        Build the Redis key for the current month's user rate limit.

        Key format: rate_limit:YYYY-MM:<user_id>
        The month segment is derived from the current ET time so that a new
        calendar month automatically uses a new key — no TTL-based reset needed.
        """
        month_str = datetime.now(_ET_TZ).strftime("%Y-%m")
        return f"rate_limit:{month_str}:{user_id}"

    def _get_next_month_reset_time_et(self) -> datetime:
        """
        Return midnight ET on the 1st of next month — i.e. when the current
        month's rate limit counter stops being relevant.
        """
        now_et = datetime.now(_ET_TZ)
        if now_et.month == 12:
            return now_et.replace(year=now_et.year + 1, month=1, day=1,
                                  hour=0, minute=0, second=0, microsecond=0)
        return now_et.replace(month=now_et.month + 1, day=1,
                              hour=0, minute=0, second=0, microsecond=0)

    async def increment_user_rate_limit(self, user_id: str) -> int:
        """
        Increment the current month's rate limit counter for a user.

        The key encodes the calendar month (YYYY-MM), so monthly isolation is
        structural — not dependent on TTL expiry. The 45-day TTL is set on
        every new key purely for Redis garbage collection.

        Args:
            user_id: The user ID from JWT token

        Returns:
            Current count for the user this month
        """
        key = self._get_user_rate_limit_key(user_id)
        exists = self.redis.exists(key)
        count = self.redis.incr(key)

        if exists == 0:
            self.redis.expire(key, _USER_RATE_LIMIT_TTL_SECONDS)
            logger.info(
                f"Created user rate limit key: {key} "
                f"with {_USER_RATE_LIMIT_TTL_SECONDS}s TTL (garbage collection only)"
            )

        return count

    async def get_user_rate_limit(self, user_id: str) -> int:
        """
        Get the current month's rate limit count for a user.

        Args:
            user_id: The user ID from JWT token

        Returns:
            Current count for the user this month (0 if key doesn't exist)
        """
        key = self._get_user_rate_limit_key(user_id)
        count = self.redis.get(key)
        return int(count) if count else 0

    async def decrement_user_rate_limit(self, user_id: str) -> int:
        """
        Decrement the current month's rate limit counter for a user.
        Called when a request fails so it doesn't count against the user's limit.

        Args:
            user_id: The user ID from JWT token

        Returns:
            Current count for the user after decrement (0 if key doesn't exist)
        """
        key = self._get_user_rate_limit_key(user_id)
        exists = self.redis.exists(key)

        if exists == 0:
            logger.warning(f"Attempted to decrement non-existent rate limit key: {key}")
            return 0

        count = self.redis.decr(key)

        if count < 0:
            self.redis.set(key, 0)
            count = 0
            logger.warning(f"Rate limit key {key} was negative, reset to 0")

        logger.info(f"Decremented user rate limit for user_id={user_id}, new count={count}")
        return count

    async def get_user_rate_limit_info(self, user_id: str, limit: int) -> dict:
        """
        Get complete rate limit information for a user.

        Reset time is derived from the calendar (1st of next month at midnight ET),
        not from the Redis TTL, since the TTL is only for garbage collection.

        Args:
            user_id: The user ID from JWT token
            limit: The configured monthly limit

        Returns:
            dict with current_usage, remaining, limit, reset_at, reset_in_seconds
        """
        current_usage = await self.get_user_rate_limit(user_id)

        now_et = datetime.now(_ET_TZ)
        reset_time = self._get_next_month_reset_time_et()
        reset_in_seconds = int((reset_time - now_et).total_seconds())

        return {
            "current_usage": current_usage,
            "remaining": max(0, limit - current_usage),
            "limit": limit,
            "reset_at": reset_time.isoformat(),
            "reset_in_seconds": reset_in_seconds,
        }