import argparse
import asyncio
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from upstash_redis import Redis

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_ET_TZ = ZoneInfo("America/New_York")
_DEFAULT_MONTHLY_LIMIT = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch sync Redis monthly rate-limit usage to Supabase."
    )
    parser.add_argument("--year", type=int, help="Target year (e.g. 2026)")
    parser.add_argument("--month", type=int, help="Target month (1-12)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Supabase")
    parser.add_argument("--scan-count", type=int, default=500, help="SCAN page size")
    parser.add_argument(
        "--write-batch-size", type=int, default=500, help="Supabase upsert batch size"
    )
    return parser.parse_args()


def target_period(year: int | None, month: int | None) -> tuple[int, int]:
    if year is not None and month is not None:
        if month < 1 or month > 12:
            raise ValueError("month must be between 1 and 12")
        return year, month

    if year is not None and month is None:
        raise ValueError("month is required when year is provided")
    if month is not None and year is None:
        raise ValueError("year is required when month is provided")

    now_et = datetime.now(_ET_TZ)
    return now_et.year, now_et.month


def _parse_env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw}") from exc


def _required_env(name: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return raw


def resolve_period(args: argparse.Namespace) -> tuple[int, int]:
    """
    Resolve target year/month with precedence:
      1) CLI flags (--year/--month)
      2) env vars (RATE_LIMIT_SYNC_YEAR/RATE_LIMIT_SYNC_MONTH)
      3) current ET month/year
    """
    env_year = _parse_env_int("RATE_LIMIT_SYNC_YEAR")
    env_month = _parse_env_int("RATE_LIMIT_SYNC_MONTH")

    cli_year = args.year
    cli_month = args.month

    selected_year = cli_year if cli_year is not None else env_year
    selected_month = cli_month if cli_month is not None else env_month
    return target_period(selected_year, selected_month)


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def extract_user_id(redis_key: str) -> str:
    # Expected: rate_limit:YYYY-MM:<user_id>
    parts = redis_key.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected key format: {redis_key}")
    return parts[2]


def scan_month_keys(redis: Redis, mkey: str, scan_count: int) -> list[str]:
    pattern = f"rate_limit:{mkey}:*"
    keys: list[str] = []

    # Prefer SCAN; fallback to KEYS if SCAN is not available.
    scan_method = getattr(redis, "scan", None)
    if callable(scan_method):
        cursor = 0
        while True:
            result = scan_method(cursor=cursor, match=pattern, count=scan_count)
            if not isinstance(result, (list, tuple)) or len(result) != 2:
                break

            cursor, batch = result
            if batch:
                keys.extend(batch)

            cursor = int(cursor)
            if cursor == 0:
                break
    else:
        keys = redis.keys(pattern) or []

    return keys


def build_rows(
    redis: Redis,
    keys: list[str],
    year: int,
    month: int,
    limit_per_month: int,
) -> list[dict]:
    now_iso = datetime.now(_ET_TZ).isoformat()
    rows: list[dict] = []

    for key in keys:
        raw = redis.get(key)
        usage = int(raw) if raw else 0
        user_id = extract_user_id(key)

        rows.append(
            {
                "user_id": user_id,
                "year": year,
                "month": month,
                "current_usage": usage,
                "remaining": max(0, limit_per_month - usage),
                "limit": limit_per_month,
                "updated_at": now_iso,
            }
        )

    return rows


async def upsert_rows(
    supabase_url: str,
    service_role_key: str,
    table_name: str,
    rows: list[dict],
    batch_size: int,
) -> int:
    if not rows:
        return 0

    url = f"{supabase_url}/rest/v1/{table_name}?on_conflict=user_id,year,month"
    headers = {
        "Content-Type": "application/json",
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    written = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            response = await client.post(url, headers=headers, json=batch)
            response.raise_for_status()
            written += len(batch)

    return written


async def main() -> None:
    load_dotenv()
    args = parse_args()
    upstash_redis_rest_url = _required_env("UPSTASH_REDIS_REST_URL")
    upstash_redis_rest_token = _required_env("UPSTASH_REDIS_REST_TOKEN")
    supabase_url = _required_env("SUPABASE_URL")
    supabase_service_role_key = _required_env("SUPABASE_SERVICE_ROLE_KEY")

    year, month = resolve_period(args)
    mkey = month_key(year, month)
    table_name = "rate_limit_usage"

    logger.info("=" * 60)
    logger.info("RATE LIMIT BATCH SYNC")
    logger.info("=" * 60)
    logger.info("Target month: %s", mkey)
    logger.info("Dry run: %s", args.dry_run)

    redis = Redis(
        url=upstash_redis_rest_url,
        token=upstash_redis_rest_token,
    )

    keys = scan_month_keys(redis, mkey, args.scan_count)
    logger.info("Found %d Redis keys for month %s", len(keys), mkey)

    rows = build_rows(
        redis=redis,
        keys=keys,
        year=year,
        month=month,
        limit_per_month=_DEFAULT_MONTHLY_LIMIT,
    )
    logger.info("Prepared %d rows for Supabase upsert", len(rows))

    if args.dry_run:
        logger.info("Dry run enabled, skipping Supabase write.")
        return

    written = await upsert_rows(
        supabase_url=supabase_url,
        service_role_key=supabase_service_role_key,
        table_name=table_name,
        rows=rows,
        batch_size=args.write_batch_size,
    )
    logger.info("Upserted %d rows into Supabase table '%s'", written, table_name)


if __name__ == "__main__":
    asyncio.run(main())
