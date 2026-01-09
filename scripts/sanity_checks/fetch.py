"""
Fetching utilities with infinite retry logic

Handles fetching manifest and transcripts with exponential backoff.
"""

import asyncio
import logging
import httpx
from typing import Dict, Any

from .config import (
    MANIFEST_URL,
    SANITY_CHECK_RETRY_DELAY,
    SANITY_CHECK_MAX_DELAY,
    SANITY_CHECK_BACKOFF_MULTIPLIER,
)

logger = logging.getLogger(__name__)


async def fetch_manifest() -> Dict[str, Any] | None:
    """
    Fetch manifest data from the API.
    
    Returns:
        Dictionary mapping document IDs to their metadata, or None on failure
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(MANIFEST_URL, json={})
            if response.status_code != 200:
                logger.error(f"[MANIFEST] Failed with status {response.status_code}")
                return None
            return response.json()
    except Exception as e:
        logger.error(f"[MANIFEST] Failed to fetch: {e}")
        return None


async def fetch_transcript_with_infinite_retry(
    transcript_url: str,
    doc_title: str,
    max_attempts: int | None = None
) -> str:
    """
    Fetch transcript content with infinite retries and exponential backoff.
    
    For sanity check, we MUST get the transcript to generate expected chunks.
    Will retry forever (or until max_attempts if specified) with exponential backoff.
    
    Args:
        transcript_url: URL to fetch transcript from
        doc_title: Document title for logging
        max_attempts: Optional maximum number of attempts (None = infinite)
        
    Returns:
        Transcript content as string
        
    Raises:
        Exception: Only if max_attempts is reached
    """
    attempt = 0
    delay = SANITY_CHECK_RETRY_DELAY
    
    while True:
        attempt += 1
        
        # Check max attempts if specified
        if max_attempts is not None and attempt > max_attempts:
            error_msg = f"Failed to fetch transcript after {max_attempts} attempts"
            logger.error(f"[FETCH FAILED] {doc_title}: {error_msg}")
            raise Exception(error_msg)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(transcript_url)
            
            if not response.content:
                raise Exception(f"Empty response from {transcript_url}")
            
            content = response.content.decode("utf-8")
            
            if attempt > 1:
                logger.info(
                    f"[FETCH SUCCESS] {doc_title} "
                    f"(succeeded on attempt {attempt})"
                )
            else:
                logger.debug(f"[FETCH SUCCESS] {doc_title}")
            
            return content
            
        except Exception as e:
            logger.warning(
                f"[FETCH RETRY] {doc_title} failed "
                f"(attempt {attempt}" + (f"/{max_attempts}" if max_attempts else "") + f"). "
                f"Retrying in {delay}s... Error: {e}"
            )
            
            # Log persistent issues
            if attempt % 10 == 0:
                logger.error(
                    f"[FETCH PERSISTENT] {doc_title} still failing after {attempt} attempts. "
                    f"This may indicate a persistent issue with URL: {transcript_url}"
                )
            
            await asyncio.sleep(delay)
            
            # Exponential backoff with cap
            delay = min(
                delay * SANITY_CHECK_BACKOFF_MULTIPLIER,
                SANITY_CHECK_MAX_DELAY
            )

