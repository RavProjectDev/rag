"""AGENTIC_MULTI_CALL chunking strategy – two-stage LLM approach.

Stage 1 (1 call):  Detect semantic section boundaries across the full transcript.
Stage 2 (N calls, concurrent): For each section a dedicated LLM call rewrites the
    raw transcript into a clean, retrieval-optimised passage stored as `text_to_embed`,
    while the original timestamped segments are preserved in `full_text`.

Retry policy: exponential backoff with jitter on 429 / rate-limit responses.
"""

import asyncio
import concurrent.futures
import json
import logging
import random
import time
import uuid
from typing import Callable, Optional

from rag.app.schemas.data import Chunk, LLMModel
from rag.app.services.preprocess.constants import (
    SRT_LINES_PER_SEGMENT,
    AGENTIC_MIN_SECTION_SEGMENTS,
    AGENTIC_MAX_SECTION_SEGMENTS,
    AGENTIC_MULTI_CALL_MAX_RETRIES,
    AGENTIC_MULTI_CALL_RETRY_BASE_DELAY,
)
from rag.app.services.preprocess.strategies.agentic import (
    _validate_and_repair_sections,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage-1 prompts  (boundary detection — identical logic to agentic.py)
# ---------------------------------------------------------------------------

_BOUNDARY_SRT_PROMPT = """\
You are analyzing a transcript of a Rav Soloveitchik lecture.
Below is the full transcript as a numbered list of text segments, each covering \
{lines_per_segment} subtitle lines with its time range.

# Transcript Segments
{segments}

# Task
Identify the natural logical sections of this lecture. Each section should \
represent one coherent topic, argument, or passage that stands alone as a \
meaningful unit for retrieval.

Return ONLY a valid JSON object with this exact structure (no prose, no markdown):
{{
  "sections": [
    {{
      "start_segment": <integer, 0-indexed>,
      "end_segment": <integer, 0-indexed, inclusive>,
      "title": "<brief descriptive title for this section>"
    }}
  ]
}}

Rules:
- Every segment index from 0 to {last_segment_idx} must belong to exactly one section.
- Sections must be contiguous: the first section starts at 0, each subsequent \
section starts where the previous one ended + 1, and the last section ends at {last_segment_idx}.
- Aim for sections of {min_segs}–{max_segs} segments each (roughly 3–10 minutes of speech).
- Return ONLY valid JSON.\
"""

_BOUNDARY_TXT_PROMPT = """\
You are analyzing a plain-text transcript of a Rav Soloveitchik lecture.
Below is the full text split into numbered paragraphs.

# Transcript Paragraphs
{paragraphs}

# Task
Identify the natural logical sections of this lecture. Each section should \
represent one coherent topic, argument, or passage that stands alone as a \
meaningful unit for retrieval.

Return ONLY a valid JSON object with this exact structure (no prose, no markdown):
{{
  "sections": [
    {{
      "start_paragraph": <integer, 0-indexed>,
      "end_paragraph": <integer, 0-indexed, inclusive>,
      "title": "<brief descriptive title for this section>"
    }}
  ]
}}

Rules:
- Every paragraph index from 0 to {last_paragraph_idx} must belong to exactly one section.
- Sections must be contiguous: the first section starts at 0, each subsequent \
section starts where the previous one ended + 1, and the last section ends at \
{last_paragraph_idx}.
- Aim for sections of {min_segs}–{max_segs} paragraphs each.
- Return ONLY valid JSON.\
"""

_BOUNDARY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "agentic_sections",
        "schema": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_segment": {"type": "integer"},
                            "end_segment": {"type": "integer"},
                            "title": {"type": "string"},
                        },
                        "required": ["start_segment", "end_segment", "title"],
                    },
                }
            },
            "required": ["sections"],
            "additionalProperties": False,
        },
        "strict": False,
    },
}

_BOUNDARY_TXT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "agentic_txt_sections",
        "schema": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_paragraph": {"type": "integer"},
                            "end_paragraph": {"type": "integer"},
                            "title": {"type": "string"},
                        },
                        "required": ["start_paragraph", "end_paragraph", "title"],
                    },
                }
            },
            "required": ["sections"],
            "additionalProperties": False,
        },
        "strict": False,
    },
}


# ---------------------------------------------------------------------------
# Stage-2 prompt  (per-section refinement)
# ---------------------------------------------------------------------------

_REFINE_SECTION_PROMPT = """\
You are processing a section of a Rav Soloveitchik lecture transcript for a \
semantic search index.

# Section Title
{title}

# Raw Transcript Text
{raw_text}

# Task
Rewrite this section as a clean, coherent passage optimised for semantic retrieval.

Guidelines:
- Remove transcription artifacts: filler words ("um", "uh", "you know"), false \
starts, repetitions, and stutters.
- Preserve the full theological and philosophical content — do not omit ideas, \
arguments, or scriptural references.
- Keep the Rav's distinctive voice, terminology, and cadence.
- Write in flowing prose; do not add headings or bullet points.
- The rewrite should be self-contained so a reader with no prior context \
understands the idea.

Return ONLY a valid JSON object (no prose, no markdown):
{{"refined_text": "<the cleaned, retrieval-optimised prose>"}}
"""

_REFINE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "refined_section",
        "schema": {
            "type": "object",
            "properties": {
                "refined_text": {"type": "string"},
            },
            "required": ["refined_text"],
            "additionalProperties": False,
        },
        "strict": False,
    },
}


# ---------------------------------------------------------------------------
# Retry-aware LLM call
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception looks like a 429 / rate-limit response."""
    msg = str(exc).lower()
    keywords = ("429", "rate limit", "rate_limit", "quota", "resource_exhausted",
                "too many requests", "resourceexhausted")
    return any(kw in msg for kw in keywords)


async def _call_llm_with_retry(
    prompt: str,
    model: LLMModel,
    response_format: dict,
    label: str,
    max_retries: int = AGENTIC_MULTI_CALL_MAX_RETRIES,
    base_delay: float = AGENTIC_MULTI_CALL_RETRY_BASE_DELAY,
) -> str:
    """
    Async LLM call with exponential backoff + jitter on rate-limit errors.

    :param prompt: Prompt text.
    :param model: LLM model enum value.
    :param response_format: JSON schema dict for structured output.
    :param label: Human-readable label used in log messages (e.g. doc name / section title).
    :param max_retries: Maximum number of attempts (including the first).
    :param base_delay: Base delay in seconds for the first retry; doubles each time.
    :return: Raw string response from the LLM.
    :raises: Re-raises the last exception if all retries are exhausted.
    """
    from rag.app.services.llm import get_llm_response  # avoid circular import

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.perf_counter()
            response = await get_llm_response(
                prompt=prompt,
                model=model,
                response_format=response_format,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "[AGENTIC_MULTI] LLM call OK | label='%s' | model=%s | attempt=%d | elapsed=%dms",
                label, model.value, attempt, elapsed_ms,
            )
            return response

        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc) and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 1.0)
                logger.warning(
                    "[AGENTIC_MULTI] Rate-limit hit | label='%s' | attempt=%d/%d | "
                    "retrying in %.1fs | error: %s",
                    label, attempt, max_retries, delay, exc,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "[AGENTIC_MULTI] LLM call FAILED | label='%s' | attempt=%d/%d | error: %s",
                    label, attempt, max_retries, exc,
                )
                raise

    raise last_exc  # pragma: no cover


# ---------------------------------------------------------------------------
# Async/sync bridge
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine safely whether or not a loop is already running."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# SRT implementation
# ---------------------------------------------------------------------------

def build_chunks_agentic_multi_call_srt(
    sub_entries: list[dict],
    name_space: str,
    build_chunk_lines_fn: Callable,
    compute_text_hash_fn: Callable,
    model: LLMModel = LLMModel.GEMINI_FLASH,
) -> list[Chunk]:
    """
    AGENTIC_MULTI_CALL chunking for SRT files.

    Stage 1 — one LLM call identifies logical section boundaries across the
    full transcript (same approach as AGENTIC strategy).

    Stage 2 — one concurrent LLM call per section rewrites the raw transcript
    text into a clean, retrieval-optimised passage.  The refined text becomes
    `text_to_embed`; the original timestamped segments are kept in `full_text`.

    :param sub_entries: Flat list of subtitle entry dicts (from _flatten_subs).
    :param name_space: Source file name used as the chunk namespace.
    :param build_chunk_lines_fn: Shared helper that merges N sub-entries into
        (text, (start, end)) timestamp segments.
    :param compute_text_hash_fn: Shared SHA-256 helper.
    :param model: LLM to use for both boundary detection and section refinement.
    :return: List of Chunk objects, one per LLM-identified section.
    """
    if not sub_entries:
        return []

    return _run_async(
        _build_chunks_srt_async(
            sub_entries=sub_entries,
            name_space=name_space,
            build_chunk_lines_fn=build_chunk_lines_fn,
            compute_text_hash_fn=compute_text_hash_fn,
            model=model,
        )
    )


async def _build_chunks_srt_async(
    sub_entries: list[dict],
    name_space: str,
    build_chunk_lines_fn: Callable,
    compute_text_hash_fn: Callable,
    model: LLMModel,
) -> list[Chunk]:
    # ── Stage 1: boundary detection ──────────────────────────────────────────
    all_segments: list[tuple[str, tuple[Optional[str], Optional[str]]]] = (
        build_chunk_lines_fn(sub_entries, lines_per_segment=SRT_LINES_PER_SEGMENT)
    )
    total_segments = len(all_segments)

    formatted = "\n".join(
        f"[{idx}] ({start} \u2192 {end}) {text}"
        for idx, (text, (start, end)) in enumerate(all_segments)
    )

    boundary_prompt = _BOUNDARY_SRT_PROMPT.format(
        lines_per_segment=SRT_LINES_PER_SEGMENT,
        segments=formatted,
        last_segment_idx=total_segments - 1,
        min_segs=AGENTIC_MIN_SECTION_SEGMENTS,
        max_segs=AGENTIC_MAX_SECTION_SEGMENTS,
    )

    logger.info(
        "[AGENTIC_MULTI] Stage-1 boundary detection | doc='%s' | segments=%d | model=%s",
        name_space, total_segments, model.value,
    )

    raw_boundary = await _call_llm_with_retry(
        prompt=boundary_prompt,
        model=model,
        response_format=_BOUNDARY_RESPONSE_FORMAT,
        label=name_space,
    )

    try:
        raw_sections = json.loads(raw_boundary)["sections"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error(
            "[AGENTIC_MULTI] Failed to parse boundary response for '%s': %s\n"
            "Raw (first 500): %s",
            name_space, exc, raw_boundary[:500],
        )
        raise ValueError(
            f"AGENTIC_MULTI_CALL boundary detection failed for '{name_space}'"
        ) from exc

    sections = _validate_and_repair_sections(
        raw_sections,
        total_units=total_segments,
        unit_key_start="start_segment",
        unit_key_end="end_segment",
        name_space=name_space,
    )

    logger.info(
        "[AGENTIC_MULTI] Stage-1 complete | doc='%s' | sections=%d | titles=%s",
        name_space, len(sections), [s.get("title", "?") for s in sections],
    )

    # ── Stage 2: concurrent per-section refinement ───────────────────────────
    async def _refine_section(section: dict) -> tuple[dict, str]:
        start_seg = section["start_segment"]
        end_seg = section["end_segment"]
        title = section.get("title", "")

        first_entry_idx = start_seg * SRT_LINES_PER_SEGMENT
        last_entry_idx = min((end_seg + 1) * SRT_LINES_PER_SEGMENT, len(sub_entries))
        section_entries = sub_entries[first_entry_idx:last_entry_idx]
        raw_text = " ".join(entry["text"] for entry in section_entries)

        refine_prompt = _REFINE_SECTION_PROMPT.format(title=title, raw_text=raw_text)

        label = f"{name_space} § {title}"
        logger.info(
            "[AGENTIC_MULTI] Stage-2 refine START | doc='%s' | section='%s' | raw_chars=%d",
            name_space, title, len(raw_text),
        )

        raw_refine = await _call_llm_with_retry(
            prompt=refine_prompt,
            model=model,
            response_format=_REFINE_RESPONSE_FORMAT,
            label=label,
        )

        try:
            refined_text = json.loads(raw_refine)["refined_text"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "[AGENTIC_MULTI] Could not parse refined_text for section '%s' in '%s'; "
                "falling back to raw text.",
                title, name_space,
            )
            refined_text = raw_text

        logger.info(
            "[AGENTIC_MULTI] Stage-2 refine DONE | doc='%s' | section='%s' | "
            "raw_chars=%d → refined_chars=%d",
            name_space, title, len(raw_text), len(refined_text),
        )
        return section, refined_text

    results = await asyncio.gather(*[_refine_section(s) for s in sections])

    # ── Assemble Chunks ───────────────────────────────────────────────────────
    chunks: list[Chunk] = []
    for section, refined_text in results:
        start_seg = section["start_segment"]
        end_seg = section["end_segment"]

        first_entry_idx = start_seg * SRT_LINES_PER_SEGMENT
        last_entry_idx = min((end_seg + 1) * SRT_LINES_PER_SEGMENT, len(sub_entries))
        section_entries = sub_entries[first_entry_idx:last_entry_idx]

        if not section_entries:
            logger.warning(
                "[AGENTIC_MULTI] Empty section [%d–%d] in '%s', skipping.",
                start_seg, end_seg, name_space,
            )
            continue

        section_segments = all_segments[start_seg: end_seg + 1]
        time_start = section_segments[0][1][0]
        time_end = section_segments[-1][1][1]
        embed_tokens = sum(entry["tokens"] for entry in section_entries)
        text_hash = compute_text_hash_fn(refined_text)

        chunks.append(
            Chunk(
                full_text_id=uuid.uuid4(),
                time_start=time_start,
                time_end=time_end,
                full_text=section_segments,
                text_to_embed=refined_text,
                chunk_size=embed_tokens,
                embed_size=embed_tokens,
                name_space=name_space,
                text_hash=text_hash,
            )
        )

    logger.info(
        "[AGENTIC_MULTI] Created %d chunks from '%s'", len(chunks), name_space
    )
    return chunks


# ---------------------------------------------------------------------------
# TXT implementation
# ---------------------------------------------------------------------------

def build_chunks_agentic_multi_call_txt(
    tokens: list[int],
    encoder,
    name_space: str,
    compute_text_hash_fn: Callable,
    model: LLMModel = LLMModel.GEMINI_FLASH,
    paragraph_token_size: int = 200,
) -> list[Chunk]:
    """
    AGENTIC_MULTI_CALL chunking for plain-text files.

    Stage 1 — one LLM call identifies section boundaries across paragraph units.
    Stage 2 — one concurrent LLM call per section rewrites the raw text into a
    clean, retrieval-optimised passage stored as `text_to_embed`.

    :param tokens: Full token list from tiktoken.
    :param encoder: Tiktoken encoder instance.
    :param name_space: Source file name.
    :param compute_text_hash_fn: Shared SHA-256 helper.
    :param model: LLM to use for both stages.
    :param paragraph_token_size: Token budget per paragraph unit sent to the LLM.
    :return: List of Chunk objects, one per LLM-identified section.
    """
    if not tokens:
        return []

    return _run_async(
        _build_chunks_txt_async(
            tokens=tokens,
            encoder=encoder,
            name_space=name_space,
            compute_text_hash_fn=compute_text_hash_fn,
            model=model,
            paragraph_token_size=paragraph_token_size,
        )
    )


async def _build_chunks_txt_async(
    tokens: list[int],
    encoder,
    name_space: str,
    compute_text_hash_fn: Callable,
    model: LLMModel,
    paragraph_token_size: int,
) -> list[Chunk]:
    # ── Stage 1: boundary detection ──────────────────────────────────────────
    paragraphs: list[str] = []
    for i in range(0, len(tokens), paragraph_token_size):
        paragraphs.append(encoder.decode(tokens[i: i + paragraph_token_size]))

    total_paragraphs = len(paragraphs)
    formatted = "\n\n".join(f"[{idx}] {text}" for idx, text in enumerate(paragraphs))

    boundary_prompt = _BOUNDARY_TXT_PROMPT.format(
        paragraphs=formatted,
        last_paragraph_idx=total_paragraphs - 1,
        min_segs=AGENTIC_MIN_SECTION_SEGMENTS,
        max_segs=AGENTIC_MAX_SECTION_SEGMENTS,
    )

    logger.info(
        "[AGENTIC_MULTI] Stage-1 boundary detection | doc='%s' | paragraphs=%d | model=%s",
        name_space, total_paragraphs, model.value,
    )

    raw_boundary = await _call_llm_with_retry(
        prompt=boundary_prompt,
        model=model,
        response_format=_BOUNDARY_TXT_RESPONSE_FORMAT,
        label=name_space,
    )

    try:
        raw_sections = json.loads(raw_boundary)["sections"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error(
            "[AGENTIC_MULTI] Failed to parse boundary response for '%s': %s\n"
            "Raw (first 500): %s",
            name_space, exc, raw_boundary[:500],
        )
        raise ValueError(
            f"AGENTIC_MULTI_CALL boundary detection failed for '{name_space}'"
        ) from exc

    sections = _validate_and_repair_sections(
        raw_sections,
        total_units=total_paragraphs,
        unit_key_start="start_paragraph",
        unit_key_end="end_paragraph",
        name_space=name_space,
    )

    logger.info(
        "[AGENTIC_MULTI] Stage-1 complete | doc='%s' | sections=%d | titles=%s",
        name_space, len(sections), [s.get("title", "?") for s in sections],
    )

    para_token_starts = list(range(0, len(tokens), paragraph_token_size))

    # ── Stage 2: concurrent per-section refinement ───────────────────────────
    async def _refine_section(section: dict) -> tuple[dict, str, int, int]:
        start_para = section["start_paragraph"]
        end_para = section["end_paragraph"]
        title = section.get("title", "")

        token_start = para_token_starts[start_para]
        token_end = (
            para_token_starts[end_para + 1]
            if end_para + 1 < len(para_token_starts)
            else len(tokens)
        )
        section_tokens = tokens[token_start:token_end]
        raw_text = encoder.decode(section_tokens)

        refine_prompt = _REFINE_SECTION_PROMPT.format(title=title, raw_text=raw_text)

        label = f"{name_space} § {title}"
        logger.info(
            "[AGENTIC_MULTI] Stage-2 refine START | doc='%s' | section='%s' | raw_chars=%d",
            name_space, title, len(raw_text),
        )

        raw_refine = await _call_llm_with_retry(
            prompt=refine_prompt,
            model=model,
            response_format=_REFINE_RESPONSE_FORMAT,
            label=label,
        )

        try:
            refined_text = json.loads(raw_refine)["refined_text"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "[AGENTIC_MULTI] Could not parse refined_text for section '%s' in '%s'; "
                "falling back to raw text.",
                title, name_space,
            )
            refined_text = raw_text

        logger.info(
            "[AGENTIC_MULTI] Stage-2 refine DONE | doc='%s' | section='%s' | "
            "raw_chars=%d → refined_chars=%d",
            name_space, title, len(raw_text), len(refined_text),
        )
        return section, refined_text, token_start, token_end

    results = await asyncio.gather(*[_refine_section(s) for s in sections])

    # ── Assemble Chunks ───────────────────────────────────────────────────────
    chunks: list[Chunk] = []
    for section, refined_text, token_start, token_end in results:
        section_tokens = tokens[token_start:token_end]
        if not section_tokens:
            logger.warning(
                "[AGENTIC_MULTI] Empty section '%s' in '%s', skipping.",
                section.get("title", "?"), name_space,
            )
            continue

        token_count = len(section_tokens)
        text_hash = compute_text_hash_fn(refined_text)

        chunks.append(
            Chunk(
                full_text_id=uuid.uuid4(),
                time_start=None,
                time_end=None,
                full_text=encoder.decode(section_tokens),
                text_to_embed=refined_text,
                chunk_size=token_count,
                embed_size=token_count,
                name_space=name_space,
                text_hash=text_hash,
            )
        )

    logger.info(
        "[AGENTIC_MULTI] Created %d chunks from '%s'", len(chunks), name_space
    )
    return chunks
