"""AGENTIC chunking strategy - uses an LLM to identify logical section boundaries."""
import asyncio
import concurrent.futures
import json
import logging
import time
import uuid
from typing import Callable, Optional

from rag.app.schemas.data import Chunk, LLMModel
from rag.app.services.preprocess.constants import (
    SRT_LINES_PER_SEGMENT,
    AGENTIC_MIN_SECTION_SEGMENTS,
    AGENTIC_MAX_SECTION_SEGMENTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_AGENTIC_SRT_PROMPT = """\
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

_AGENTIC_TXT_PROMPT = """\
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

_RESPONSE_FORMAT = {
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


# ---------------------------------------------------------------------------
# Async/sync bridge
# ---------------------------------------------------------------------------

def _run_async(coro):
    """
    Run an async coroutine regardless of whether we are already inside a
    running event loop (e.g. called from an async script) or not.

    When there IS a running loop we submit the coroutine to a fresh thread so
    that asyncio.run() can create its own loop there without conflicting.
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


async def _call_llm(prompt: str, model: LLMModel, name_space: str) -> str:
    from rag.app.services.llm import get_llm_response  # local import to avoid circular deps

    prompt_chars = len(prompt)
    logger.info(
        "[AGENTIC] LLM API call START  | doc='%s' | model=%s | prompt_chars=%d",
        name_space,
        model.value,
        prompt_chars,
    )
    t0 = time.perf_counter()

    response = await get_llm_response(
        prompt=prompt,
        model=model,
        response_format=_RESPONSE_FORMAT,
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[AGENTIC] LLM API call END    | doc='%s' | model=%s | elapsed=%dms | response_chars=%d",
        name_space,
        model.value,
        elapsed_ms,
        len(response),
    )
    return response


# ---------------------------------------------------------------------------
# Validation / repair helpers
# ---------------------------------------------------------------------------

def _validate_and_repair_sections(
    sections: list[dict],
    total_units: int,
    unit_key_start: str,
    unit_key_end: str,
    name_space: str,
) -> list[dict]:
    """
    Ensure sections are contiguous, cover [0, total_units-1], and have sane
    bounds.  Returns a clean list or raises ValueError if the response is
    fundamentally broken.
    """
    if not sections:
        raise ValueError("LLM returned zero sections.")

    # Sort by start index so we can patch gaps easily
    sections = sorted(sections, key=lambda s: s[unit_key_start])

    repaired: list[dict] = []
    expected_start = 0

    for sec in sections:
        start = sec[unit_key_start]
        end = sec[unit_key_end]

        if start > expected_start:
            logger.warning(
                "[AGENTIC] Gap detected in '%s': segments %d–%d unassigned, "
                "extending previous section to cover them.",
                name_space,
                expected_start,
                start - 1,
            )
            if repaired:
                repaired[-1] = dict(repaired[-1])
                repaired[-1][unit_key_end] = start - 1
            else:
                sec = dict(sec)
                sec[unit_key_start] = 0
                start = 0

        if end < start:
            logger.warning(
                "[AGENTIC] Section '%s' has end < start, skipping.", sec.get("title", "")
            )
            continue

        if end >= total_units:
            end = total_units - 1
            sec = dict(sec)
            sec[unit_key_end] = end

        repaired.append(dict(sec) | {unit_key_start: start, unit_key_end: end})
        expected_start = end + 1

    # Extend last section to cover any trailing units
    if repaired and repaired[-1][unit_key_end] < total_units - 1:
        logger.warning(
            "[AGENTIC] Last section in '%s' ends at %d but transcript has %d units; extending.",
            name_space,
            repaired[-1][unit_key_end],
            total_units,
        )
        repaired[-1] = dict(repaired[-1])
        repaired[-1][unit_key_end] = total_units - 1

    return repaired


# ---------------------------------------------------------------------------
# SRT implementation
# ---------------------------------------------------------------------------

def build_chunks_agentic_srt(
    sub_entries: list[dict],
    name_space: str,
    build_chunk_lines_fn: Callable,
    compute_text_hash_fn: Callable,
    model: LLMModel = LLMModel.GEMINI_FLASH,
) -> list[Chunk]:
    """
    AGENTIC chunking for SRT files.

    Merges subtitle lines into 6-line segments (same as other SRT strategies),
    formats the full list with timestamps, and asks the LLM to identify which
    consecutive segments form each logical section.  Each section becomes one
    independent Chunk (its own UUID, its own embedding).

    :param sub_entries: Flat list of subtitle entry dicts (from _flatten_subs).
    :param name_space: Source file name used as the chunk namespace.
    :param build_chunk_lines_fn: Shared helper that merges N sub-entries into
        (text, (start, end)) timestamp segments.
    :param compute_text_hash_fn: Shared SHA-256 helper.
    :param model: LLM to use for section detection.
    :return: List of Chunk objects, one per LLM-identified section.
    """
    if not sub_entries:
        return []

    # Build the merged 6-line segments that carry timestamps
    all_segments: list[tuple[str, tuple[Optional[str], Optional[str]]]] = (
        build_chunk_lines_fn(sub_entries, lines_per_segment=SRT_LINES_PER_SEGMENT)
    )
    total_segments = len(all_segments)

    # Format for the prompt: [idx] (start → end) text
    formatted = "\n".join(
        f"[{idx}] ({start} \u2192 {end}) {text}"
        for idx, (text, (start, end)) in enumerate(all_segments)
    )

    prompt = _AGENTIC_SRT_PROMPT.format(
        lines_per_segment=SRT_LINES_PER_SEGMENT,
        segments=formatted,
        last_segment_idx=total_segments - 1,
        min_segs=AGENTIC_MIN_SECTION_SEGMENTS,
        max_segs=AGENTIC_MAX_SECTION_SEGMENTS,
    )

    logger.info(
        "[AGENTIC] Sending %d segments to LLM for '%s' (model: %s)",
        total_segments,
        name_space,
        model.value,
    )

    raw_response = _run_async(_call_llm(prompt, model, name_space))

    try:
        data = json.loads(raw_response)
        raw_sections = data["sections"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error(
            "[AGENTIC] Failed to parse LLM response for '%s': %s\nRaw (first 500 chars): %s",
            name_space,
            exc,
            raw_response[:500],
        )
        raise ValueError(
            f"AGENTIC chunking failed: invalid LLM response for '{name_space}'"
        ) from exc

    sections = _validate_and_repair_sections(
        raw_sections,
        total_units=total_segments,
        unit_key_start="start_segment",
        unit_key_end="end_segment",
        name_space=name_space,
    )

    logger.info(
        "[AGENTIC] Sections identified: %d | doc='%s' | titles: %s",
        len(sections),
        name_space,
        [s.get("title", "?") for s in sections],
    )

    chunks: list[Chunk] = []
    for section in sections:
        start_seg = section["start_segment"]
        end_seg = section["end_segment"]

        # Map segment indices back to sub_entries indices
        first_entry_idx = start_seg * SRT_LINES_PER_SEGMENT
        last_entry_idx = min((end_seg + 1) * SRT_LINES_PER_SEGMENT, len(sub_entries))
        section_entries = sub_entries[first_entry_idx:last_entry_idx]

        if not section_entries:
            logger.warning("[AGENTIC] Empty section [%d–%d], skipping.", start_seg, end_seg)
            continue

        text_to_embed = " ".join(entry["text"] for entry in section_entries)
        section_segments = all_segments[start_seg : end_seg + 1]
        time_start = section_segments[0][1][0]
        time_end = section_segments[-1][1][1]
        embed_tokens = sum(entry["tokens"] for entry in section_entries)
        text_hash = compute_text_hash_fn(text_to_embed)

        chunks.append(
            Chunk(
                full_text_id=uuid.uuid4(),
                time_start=time_start,
                time_end=time_end,
                full_text=section_segments,
                text_to_embed=text_to_embed,
                chunk_size=embed_tokens,
                embed_size=embed_tokens,
                name_space=name_space,
                text_hash=text_hash,
            )
        )

    logger.info("[AGENTIC] Created %d chunks from '%s'", len(chunks), name_space)
    return chunks


# ---------------------------------------------------------------------------
# TXT implementation
# ---------------------------------------------------------------------------

def build_chunks_agentic_txt(
    tokens: list[int],
    encoder,
    name_space: str,
    compute_text_hash_fn: Callable,
    model: LLMModel = LLMModel.GEMINI_FLASH,
    paragraph_token_size: int = 200,
) -> list[Chunk]:
    """
    AGENTIC chunking for plain-text files.

    Splits the token stream into fixed-size paragraphs, sends the full list to
    the LLM, and asks it to group consecutive paragraphs into logical sections.
    Each section becomes one independent Chunk.

    :param tokens: Full token list from tiktoken.
    :param encoder: Tiktoken encoder instance.
    :param name_space: Source file name.
    :param compute_text_hash_fn: Shared SHA-256 helper.
    :param model: LLM to use for section detection.
    :param paragraph_token_size: Token budget per paragraph unit sent to the LLM.
    :return: List of Chunk objects, one per LLM-identified section.
    """
    if not tokens:
        return []

    # Split token stream into paragraph-sized units
    paragraphs: list[str] = []
    for i in range(0, len(tokens), paragraph_token_size):
        para_tokens = tokens[i : i + paragraph_token_size]
        paragraphs.append(encoder.decode(para_tokens))

    total_paragraphs = len(paragraphs)

    formatted = "\n\n".join(
        f"[{idx}] {text}" for idx, text in enumerate(paragraphs)
    )

    prompt = _AGENTIC_TXT_PROMPT.format(
        paragraphs=formatted,
        last_paragraph_idx=total_paragraphs - 1,
        min_segs=AGENTIC_MIN_SECTION_SEGMENTS,
        max_segs=AGENTIC_MAX_SECTION_SEGMENTS,
    )

    logger.info(
        "[AGENTIC] Sending %d paragraphs to LLM for '%s' (model: %s)",
        total_paragraphs,
        name_space,
        model.value,
    )

    # Reuse the SRT response format — field names still work because we use
    # start_segment/end_segment for paragraphs too (mapped via _validate_and_repair_sections)
    txt_response_format = {
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

    async def _call_llm_txt() -> str:
        from rag.app.services.llm import get_llm_response
        prompt_chars = len(prompt)
        logger.info(
            "[AGENTIC] LLM API call START  | doc='%s' | model=%s | prompt_chars=%d",
            name_space,
            model.value,
            prompt_chars,
        )
        t0 = time.perf_counter()
        response = await get_llm_response(
            prompt=prompt,
            model=model,
            response_format=txt_response_format,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "[AGENTIC] LLM API call END    | doc='%s' | model=%s | elapsed=%dms | response_chars=%d",
            name_space,
            model.value,
            elapsed_ms,
            len(response),
        )
        return response

    raw_response = _run_async(_call_llm_txt())

    try:
        data = json.loads(raw_response)
        raw_sections = data["sections"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error(
            "[AGENTIC] Failed to parse LLM response for '%s': %s\nRaw (first 500 chars): %s",
            name_space,
            exc,
            raw_response[:500],
        )
        raise ValueError(
            f"AGENTIC chunking failed: invalid LLM response for '{name_space}'"
        ) from exc

    sections = _validate_and_repair_sections(
        raw_sections,
        total_units=total_paragraphs,
        unit_key_start="start_paragraph",
        unit_key_end="end_paragraph",
        name_space=name_space,
    )

    logger.info(
        "[AGENTIC] Sections identified: %d | doc='%s' | titles: %s",
        len(sections),
        name_space,
        [s.get("title", "?") for s in sections],
    )

    # Paragraph index → token range mapping
    para_token_starts = list(range(0, len(tokens), paragraph_token_size))

    chunks: list[Chunk] = []
    for section in sections:
        start_para = section["start_paragraph"]
        end_para = section["end_paragraph"]

        # Reconstruct text from token ranges
        token_start = para_token_starts[start_para]
        if end_para + 1 < len(para_token_starts):
            token_end = para_token_starts[end_para + 1]
        else:
            token_end = len(tokens)

        section_tokens = tokens[token_start:token_end]
        if not section_tokens:
            logger.warning("[AGENTIC] Empty section [%d–%d], skipping.", start_para, end_para)
            continue

        text_to_embed = encoder.decode(section_tokens)
        token_count = len(section_tokens)
        text_hash = compute_text_hash_fn(text_to_embed)

        chunks.append(
            Chunk(
                full_text_id=uuid.uuid4(),
                time_start=None,
                time_end=None,
                full_text=text_to_embed,
                text_to_embed=text_to_embed,
                chunk_size=token_count,
                embed_size=token_count,
                name_space=name_space,
                text_hash=text_hash,
            )
        )

    logger.info("[AGENTIC] Created %d chunks from '%s'", len(chunks), name_space)
    return chunks
