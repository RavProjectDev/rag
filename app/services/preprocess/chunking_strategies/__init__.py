"""
Chunking strategies module.

Each strategy is implemented in its own file:
- fixed_size.py: Strategy A - Fixed word-based chunking
- semantic.py: Strategy B - Semantic similarity-based chunking
- sliding_window.py: Strategy C - Token-based sliding window with overlap
"""

from rag.app.services.preprocess.chunking_strategies.fixed_size import (
    chunk_srt_fixed,
    chunk_txt_fixed,
    preprocess_raw_transcripts_fixed,
)
from rag.app.services.preprocess.chunking_strategies.semantic import (
    chunk_srt_semantic,
    chunk_txt_semantic,
    preprocess_raw_transcripts_semantic,
)
from rag.app.services.preprocess.chunking_strategies.sliding_window import (
    chunk_srt_sliding_window,
    chunk_txt_sliding_window,
    preprocess_raw_transcripts_sliding_window,
)

__all__ = [
    "chunk_srt_fixed",
    "chunk_txt_fixed",
    "preprocess_raw_transcripts_fixed",
    "chunk_srt_semantic",
    "chunk_txt_semantic",
    "preprocess_raw_transcripts_semantic",
    "chunk_srt_sliding_window",
    "chunk_txt_sliding_window",
    "preprocess_raw_transcripts_sliding_window",
]

