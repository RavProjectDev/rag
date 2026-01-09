# Preprocessing Module - Refactored Architecture

## Overview

The preprocessing module has been refactored to separate chunking strategies into individual files, with a main router that selects the appropriate strategy based on configuration.

## Structure

```
preprocess/
├── transcripts.py                    # Main router (extracts text, routes to strategy)
├── constants.py                       # Shared constants
├── chunking_strategies/
│   ├── __init__.py                   # Exports all strategies
│   ├── fixed_size.py                 # Strategy A: Fixed word-based chunking
│   ├── semantic.py                   # Strategy B: Semantic similarity chunking
│   └── sliding_window.py             # Strategy C: Token-based sliding window
└── user_input.py                     # User query preprocessing
```

## Main Entry Point

### `preprocess_raw_transcripts()`

The main function in `transcripts.py` that:
1. Extracts text from SRT/TXT files
2. Routes to the appropriate chunking strategy based on configuration
3. Returns processed chunks

**Usage:**

```python
from rag.app.services.preprocess.transcripts import preprocess_raw_transcripts
from rag.app.schemas.data import TypeOfFormat, ChunkingStrategy

# Uses default strategy from config
chunks = await preprocess_raw_transcripts(
    raw_transcripts=[("file.txt", "content...")],
    data_format=TypeOfFormat.TXT,
)

# Explicitly specify strategy
chunks = await preprocess_raw_transcripts(
    raw_transcripts=[("file.txt", "content...")],
    data_format=TypeOfFormat.TXT,
    chunking_strategy=ChunkingStrategy.SLIDING_WINDOW,
)
```

## Configuration

The chunking strategy is configured in `rag/app/core/config.py`:

```python
class Settings(BaseSettings):
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
```

Available strategies:
- `ChunkingStrategy.FIXED_SIZE` - Strategy A (default)
- `ChunkingStrategy.SEMANTIC` - Strategy B
- `ChunkingStrategy.SLIDING_WINDOW` - Strategy C

## Strategy Details

### Strategy A: Fixed-Size (`fixed_size.py`)
- **Function**: `preprocess_raw_transcripts_fixed()`
- **Method**: Fixed word-based chunks (50 words default)
- **Sync/Async**: Synchronous
- **Use Case**: Simple, fast chunking

### Strategy B: Semantic (`semantic.py`)
- **Function**: `preprocess_raw_transcripts_semantic()`
- **Method**: Semantic similarity between sentences
- **Sync/Async**: Async (requires embeddings)
- **Use Case**: High-quality, semantically coherent chunks
- **Requires**: `embedding_configuration` parameter

### Strategy C: Sliding Window (`sliding_window.py`)
- **Function**: `preprocess_raw_transcripts_sliding_window()`
- **Method**: Fixed token size (500) with overlap (100)
- **Sync/Async**: Synchronous
- **Use Case**: Industry baseline, consistent chunk sizes

## Migration Guide

### Before (Old Code)
```python
from rag.app.services.preprocess.transcripts import preprocess_raw_transcripts

chunks = preprocess_raw_transcripts(
    raw_transcripts=[("file.txt", "content...")],
    data_format=TypeOfFormat.TXT,
)
```

### After (New Code)
```python
from rag.app.services.preprocess.transcripts import preprocess_raw_transcripts

# Same API, but now async and routes to configured strategy
chunks = await preprocess_raw_transcripts(
    raw_transcripts=[("file.txt", "content...")],
    data_format=TypeOfFormat.TXT,
)
```

**Note**: The function is now `async` because semantic chunking requires async embedding generation.

## Environment Variables

Add to your `.env` file:

```env
CHUNKING_STRATEGY=fixed_size  # or "semantic" or "sliding_window"
```

## Direct Strategy Access

You can also import and use strategies directly:

```python
from rag.app.services.preprocess.chunking_strategies import (
    preprocess_raw_transcripts_fixed,
    preprocess_raw_transcripts_semantic,
    preprocess_raw_transcripts_sliding_window,
)

# Use specific strategy directly
chunks = await preprocess_raw_transcripts_semantic(
    raw_transcripts=[("file.txt", "content...")],
    data_format=TypeOfFormat.TXT,
    embedding_configuration=EmbeddingConfiguration.BERT_SMALL,
)
```

## Benefits of Refactoring

1. **Separation of Concerns**: Each strategy is in its own file
2. **Maintainability**: Easy to modify or add new strategies
3. **Testability**: Each strategy can be tested independently
4. **Configuration-Driven**: Strategy selection via environment config
5. **Backward Compatible**: Main API remains the same (just async)

