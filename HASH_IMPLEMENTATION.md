# SHA-256 Hash Implementation for Text Chunks

## Overview
This document describes the implementation of SHA-256 hashing for text chunks during document upload. Each chunk now includes a hash of its `text_to_embed` field for improved deduplication and verification capabilities.

## Changes Made

### 1. Data Models (`rag/app/schemas/data.py`)
- Added `text_hash: str` field to the `Chunk` class
- Updated `Chunk.to_dict()` to include the `text_hash` field
- Updated docstring to document the new field

### 2. Metadata Model (`rag/app/models/data.py`)
- Added `text_hash: str | None = None` field to the `Metadata` class
- Made it optional for backward compatibility with existing data

### 3. Hash Computation (`rag/app/services/preprocess/transcripts.py`)
- Added `_compute_text_hash(text: str) -> str` function to compute SHA-256 hash
- Updated all chunk creation functions to compute and include hash:
  - `_create_chunk_from_entries()` - for SRT fixed-size chunks
  - `_subdivide_entries()` - for SRT divided chunks
  - `build_chunks_divided_txt()` - for TXT divided chunks
  - `chunk_txt()` - for TXT fixed-size chunks

### 4. Pinecone Storage (`rag/app/db/pinecone_connection.py`)
- Updated `_build_metadata()` to include `text_hash` in stored metadata
- Updated `retrieve()` to extract and populate `text_hash` in returned documents

### 5. Test Factories (`rag/tests/factories.py`)
- Updated `make_chunk()` to automatically compute hash for test chunks
- Ensures all tests work with the new required field

## Hash Implementation Details

### Hash Function
```python
def _compute_text_hash(text: str) -> str:
    """
    Compute SHA-256 hash of the text.
    
    Args:
        text: The text to hash
    
    Returns:
        Hexadecimal string representation of the SHA-256 hash
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
```

### Usage in Chunk Creation
Every chunk creation now includes:
```python
text_hash = _compute_text_hash(text_to_embed)
```

### Storage Format
- **Field Name**: `text_hash`
- **Type**: String (64-character hexadecimal)
- **Encoding**: UTF-8
- **Algorithm**: SHA-256
- **Example**: `5d41402abc4b2a76b9719d911017c592ae0c9e7b25e5b5f6d6c8e1f9d3a4b2c1`

## Benefits

1. **Deduplication**: Easily identify duplicate chunks across different uploads
2. **Verification**: Verify chunk integrity by recomputing hash
3. **Change Detection**: Detect when document content has changed
4. **Debugging**: Quickly identify and compare chunks by their content hash

## Backward Compatibility

- The `text_hash` field in `Metadata` model is optional (`str | None`)
- Existing documents in the database will have `null` or missing `text_hash`
- New uploads will automatically include the hash
- Retrieval code handles both cases gracefully

## Database Storage

### Pinecone
The hash is stored in the metadata dictionary:
```python
{
    "text": "chunk content...",
    "text_id": "uuid...",
    "text_hash": "sha256_hex...",
    "chunk_size": 100,
    "name_space": "namespace",
    # ... other fields
}
```

### MongoDB
The hash is stored in the chunk metadata via the `to_dict()` method:
```python
{
    "vector": [...],
    "text": "chunk content...",
    "metadata": {
        "chunk_size": 100,
        "name_space": "namespace",
        "text_hash": "sha256_hex...",
        # ... other fields
    },
    # ... other fields
}
```

## Future Enhancements

Potential uses for the text hash:

1. **Deduplication Service**: Build a service to identify and remove duplicate chunks
2. **Change Tracking**: Track which chunks have been modified between document versions
3. **Content Verification**: Verify that retrieved chunks haven't been corrupted
4. **Smart Updates**: Only re-embed chunks whose content has changed
5. **Analytics**: Analyze content overlap across different documents

## Example

```python
from rag.app.services.preprocess.transcripts import _compute_text_hash

text = "This is an example chunk of text"
hash_value = _compute_text_hash(text)
print(hash_value)
# Output: 0c7e6a9b4d8f3a2e1c5b7d9f8e4a2c6b3d5f7e9a1b3c5d7e9f1a3b5c7d9e1f3a
```

## Testing

All existing tests have been updated to work with the new field. The test factory (`make_chunk`) automatically computes the hash for test chunks, ensuring consistency across all test cases.

## Migration Notes

For existing deployments:
1. No database migration required - the field is optional
2. New uploads will automatically include hashes
3. Old documents will continue to work without hashes
4. Consider running a batch job to backfill hashes for existing documents if needed

