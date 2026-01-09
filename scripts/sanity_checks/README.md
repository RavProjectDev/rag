# Sanity Check Module

Comprehensive database consistency checker for the RAG system. Verifies that all expected chunks are present in Pinecone by comparing text-based chunk IDs.

## Features

- ✅ **Text-based comparison** - No embedding generation needed
- ✅ **Infinite retries** - Never fails on transient errors
- ✅ **Exact missing chunks** - Know precisely what text is missing
- ✅ **Full text in report** - See the actual content that needs re-embedding
- ✅ **Timestamps** - Know where in video each missing chunk is from
- ✅ **Extra chunk detection** - Find duplicates or errors
- ✅ **Per-configuration reports** - Separate results for each index/namespace
- ✅ **Modular architecture** - Clean separation of concerns

## Module Structure

```
sanity_checks/
├── __init__.py           # Module exports
├── __main__.py           # Entry point
├── config.py             # Constants and configuration
├── fetch.py              # Manifest and transcript fetching
├── chunk_generator.py    # Chunk generation with retries
├── pinecone_query.py     # Pinecone query utilities
├── comparator.py         # Chunk comparison logic
└── reporter.py           # Report generation
```

## How It Works

### 1. Expected State Calculation
- Fetches manifest to get all documents that should exist
- For each document, runs preprocessing to generate chunks
- Creates chunk IDs using: `{doc_id}_{chunk_#}_{md5_hash_of_text}`

### 2. Actual State Query
- Connects to Pinecone for each index/namespace combination
- Queries vectors by `sanity_id` filter
- Extracts actual chunk IDs

### 3. Comparison
- Compares expected vs actual chunk ID sets
- Missing chunks = Expected - Actual
- Extra chunks = Actual - Expected

### 4. Report Generation
- Detailed JSON report with all inconsistencies
- Console summary with statistics
- Missing chunk details with full text

## Usage

### Basic Usage

```bash
# Check all configurations
python -m rag.scripts.sanity_checks

# Check specific embedding model
EMBEDDING_MODELS=gemini python -m rag.scripts.sanity_checks

# Check specific chunking strategy
CHUNKING_STRATEGIES=semantic python -m rag.scripts.sanity_checks

# Combine both
EMBEDDING_MODELS=gemini,openai CHUNKING_STRATEGIES=semantic python -m rag.scripts.sanity_checks
```

### Advanced Options

```bash
# Custom output file
OUTPUT_FILE=check_2026_01_08.json python -m rag.scripts.sanity_checks

# Limit retry attempts (for testing)
MAX_ATTEMPTS=5 python -m rag.scripts.sanity_checks

# With all Pinecone settings
PINECONE_API_KEY=xxx \
PINECONE_CLOUD=aws \
PINECONE_REGION=us-east-1 \
python -m rag.scripts.sanity_checks
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODELS` | Comma-separated list: gemini, openai, cohere | All models |
| `CHUNKING_STRATEGIES` | Comma-separated list: fixed-size, semantic, sliding-window | All strategies |
| `OUTPUT_FILE` | Path to output JSON report | sanity_check_report.json |
| `MAX_ATTEMPTS` | Maximum retry attempts (None = infinite) | None (infinite) |
| `PINECONE_API_KEY` | Pinecone API key | **Required** |
| `PINECONE_CLOUD` | Cloud provider | aws |
| `PINECONE_REGION` | Cloud region | us-east-1 |

## Output Report Structure

```json
{
  "timestamp": "2026-01-08T10:30:00",
  "total_documents": 150,
  "configurations_checked": 9,
  "summary": {
    "total_issues": 23,
    "ok_documents": 127,
    "missing_documents": 5,
    "incomplete_documents": 15,
    "excess_vectors": 3,
    "errors": 0
  },
  "configurations": [
    {
      "embedding_config": "gemini",
      "chunking_strategy": "semantic",
      "index_name": "gemini",
      "namespace": "semantic",
      "pinecone_stats": {
        "vector_count": 7450
      },
      "documents": [
        {
          "document_id": "doc-123",
          "document_title": "Lecture Title",
          "status": "INCOMPLETE",
          "expected_chunk_count": 50,
          "actual_chunk_count": 48,
          "missing_chunks": {
            "count": 2,
            "ids": ["doc-123_chunk-5_a1b2", "doc-123_chunk-12_c3d4"],
            "details": [
              {
                "chunk_id": "doc-123_chunk-5_a1b2",
                "text_preview": "Full text of missing chunk...",
                "time_start": 125.5,
                "time_end": 140.2
              }
            ]
          },
          "extra_chunks": {
            "count": 0,
            "ids": []
          }
        }
      ]
    }
  ]
}
```

## Status Codes

| Status | Description |
|--------|-------------|
| `OK` | All chunks present, no issues |
| `MISSING_ALL` | Document completely missing (0 chunks) |
| `INCOMPLETE` | Some chunks missing |
| `MISMATCH` | Extra chunks present that shouldn't be there |
| `ERROR` | Failed to check (even after retries) |

## Integration with Upload Script

The sanity check uses the **same chunk ID generation logic** as the upload script:

```python
# Both use this formula:
chunk_id = f"{sanity_id}_{full_text_id}_{md5_hash[:8]}"
```

This ensures perfect consistency between what's expected and what's uploaded.

## Performance

- **No embedding generation** - Only text preprocessing
- **Fast** - No API rate limits for embeddings
- **Cheap** - No embedding API costs
- **Reliable** - Only local processing needs to succeed

## Error Handling

- **Infinite retries** for transcript fetching (with exponential backoff)
- **Infinite retries** for chunk generation
- **Graceful degradation** - Continues checking other documents if one fails
- **Detailed error logging** - Know exactly what went wrong

## Example Output

```
[SANITY CHECK] FINAL SUMMARY
================================================================================
Total Documents: 150
Configurations Checked: 9
Total Issues: 23
  - OK Documents: 127
  - Missing Documents: 5
  - Incomplete Documents: 15
  - Excess Vectors: 3
  - Errors: 0
================================================================================

[CONFIG] gemini / semantic
  Index: gemini
  Namespace: semantic
  Pinecone Vectors: 7450
  Documents Checked: 150
  Summary: {'ok': 145, 'missing_all': 2, 'incomplete': 3, 'mismatch': 0, 'errors': 0}
```

