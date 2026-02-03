# Sync Script with API Configuration

## Overview

A new sync script has been created that fetches its configuration from the API's `/info` endpoint. This ensures the sync always uses the same embedding model and chunking strategy as the running API service.

## What Was Created

### 1. API Endpoint: `/api/v1/info/`

**File**: `rag/app/api/v1/info.py`

Returns the current configuration:
```json
{
  "embedding_model": "openai",
  "chunking_strategy": "divided",
  "database_backend": "pinecone",
  "environment": "PRD"
}
```

### 2. Sync Script: `scripts/sync_manifest.py`

**File**: `rag/scripts/sync_manifest.py`

Features:
- Fetches configuration from API `/info` endpoint
- Compares existing chunks vs new chunks (by SHA-256 hash)
- Only embeds NEW chunks (skips duplicates)
- Removes redundant chunks
- **Fails if API is unreachable** (no defaults)

## Testing

### Step 1: Test the API Endpoint

If your API is running locally:
```bash
curl http://localhost:8000/api/v1/info/
```

If testing against production:
```bash
curl https://api.theravlegacy.org/api/v1/info/
```

Expected response:
```json
{
  "embedding_model": "openai",
  "chunking_strategy": "divided",
  "database_backend": "mongo",
  "environment": "STG"
}
```

### Step 2: Run the Sync Script

**Against production API**:
```bash
cd /Users/dothanbardichev/Desktop/rav-project/AI
python -m rag.scripts.sync_manifest
```

**Against staging API**:
```bash
python -m rag.scripts.sync_manifest --api-url https://stg-api.theravlegacy.org
```

**Against local API**:
```bash
python -m rag.scripts.sync_manifest --api-url http://localhost:8000
```

### Step 3: Verify Sync Behavior

The script will:
1. Fetch config from API (fails if endpoint unreachable)
2. Connect to database (using your `.env` settings)
3. Fetch manifest
4. For each document:
   - Get existing chunk hashes from MongoDB
   - Chunk the document
   - Compare hashes
   - Insert only new chunks
   - Remove redundant chunks
5. Display summary statistics

Example output:
```
==============================================================
SYNC SCRIPT - Hash-based synchronization
==============================================================
[CONFIG] Using API URL: https://api.theravlegacy.org
[CONFIG] Fetching configuration from https://api.theravlegacy.org/api/v1/info/
[CONFIG] ✓ Embedding model: openai
[CONFIG] ✓ Chunking strategy: divided
[CONFIG] ✓ Database backend: pinecone
[CONFIG] ✓ Environment: PRD
==============================================================
CONFIGURATION
==============================================================
Database backend: pinecone
Embedding model: openai
Chunking strategy: divided
==============================================================
[SYNC] Fetching manifest data...
[SYNC] Fetched 150 documents from manifest
==============================================================
[SYNC] Document 1/150: Understanding Torah
==============================================================
[SYNC] Found 45 existing chunk hashes for slug='understanding-torah'
[SYNC] Fetching transcript for 'Understanding Torah'...
[SYNC] Chunking document with strategy: divided
[SYNC] Created 45 chunks from 'Understanding Torah'
[SYNC] All chunks already exist for 'Understanding Torah'
[SYNC] No redundant chunks to remove
...
==============================================================
SYNC COMPLETE - Summary
==============================================================
Documents processed: 150/150
Total chunks: 5000
New chunks inserted: 120
Existing chunks reused: 4850
Redundant chunks removed: 30
Efficiency: 97.0% chunks reused (avoided re-embedding)
==============================================================
```

## Configuration

### Changing API URL

To sync against different environments, edit the constant in `sync_manifest.py`:

```python
# For staging
BASE_API_URL = "https://stg-api.theravlegacy.org"

# For local development
BASE_API_URL = "http://localhost:8000"

# For production
BASE_API_URL = "https://api.theravlegacy.org"
```

Or use the command-line argument:
```bash
python -m rag.scripts.sync_manifest --api-url https://stg-api.theravlegacy.org
```

### Environment Variables Required

The script uses your `.env` file for database credentials and API keys:

```bash
# Database connection
MONGODB_URI=mongodb://...
MONGODB_DB_NAME=rav_dev
MONGODB_VECTOR_COLLECTION=your_collection
COLLECTION_INDEX=vector_index
VECTOR_PATH=vector

# API Keys (based on which embedding model the API is using)
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
COHERE_API_KEY=your_key

# Pinecone (if using Pinecone backend)
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=optional_override
PINECONE_NAMESPACE=optional_override
```

## Error Handling

### API Endpoint Not Found

```
[CONFIG] Failed to fetch configuration from API. Status: 404
[CONFIG] Cannot proceed without valid configuration from API.
```

**Solution**: Ensure your API is running and the `/api/v1/info/` endpoint exists.

### API Connection Failed

```
[CONFIG] Cannot connect to API at https://api.theravlegacy.org/api/v1/info/
[CONFIG] Please check the BASE_API_URL and ensure the API is running.
```

**Solution**: Check the API URL and ensure the service is accessible.

### Invalid Configuration

```
[CONFIG] Invalid configuration values from API: 'invalid_model' is not a valid EmbeddingConfiguration
```

**Solution**: The API returned an invalid embedding model or chunking strategy. Check the API's `.env` configuration.

## Comparison: Upload vs Sync

| Feature | `upload_manifest.py` | `sync_manifest.py` |
|---------|---------------------|-------------------|
| Configuration | CLI args or `.env` | Fetched from API |
| Deduplication | ❌ No | ✅ Yes (hash-based) |
| Removes old chunks | ❌ No | ✅ Yes |
| Use case | Initial load, full rebuild | Regular updates, production |
| Efficiency | Embeds everything | Only embeds new chunks |

## Best Practices

1. **Production sync**: Use `sync_manifest.py` to keep your database in sync with the manifest
2. **Initial load**: Use `upload_manifest.py` for first-time population
3. **Testing**: Use `sync_manifest.py --api-url http://localhost:8000` when testing locally
4. **Staging**: Edit `BASE_API_URL` to point to staging when testing new configurations

## Troubleshooting

**Q: The script hangs at "Fetching configuration"**
- Check if the API URL is correct and accessible
- Try accessing the `/info` endpoint manually with `curl`

**Q: Script says "Cannot proceed without valid configuration"**
- The API's `/info` endpoint returned an error
- Check that the API is running with valid configuration
- Verify the API's `.env` has valid `EMBEDDING_CONFIGURATION` and `CHUNKING_STRATEGY`

**Q: Chunks aren't being deduplicated**
- Ensure the `chunks` MongoDB collection exists
- Check that previous uploads/syncs used the same embedding model and chunking strategy

**Q: Want to force re-embedding everything**
- Use `upload_manifest.py` instead, which doesn't check for existing chunks
- Or manually clear the `chunks` collection and vector collection
