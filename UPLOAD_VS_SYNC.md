# Upload vs Sync: Understanding the Difference

## Overview

The system has two distinct approaches for getting documents into the vector database:

1. **Upload** (`scripts/upload_manifest.py`) - Simple, brute-force upload
2. **Sync** (`app/services/sync_service/sync_db.py`) - Smart, hash-based synchronization

## Upload Script (`upload_manifest.py`)

### What It Does

**Simple Upload**: Fetches all documents from manifest and uploads them.

```
1. Fetch manifest (all documents)
2. For each document:
   - Fetch transcript
   - Chunk it
   - Embed ALL chunks
   - Insert ALL chunks
3. Done
```

### Characteristics

- ❌ **No deduplication**: Embeds everything
- ❌ **No cleanup**: Doesn't remove old chunks
- ❌ **No comparison**: Doesn't check what exists
- ✅ **Simple**: Straightforward, easy to understand
- ✅ **Predictable**: Always processes everything
- ✅ **Fresh start**: Good for initial setup or complete rebuild

### When to Use

- **Initial setup**: First time populating the database
- **Complete rebuild**: Want to start from scratch
- **Testing**: Want to process specific documents
- **Single document**: Uploading one or few documents

### Command

```bash
# Upload all documents from manifest
python scripts/upload_manifest.py --embedding-model openai --chunking-strategy divided
```

### Code Flow

```python
# upload_manifest.py
manifest = await fetch_manifest()
documents = [SanityData(id=doc_id, **content) for doc_id, content in manifest.items()]

await upload_documents(
    documents=documents,
    connection=conn,
    embedding_configuration=embedding_model,
    chunking_strategy=chunking_strategy,
)
```

---

## Sync Service (`sync_db.py`)

### What It Does

**Smart Sync**: Compares chunks by hash and only processes what changed.

```
1. Fetch manifest (all documents)
2. Get existing chunk hashes from DB
3. For each document:
   - Fetch transcript
   - Chunk it
   - For each chunk:
     • Check if hash exists
     • If NO → embed and insert
     • If YES → skip
4. Remove chunks in DB but not in manifest
```

### Characteristics

- ✅ **Deduplication**: Skips existing chunks
- ✅ **Cleanup**: Removes outdated chunks
- ✅ **Efficient**: Minimal API calls (only for new chunks)
- ✅ **Smart**: Hash-based comparison
- ✅ **Cost-effective**: Saves embedding API costs
- ⚠️ **Complex**: More moving parts

### When to Use

- **Regular updates**: Daily/hourly sync from CMS
- **Content changes**: Documents are frequently updated
- **Large datasets**: Minimize API costs on big collections
- **Production**: Ongoing maintenance and updates

### Command

```bash
# Run the sync service directly
python app/services/sync_service/sync_db.py

# Or import and use programmatically
from rag.app.services.sync_service.sync_db import run
await run(connection, embedding_config, chunking_strategy, chunks_collection)
```

### Code Flow

```python
# sync_db.py
manifest = await fetch_manifest()
existing_hashes = await get_existing_chunk_hashes(chunks_collection)
new_hashes = set()

for doc in manifest:
    chunks = process_transcript_contents(...)
    for chunk in chunks:
        new_hashes.add(chunk.text_hash)
        if chunk.text_hash not in existing_hashes:
            # Only embed and insert NEW chunks
            await embed_and_insert(chunk)

# Remove chunks that are in DB but not in new manifest
await remove_redundant_chunks(existing_hashes - new_hashes)
```

---

## Comparison

| Feature | Upload Script | Sync Service |
|---------|---------------|--------------|
| **Purpose** | Upload all documents | Keep DB in sync with manifest |
| **Deduplication** | ❌ No | ✅ Yes (hash-based) |
| **Cleanup** | ❌ No | ✅ Yes (removes outdated) |
| **API Efficiency** | ❌ Low (processes all) | ✅ High (only new chunks) |
| **Complexity** | Simple | Smart |
| **Use Case** | Initial setup, rebuild | Regular updates, production |
| **Dependencies** | Manifest API | Manifest API + chunks collection |

## Example Scenario

**Manifest**: 100 documents, 10,000 chunks  
**Database**: 8,000 existing chunks (80% overlap)

### Upload Script Would:
- Process: **10,000 chunks**
- Embed: **10,000 chunks**
- API calls: **10,000**
- Time: ~30 minutes
- Cost: $$$

### Sync Service Would:
- Process: **10,000 chunks**
- Embed: **2,000 NEW chunks** (20%)
- Skip: **8,000 existing chunks** (80%)
- Remove: **0 redundant chunks** (all still valid)
- API calls: **2,000**
- Time: ~6 minutes
- Cost: $

**Savings**: 80% reduction in API calls and costs!

## When Content Changes

**Scenario**: One document updated (100 chunks changed)

### Upload Script:
```
- Processes: 10,000 chunks (entire manifest)
- Embeds: 10,000 chunks
- Wastes: 9,900 redundant embeddings
```

### Sync Service:
```
- Processes: 10,000 chunks (entire manifest)  
- Embeds: 100 NEW chunks (only changed ones)
- Skips: 9,900 existing chunks
- Removes: 100 old versions of changed chunks
```

**Savings**: 99% reduction in embeddings!

## Recommended Workflow

### Initial Setup
```bash
# 1. First time: use upload script
python scripts/upload_manifest.py --embedding-model openai --chunking-strategy divided
```

### Production/Regular Updates
```bash
# 2. Regular updates: use sync service
python app/services/sync_service/sync_db.py
```

### Development/Testing
```bash
# Use upload script for quick testing with small datasets
python scripts/upload_manifest.py --embedding-model mock --chunking-strategy fixed_size
```

## Implementation Details

### Upload Script
- Location: `scripts/upload_manifest.py`
- Imports: `upload_documents` from `data_upload_service`
- Function: `upload_from_manifest()`
- Dependencies: Manifest API, database connection

### Sync Service
- Location: `app/services/sync_service/sync_db.py`
- Imports: Multiple from `data_upload_service`, `embedding`
- Function: `run()`
- Dependencies: Manifest API, database connection, chunks collection

## Summary

**Upload** = Stupid, simple, "upload everything"  
**Sync** = Smart, efficient, "only what's needed"

Use **upload** for initial setup or complete rebuilds.  
Use **sync** for regular updates and production maintenance.

The two scripts serve different purposes and should be used accordingly! 🎯

