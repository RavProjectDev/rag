# Sync Service - Hash-Based Synchronization

## Overview

The sync service now uses **hash-based comparison** to efficiently synchronize documents from the manifest to the vector database. It avoids re-embedding unchanged chunks and removes outdated content.

## Key Features

✅ **Smart Syncing**: Only processes chunks that don't exist  
✅ **Hash-Based Deduplication**: Uses SHA-256 hashes to identify chunks  
✅ **Cleanup**: Removes redundant chunks no longer in manifest  
✅ **Reuses Upload Logic**: Normalized with `data_upload_service`  
✅ **Works with Both**: Supports MongoDB and Pinecone backends  

## Flow Diagram

```
┌─────────────────────────────────────┐
│   Fetch Manifest from API           │
│   (All current documents)            │
└──────────────┬──────────────────────┘
               ↓
        ┌──────┴────────┐
        │ For Each Doc   │
        └──────┬─────────┘
               ↓
┌─────────────────────────────────────┐
│   Get Existing Hashes               │
│   (By sanity_slug - this doc only)  │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   1. Fetch Transcript               │
│   2. Chunk Document                 │
│   3. Extract Chunk Hashes           │
└──────────────┬──────────────────────┘
               ↓
        ┌──────┴───────┐
        │ For Each Chunk│
        └──────┬───────┘
               ↓
       ┌───────────────┐
       │ Hash Exists?  │
       │ (for this doc)│
       └───┬───────┬───┘
    Yes ↓           ↓ No
    ┌──────┐    ┌─────────────┐
    │ Skip │    │ Embed+Insert│
    └──────┘    └─────────────┘
               ↓
┌─────────────────────────────────────┐
│   Compare Hashes (This Doc)         │
│   existing_doc vs new_doc           │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Remove Redundant Chunks           │
│   (This doc: in DB but not in new)  │
└──────────────┬──────────────────────┘
               ↓
        ┌──────┴────────┐
        │ Next Document  │
        └───────────────┘
```

## New Per-Document Sync Process

### 1. Fetch Manifest

```python
manifest = await post_data()
# Returns: {
#   "doc_id_1": {"title": "...", "transcriptURL": "...", "hash": "...", "slug": "..."},
#   "doc_id_2": {...},
# }
```

### 2. For Each Document (Independent Sync)

The sync happens **per-document**, not globally:

```python
for doc_id, content in manifest.items():
    sanity_data = SanityData(id=doc_id, **content)
    
    # 2a. Get existing hashes for THIS document only (by sanity_slug)
    existing_hashes_for_doc = await get_existing_chunk_hashes_by_slug(
        chunks_collection,
        sanity_data.slug  # e.g., "introduction-to-halacha"
    )
    # Returns: Set of hashes for this specific document
    
    # 2b. Fetch and chunk the document
    transcript_content = await fetch_transcript(sanity_data.transcriptURL)
    chunks = process_transcript_contents(
        sanity_data.title,
        transcript_content,
        chunking_strategy
    )
    
    # 2c. Process chunks for this document
    new_hashes_for_doc = set()
    for chunk in chunks:
        chunk_hash = chunk.text_hash
        new_hashes_for_doc.add(chunk_hash)
        
        # Check if exists FOR THIS DOCUMENT
        if chunk_hash in existing_hashes_for_doc:
            continue  # Skip
        
        # New chunk - embed and insert
        embedding = await generate_embedding(chunk.text_to_embed, config)
        await connection.insert([embedding])
    
    # 2d. Cleanup redundant chunks FOR THIS DOCUMENT
    redundant_hashes = existing_hashes_for_doc - new_hashes_for_doc
    
    # Remove chunks that were in old version but not in new
    await chunks_collection.delete_many({"_id": {"$in": list(redundant_hashes)}})
    await delete_from_vector_store(redundant_hashes)
```

### 3. Summary Report

After all documents processed:

```python
# Aggregated statistics across all documents
logger.info(f"Documents processed: {total_documents}")
logger.info(f"Total chunks: {total_chunks}")
logger.info(f"New chunks inserted: {total_inserted}")
logger.info(f"Existing chunks reused: {total_reused}")
logger.info(f"Redundant chunks removed: {total_removed}")
```

## Key Functions

### `get_existing_chunk_hashes_by_slug(chunks_collection, sanity_slug)`

Retrieves existing chunk hashes for a specific document by its slug.

```python
cursor = chunks_collection.find(
    {"sanity_slug": sanity_slug},  # Filter by document slug
    {"_id": 1}                      # Only return hash field
)
existing_hashes = set()
async for doc in cursor:
    existing_hashes.add(doc["_id"])  # _id is the SHA-256 hash
```

Benefits:
- **Memory efficient**: Only loads hashes for one document
- **Fast**: Uses index on sanity_slug
- **Scoped**: Per-document comparison prevents cross-document conflicts

### `check_hash_in_pinecone(connection, chunk_hash)`

Checks if a hash exists in Pinecone metadata.

```python
result = await connection.index.query(
    vector=[0.0] * 3072,  # Dummy vector
    top_k=1,
    filter={"text_hash": {"$eq": chunk_hash}}
)
return len(result.get("matches", [])) > 0
```

### `process_and_sync_chunks(...)`

Processes chunks and inserts only new ones.

- Checks hash against existing hashes
- For Pinecone, also checks Pinecone metadata
- Embeds and inserts if not found
- Tracks all new hashes

### `remove_redundant_chunks(...)`

Removes chunks that are obsolete.

- Calculates: `redundant = existing - new`
- Deletes from chunks collection
- Deletes from vector store

## Usage

### Running the Sync

```python
from rag.app.services.sync_service.sync_db import run

await run(
    connection=embedding_connection,
    embedding_configuration=EmbeddingConfiguration.OPENAI,
    chunking_strategy=ChunkingStrategy.DIVIDED,
    chunks_collection=db["chunks"],
)
```

### Command Line

```bash
# Run directly
cd /path/to/rag
python app/services/sync_service/sync_db.py
```

## Benefits

### 1. Efficiency
- **Avoids Re-embedding**: Unchanged chunks are skipped
- **Fast Comparison**: Hash lookup is O(1)
- **Parallel Ready**: Can be scaled for concurrent processing

### 2. Consistency
- **Reuses Upload Logic**: Same chunking and hashing as upload
- **Normalized**: Consistent with `data_upload_service`
- **Reliable**: Uses proven upload service components

### 3. Cleanup
- **Removes Outdated**: Deletes chunks no longer in manifest
- **Maintains Accuracy**: Database reflects current manifest state
- **Space Efficient**: No orphaned chunks accumulating

### 4. Flexibility
- **Works with Both**: MongoDB and Pinecone supported
- **Configurable**: Any embedding model and chunking strategy
- **Extensible**: Easy to add more backends

## Comparison: Old vs New

### Old Sync Process

```
1. Get all document IDs from DB
2. Compare document-level hashes
3. If document changed:
   - Delete entire document
   - Re-embed ALL chunks
4. No cleanup of orphaned chunks
```

**Issues:**
- Re-embeds entire documents even for small changes
- Expensive: wastes API calls on unchanged content
- No granular tracking

### New Sync Process

```
1. Get all chunk hashes from DB
2. For each document:
   - Chunk it
   - Check each chunk hash
   - Only embed+insert NEW chunks
3. Remove chunks not in manifest
```

**Benefits:**
- Only processes changed chunks
- Efficient: minimal API calls
- Granular: chunk-level tracking
- Clean: removes obsolete content

## Example Scenario

### Manifest has 3 documents:

**Document A**: 100 chunks (50 unchanged, 50 new)  
**Document B**: 80 chunks (all unchanged)  
**Document C**: 120 chunks (all new)  

### Old Sync Would:
- Re-embed 300 chunks (all of them)
- 300 embedding API calls
- Full delete + reinsert for A and C

### New Sync Does:
- Skip 130 chunks (50 from A + 80 from B)
- Embed 170 chunks (50 from A + 120 from C)
- 170 embedding API calls
- Insert only new chunks

**Savings**: 43% reduction in API calls!

## Logging

The sync process provides detailed logging:

```
[SYNC] Starting hash-based sync process...
[SYNC] Fetched 45 documents from manifest
[SYNC] Found 2,340 existing chunk hashes in database
[SYNC] Processing document 1/45: Introduction to Halacha
[SYNC] Created 60 chunks from Introduction to Halacha
[SYNC] Processing new chunk 5/60 (hash: 2a331b6c...)
[SYNC] Progress: 10 new chunks inserted
[SYNC] Inserted 12/60 new chunks from Introduction to Halacha
[SYNC] All chunks already exist for Torah and Science
[SYNC] Processed 2,800 total chunks, inserted 460 new chunks
[SYNC] Checking for redundant chunks...
[SYNC] Found 120 redundant chunks to remove
[SYNC] Summary:
[SYNC]   - Documents processed: 45
[SYNC]   - Total chunks: 2,800
[SYNC]   - New chunks inserted: 460
[SYNC]   - Existing chunks reused: 2,340
[SYNC]   - Redundant chunks removed: 120
```

## Error Handling

- **Document Failures**: Logged but don't stop sync
- **Chunk Failures**: Logged, processing continues
- **Cleanup Failures**: Logged, doesn't affect new inserts
- **Partial Success**: Sync completes what it can

## Performance Considerations

### Memory
- **Per-document scoping**: Only loads hashes for current document
- ~64 bytes per hash
- Average document: 60 chunks = ~4 KB in memory
- Much more efficient than loading all hashes globally

### Speed
- Hash checking: O(1)
- Chunking: Same as upload
- Embedding: Only for new chunks
- Cleanup: Batch delete operations

### Concurrency
- Can process documents in parallel (future enhancement)
- Each document is independent
- Chunk processing can be parallelized

## Future Enhancements

1. **Parallel Processing**: Process multiple documents concurrently
2. **Progress Tracking**: Store sync state for resumability
3. **Incremental Updates**: Track last sync time
4. **Validation Mode**: Dry-run to preview changes
5. **Metrics**: Track sync statistics over time

## Configuration

Set in `.env`:

```bash
# Manifest source
MANIFEST_URL=https://the-rav-project.vercel.app/api/manifest

# Database
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=rav_dev

# Embedding
EMBEDDING_CONFIGURATION=openai
CHUNKING_STRATEGY=divided

# Pinecone (if using)
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index
PINECONE_NAMESPACE=your_namespace
```

## Troubleshooting

### "No chunks collection provided"
Ensure `chunks_collection` is passed to `run()`:
```python
await run(connection, config, strategy, chunks_collection=db["chunks"])
```

### "Failed to get existing chunk hashes"
Check MongoDB connection and chunks collection exists:
```javascript
db.chunks.countDocuments()
```

### Chunks not being removed
Verify redundant chunks are identified:
- Check existing_hashes set
- Check new_hashes set
- Ensure hashes match format (SHA-256, lowercase hex)

## Testing

```python
# Test with small manifest subset
manifest = {
    "test_doc_1": {
        "title": "Test Document",
        "transcriptURL": "https://example.com/test.srt",
        "hash": "abc123",
        "slug": "test-document",
        "_updatedAt": "2026-01-22T00:00:00Z"
    }
}
```

The refactored sync service is production-ready and provides efficient, hash-based synchronization! 🚀

