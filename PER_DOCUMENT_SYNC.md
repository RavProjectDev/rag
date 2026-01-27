# Per-Document Sync Flow

## Overview

The sync service now uses **per-document hash comparison** instead of loading all hashes globally. This is more memory-efficient and allows for cleaner document-level synchronization.

## Key Improvement

**Before**: Load ALL chunk hashes globally (10,000+ chunks in memory)  
**After**: Load hashes per-document (60-100 chunks at a time)

## Detailed Flow

### Step 1: Fetch Manifest

```python
manifest = await post_data()
# Returns: {
#   "doc_id_1": {
#       "title": "Introduction to Halacha",
#       "slug": "introduction-to-halacha",  ← Used for filtering
#       "transcriptURL": "https://...",
#       "hash": "abc123..."
#   },
#   "doc_id_2": {...},
#   ...
# }
```

### Step 2: Process Each Document Independently

```python
for doc_id, content in manifest.items():
```

#### 2.1 Get Existing Hashes for THIS Document AND Configuration

```python
existing_hashes_for_doc = await get_existing_chunk_hashes_by_slug(
    chunks_collection,
    sanity_slug="introduction-to-halacha",
    embedding_model="openai",
    chunking_strategy="divided"
)

# MongoDB query:
db.chunks.find(
    {
        "sanity_slug": "introduction-to-halacha",
        "embedding_model": "openai",
        "chunking_strategy": "divided"
    },
    {"_id": 1}
)

# Returns: Set of hashes for this document WITH this configuration
# Example: {"hash1", "hash2", "hash3", ...} (60 hashes for openai+divided)
# Ignores: hashes for gemini+fixed_size or other combinations
```

**Why this matters**:
- Different chunking strategies produce different chunks (different hashes)
- Different embedding models should be stored separately
- Each configuration is independent and doesn't interfere with others

**Benefits**:
- Only loads hashes for the current document
- Memory efficient (60-100 hashes vs 10,000+)
- Fast query with index on `sanity_slug`

#### 2.2 Fetch and Chunk the New Document

```python
# Fetch transcript
transcript_content = await fetch_transcript(sanity_data.transcriptURL)

# Chunk using configured strategy
chunks = process_transcript_contents(
    sanity_data.title,
    transcript_content,
    chunking_strategy  # DIVIDED or FIXED_SIZE
)

# Each chunk has pre-computed text_hash (SHA-256)
```

#### 2.3 Process Each Chunk

```python
new_hashes_for_doc = set()
inserted_count = 0

for chunk in chunks:
    chunk_hash = chunk.text_hash
    new_hashes_for_doc.add(chunk_hash)
    
    # Compare against existing hashes FOR THIS DOCUMENT
    if chunk_hash in existing_hashes_for_doc:
        # Chunk exists - skip
        continue
    
    # NEW chunk - process it
    embedding = await generate_embedding(chunk.text_to_embed, config)
    vector_embedding = VectorEmbedding(
        vector=embedding.vector,
        dimension=len(embedding.vector),
        metadata=chunk,
        sanity_data=sanity_data
    )
    await connection.insert([vector_embedding])
    inserted_count += 1
```

**Result**:
- `new_hashes_for_doc`: Set of all hashes from the new version
- `inserted_count`: Number of new chunks inserted

#### 2.4 Remove Redundant Chunks for THIS Document

```python
# Find chunks that are in DB but not in new version
redundant_hashes = existing_hashes_for_doc - new_hashes_for_doc

if redundant_hashes:
    # Remove from chunks collection
    await chunks_collection.delete_many({
        "_id": {"$in": list(redundant_hashes)}
    })
    
    # Remove from vector store
    if Pinecone:
        for hash in redundant_hashes:
            await connection.index.delete(
                filter={"text_hash": {"$eq": hash}},
                namespace=namespace
            )
    elif MongoDB:
        await connection.collection.delete_many({
            "metadata.text_hash": {"$in": list(redundant_hashes)}
        })
```

**Example**:
```
Document: "Introduction to Halacha"
Old version: 60 chunks (hashes: {h1, h2, ..., h60})
New version: 62 chunks (hashes: {h1, h2, ..., h58, h61, h62})

Comparison:
- Unchanged: 58 chunks ({h1, h2, ..., h58}) → skip
- New: 2 chunks ({h61, h62}) → embed & insert
- Removed: 2 chunks ({h59, h60}) → delete from DB
```

### Step 3: Aggregated Summary

```python
# After all documents processed
logger.info("SYNC COMPLETE - Summary")
logger.info(f"Documents processed: {total_documents}")
logger.info(f"Total chunks: {total_chunks}")
logger.info(f"New chunks inserted: {total_inserted}")
logger.info(f"Existing chunks reused: {total_reused}")
logger.info(f"Redundant chunks removed: {total_removed}")
logger.info(f"Efficiency: {efficiency:.1f}% chunks reused")
```

---

## Example Output

```
============================================================
SYNC SERVICE - Hash-based per-document synchronization
============================================================
[SYNC] Fetching manifest data...
[SYNC] Fetched 15 documents from manifest

==================================================
[SYNC] Document 1/15: Introduction to Halacha
==================================================
[SYNC] Found 60 existing chunk hashes for slug 'introduction-to-halacha'
[SYNC] Fetching transcript for 'Introduction to Halacha'...
[SYNC] Chunking document with strategy: divided
[SYNC] Created 62 chunks from 'Introduction to Halacha'
[SYNC] Processing new chunk 59/62 (hash: abc123...)
[SYNC] Processing new chunk 60/62 (hash: def456...)
[SYNC] Inserted 2/62 new chunks from 'Introduction to Halacha'
[SYNC] Found 2 redundant chunks for 'introduction-to-halacha'
[SYNC] Removed 2 chunks from chunks collection for 'introduction-to-halacha'
[SYNC] Document 'Introduction to Halacha' complete:
[SYNC]   - Total chunks: 62
[SYNC]   - New: 2
[SYNC]   - Reused: 58
[SYNC]   - Removed: 2

==================================================
[SYNC] Document 2/15: Torah and Science
==================================================
[SYNC] Found 80 existing chunk hashes for slug 'torah-and-science'
[SYNC] Fetching transcript for 'Torah and Science'...
[SYNC] Chunking document with strategy: divided
[SYNC] Created 80 chunks from 'Torah and Science'
[SYNC] All chunks already exist for 'Torah and Science'
[SYNC] Document 'Torah and Science' complete:
[SYNC]   - Total chunks: 80
[SYNC]   - New: 0
[SYNC]   - Reused: 80
[SYNC]   - Removed: 0

... (continues for all documents)

============================================================
SYNC COMPLETE - Summary
============================================================
Documents processed: 15/15
Total chunks: 1,240
New chunks inserted: 85
Existing chunks reused: 1,155
Redundant chunks removed: 12
Efficiency: 93.1% chunks reused (avoided re-embedding)
============================================================
```

---

## Advantages of Per-Document Sync

### 1. Memory Efficiency

**Before (Global)**:
```
Load: 10,000 hashes → 640 KB memory
Keep in memory for entire sync
```

**After (Per-Document)**:
```
Load: 60 hashes → 4 KB memory
Process document
Clear memory
Load: 80 hashes → 5 KB memory
Process next document
...
```

**Savings**: 99% reduction in memory usage!

### 2. Cleaner Logic

**Per-document comparison**:
- Clear scope: each document is independent
- Easier to understand: "what changed in THIS document?"
- Simpler debugging: can focus on one document at a time

### 3. Better Error Handling

```python
# If document A fails, documents B-Z still process
for document in manifest:
    try:
        await sync_document(document)
    except Exception as e:
        logger.error(f"Failed to sync {document.title}: {e}")
        continue  # Move to next document
```

### 4. Accurate Cleanup

Old approach might accidentally remove chunks if hashes collide across documents.

New approach:
- Scoped by `sanity_slug`
- Only removes chunks for the specific document
- No cross-document interference

### 5. Database Performance

**Query per document**:
```javascript
db.chunks.find({"sanity_slug": "introduction-to-halacha"})
```

Uses index on `sanity_slug` → very fast, even with millions of chunks.

---

## Comparison: Global vs Per-Document

| Aspect | Global Hashes | Per-Document Hashes |
|--------|---------------|---------------------|
| **Memory** | 10,000 hashes (640 KB) | 60-100 hashes (4-6 KB) |
| **Query** | One query (all chunks) | One query per document |
| **Scope** | All documents mixed | Clean per-document scope |
| **Cleanup** | Global comparison | Scoped to document |
| **Debugging** | Complex (all docs) | Simple (one doc at a time) |
| **Error Isolation** | One failure affects all | Failures are isolated |

---

## Key Functions

### `get_existing_chunk_hashes_by_slug(chunks_collection, sanity_slug)`

```python
async def get_existing_chunk_hashes_by_slug(chunks_collection, sanity_slug: str):
    cursor = chunks_collection.find(
        {"sanity_slug": sanity_slug},
        {"_id": 1}
    )
    existing_hashes = set()
    async for doc in cursor:
        existing_hashes.add(doc["_id"])
    return existing_hashes
```

### `sync_document(sanity_data, connection, config, strategy, chunks_collection)`

Main per-document sync function:

```python
async def sync_document(...):
    # 1. Get existing hashes for this document
    existing = await get_existing_chunk_hashes_by_slug(chunks_collection, sanity_data.slug)
    
    # 2. Fetch and chunk
    transcript = await fetch_transcript(sanity_data.transcriptURL)
    chunks = process_transcript_contents(title, transcript, strategy)
    
    # 3. Process chunks
    inserted, new_hashes = await process_and_sync_chunks(chunks, ...)
    
    # 4. Cleanup for this document
    await remove_redundant_chunks_for_document(chunks_collection, slug, existing, new_hashes)
    
    # 5. Return stats
    return {
        "total_chunks": len(chunks),
        "new_chunks": inserted,
        "reused_chunks": len(chunks) - inserted,
        "removed_chunks": len(existing - new_hashes)
    }
```

### `remove_redundant_chunks_for_document(collection, slug, existing, new)`

```python
async def remove_redundant_chunks_for_document(...):
    redundant = existing - new  # Hashes in old but not in new
    
    if redundant:
        # Delete from chunks collection
        await chunks_collection.delete_many({"_id": {"$in": list(redundant)}})
        
        # Delete from vector store
        await delete_from_vector_store(redundant)
```

---

## Recommended Indexes

For optimal performance, create these indexes on the `chunks` collection:

```javascript
// Index on sanity_slug for per-document queries
db.chunks.createIndex({ "sanity_slug": 1 })

// Compound index for document queries with ordering
db.chunks.createIndex({ "sanity_slug": 1, "chunk_index": 1 })

// Index on sanity_id for document-level operations
db.chunks.createIndex({ "sanity_id": 1 })
```

---

## Usage

```bash
# Run the sync service
python app/services/sync_service/sync_db.py
```

The sync will process each document independently, showing per-document statistics and a final summary.

---

## Benefits Summary

✅ **Memory efficient**: 99% reduction in memory usage  
✅ **Cleaner logic**: Per-document scope is intuitive  
✅ **Better performance**: Uses indexed queries  
✅ **Isolated errors**: Document failures don't affect others  
✅ **Accurate cleanup**: No cross-document conflicts  
✅ **Easier debugging**: Focus on one document at a time  

The per-document sync is production-ready and optimized for large-scale operations! 🚀

