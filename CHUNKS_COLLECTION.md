# Chunks Collection Implementation

## Overview
A separate MongoDB collection called `chunks` is now maintained to track chunk metadata for deduplication, indexing, and analytics purposes. This collection stores lightweight records that map chunk hashes to their text identifiers and positions.

## Collection Schema

```javascript
{
  "_id": "sha256_hash_of_text",           // SHA-256 hash - primary key
  "text_id": "uuid_string",               // UUID grouping related chunks
  "chunk_index": 7,                       // Global sequential index
  "text": "chunk text content...",        // The actual chunk text
  "chunk_size": 200,                      // Token count
  "name_space": "document_name",          // Document identifier
  "embedding_model": "openai",            // Embedding model used
  "chunking_strategy": "divided",         // Chunking strategy used
  "sanity_id": "doc_id",                  // Sanity CMS document ID
  "sanity_slug": "doc-slug",              // Sanity CMS slug
  "sanity_title": "Document Title",       // Sanity CMS title
  "sanity_transcriptURL": "https://...", // Transcript URL
  "sanity_hash": "sanity_hash_value",    // Sanity document hash
  "time_start": "00:00:10,000",          // Start timestamp (SRT only)
  "time_end": "00:00:15,000",            // End timestamp (SRT only)
  "sanity_updated_at": "2026-01-22..."   // Sanity update timestamp
}
```

### Field Descriptions

#### Required Fields

- **`_id`** (string): SHA-256 hash of the chunk's `text_to_embed` field. Used as the primary key for fast deduplication lookups.
- **`text_id`** (string): The `full_text_id` UUID that groups related chunks together (e.g., subdivisions from the same main chunk in DIVIDED strategy).
- **`chunk_index`** (integer): Zero-based global sequential index across all chunks in the database. Each new chunk gets the next available index (0, 1, 2, 3...).
- **`text`** (string): The full chunk text. For SRT files with timestamps, this is a JSON string containing structured data.
- **`chunk_size`** (integer): Token count of the chunk.
- **`name_space`** (string): Document identifier/namespace (usually the document filename or title).
- **`embedding_model`** (string): The embedding model used for this chunk (e.g., "openai", "gemini-embedding-001", "cohere").
- **`chunking_strategy`** (string): The chunking strategy used (e.g., "fixed_size", "divided").
- **`sanity_id`** (string): Unique identifier from Sanity CMS.
- **`sanity_slug`** (string): URL-friendly slug from Sanity CMS.
- **`sanity_title`** (string): Human-readable document title.
- **`sanity_transcriptURL`** (string): URL to the original transcript file.
- **`sanity_hash`** (string): Hash of the document in Sanity CMS.

#### Optional Fields

- **`time_start`** (string): Start timestamp for SRT chunks (format: "HH:MM:SS,mmm").
- **`time_end`** (string): End timestamp for SRT chunks (format: "HH:MM:SS,mmm").
- **`sanity_updated_at`** (string): ISO timestamp of when the document was last updated in Sanity CMS.

## Use Cases

### 1. Deduplication
Quickly identify duplicate chunks across different documents:
```javascript
db.chunks.findOne({ "_id": "hash_value" })
```

### 2. Chronological Order
Retrieve all chunks in the order they were added:
```javascript
db.chunks.find().sort({ "chunk_index": 1 })
```

Or get chunks for a specific text_id:
```javascript
db.chunks.find({ "text_id": "uuid" }).sort({ "chunk_index": 1 })
```

### 3. Analytics
Count unique chunks, identify most common chunks, etc.:
```javascript
// Count total unique chunks
db.chunks.countDocuments()

// Find duplicate chunks (same hash, different text_ids)
db.chunks.aggregate([
  { $group: { _id: "$_id", count: { $sum: 1 }, text_ids: { $addToSet: "$text_id" } } },
  { $match: { count: { $gt: 1 } } }
])
```

### 4. Chunk Lookup
Given a hash, find all instances:
```javascript
db.chunks.find({ "_id": "hash_value" })
```

## Implementation Details

### Insertion Logic

When embeddings are inserted into the main vector collection, the `MongoEmbeddingStore` automatically tracks them in the chunks collection:

1. **Grouping**: Embeddings are grouped by their `text_id` (from `full_text_id`)
2. **Index Computation**: For each group, the system:
   - Queries for the maximum existing `chunk_index` for that `text_id`
   - Assigns sequential indices starting from `max_index + 1` (or 0 if none exist)
3. **Insertion**: Chunk records are inserted with `ordered=False` to skip duplicates gracefully

### Code Location

The implementation is in `rag/app/db/mongodb_connection.py`:
- `MongoEmbeddingStore.__init__()`: Accepts optional `chunks_collection` parameter
- `MongoEmbeddingStore.insert()`: Calls `_track_chunks()` after successful insertion
- `MongoEmbeddingStore._track_chunks()`: Handles chunk tracking logic

### Error Handling

- Duplicate hash insertions (same `_id`) fail silently - the original chunk record is preserved
- Chunk tracking failures don't block the main insertion operation
- All errors are logged but not raised to ensure system reliability

## Configuration

The chunks collection is automatically created and used when MongoDB is the backend. No additional configuration is required.

### Locations Updated

All MongoDB connection points now include the chunks collection:
- `app/main.py` - Main application startup
- `scripts/upload_manifest.py` - Document upload script
- `app/services/sync_service/sync_db.py` - Sync service
- `app/services/form/get_all_form_data.py` - Form data service
- `app/api/v1/form.py` - Form API endpoint

## Database Indexes

### Recommended Indexes

For optimal performance, create these indexes:

```javascript
// Primary key (automatically created)
db.chunks.createIndex({ "_id": 1 })

// Find chunks by text_id
db.chunks.createIndex({ "text_id": 1 })

// Find chunks by text_id and order
db.chunks.createIndex({ "text_id": 1, "chunk_index": 1 })
```

### Index Creation

These can be created via MongoDB shell or using a migration script:

```javascript
use rav_dev  // or your database name
db.chunks.createIndex({ "text_id": 1 })
db.chunks.createIndex({ "text_id": 1, "chunk_index": 1 })
```

## Example Queries

### Find All Chunks for a Document
```javascript
db.chunks.find({ "text_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479" })
  .sort({ "chunk_index": 1 })
```

### Check if a Chunk Hash Exists
```javascript
db.chunks.findOne({ "_id": "5d41402abc4b2a76b9719d911017c592ae0c9e7b..." })
```

### Count Chunks per Text ID
```javascript
db.chunks.aggregate([
  { $group: { _id: "$text_id", count: { $sum: 1 } } },
  { $sort: { count: -1 } }
])
```

### Find Chunks by Document Title
```javascript
db.chunks.find({ "sanity_title": "Introduction to Halacha" })
```

### Get All Chunks from a Specific Document
```javascript
db.chunks.find({ "sanity_id": "your-sanity-doc-id" })
  .sort({ "chunk_index": 1 })
```

### Find Chunks with Timestamps (SRT only)
```javascript
db.chunks.find({ 
  "time_start": { $exists: true },
  "time_end": { $exists: true }
})
```

### Search Chunk Text
```javascript
db.chunks.find({ 
  "text": { $regex: "Torah", $options: "i" } 
}).limit(10)
```

### Find Chunks by Embedding Model
```javascript
db.chunks.find({ "embedding_model": "openai" })
```

### Find Chunks by Chunking Strategy
```javascript
db.chunks.find({ "chunking_strategy": "divided" })
```

### Compare Different Embedding Models
```javascript
db.chunks.aggregate([
  { $group: { 
      _id: "$embedding_model", 
      count: { $sum: 1 } 
  }},
  { $sort: { count: -1 } }
])
```

### Find Chunks by Model and Strategy
```javascript
db.chunks.find({ 
  "embedding_model": "openai",
  "chunking_strategy": "divided"
})
```

### Find Orphaned Chunks
Chunks whose text_id doesn't exist in the main collection:
```javascript
// First, get all text_ids from main collection
const validTextIds = db.gemini_embeddings_v3_fixed_size.distinct("text_id")

// Find chunks with text_ids not in the main collection
db.chunks.find({ "text_id": { $nin: validTextIds } })
```

## Migration from Existing Data

If you have existing embeddings without chunk tracking:

### Option 1: Backfill Script

```python
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

async def backfill_chunks():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["rav_dev"]
    
    # Get all embeddings
    embeddings = db.gemini_embeddings_v3_fixed_size.find({})
    
    chunk_records = []
    text_id_counters = {}
    
    async for doc in embeddings:
        text_id = doc["text_id"]
        text_hash = doc["metadata"]["text_hash"]
        
        # Track index for this text_id
        if text_id not in text_id_counters:
            text_id_counters[text_id] = 0
        
        chunk_records.append({
            "_id": text_hash,
            "text_id": text_id,
            "chunk_index": text_id_counters[text_id]
        })
        text_id_counters[text_id] += 1
    
    # Insert chunks (skip duplicates)
    if chunk_records:
        try:
            await db.chunks.insert_many(chunk_records, ordered=False)
        except Exception as e:
            print(f"Some duplicates skipped: {e}")
    
    print(f"Backfilled {len(chunk_records)} chunks")

asyncio.run(backfill_chunks())
```

### Option 2: Gradual Migration

Leave existing documents without chunk tracking. Only new uploads will be tracked. Over time, as documents are updated or re-uploaded, they'll be tracked automatically.

## Monitoring

### Check Collection Stats
```javascript
db.chunks.stats()
```

### Sample Documents
```javascript
db.chunks.find().limit(5)
```

### Count by Text ID
```javascript
db.chunks.aggregate([
  { $group: { _id: "$text_id", count: { $sum: 1 } } },
  { $count: "unique_text_ids" }
])
```

## Benefits

1. **Fast Deduplication**: O(1) lookup by hash
2. **Chunk Ordering**: Maintain chunk sequence for reconstruction
3. **Storage Efficiency**: Minimal overhead (only 3 fields per chunk)
4. **Analytics**: Easy to analyze chunk distribution and overlap
5. **Debugging**: Quickly identify and inspect specific chunks
6. **Future-Proof**: Foundation for advanced features like smart updates and content versioning

## Future Enhancements

Potential features building on the chunks collection:

1. **Smart Re-embedding**: Only re-embed chunks whose content has changed
2. **Chunk Deduplication Service**: Identify and merge duplicate chunks
3. **Content Versioning**: Track chunk changes over time
4. **Chunk Analytics Dashboard**: Visualize chunk distribution and overlaps
5. **Incremental Updates**: Update only changed portions of documents
6. **Cross-Document Analysis**: Find related content across documents

## Technical Notes

- The chunks collection is **optional** - if not provided, chunk tracking is skipped
- Chunk tracking happens **after** successful vector insertion
- Failures in chunk tracking **don't affect** the main insertion
- The `_id` field uses the chunk's SHA-256 hash for natural deduplication
- Duplicate hash insertions are silently skipped (idempotent operation)

