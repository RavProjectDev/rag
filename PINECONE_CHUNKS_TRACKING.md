# Pinecone with MongoDB Chunks Tracking

## Overview

The system now tracks chunks in MongoDB's `chunks` collection even when using Pinecone as the primary vector store. This ensures consistent chunk tracking across both database backends.

## Changes Made

### 1. Updated `PineconeEmbeddingStore` (`app/db/pinecone_connection.py`)

#### Added `chunks_collection` Parameter
```python
def __init__(
    self,
    api_key: str,
    index_name: str,
    environment: str | None = None,
    namespace: str | None = None,
    host: str | None = None,
    chunks_collection=None,  # NEW
):
```

#### Added `_track_chunks()` Method
Implements the same chunk tracking logic as `MongoEmbeddingStore`:
- Groups chunks by `text_id`
- Computes sequential `chunk_index`
- Inserts into MongoDB `chunks` collection
- Uses chunk hash as `_id` for deduplication

#### Updated `insert()` Method
Now calls `_track_chunks()` after successfully inserting to Pinecone:
```python
await asyncio.to_thread(
    self.index.upsert, vectors=vectors, namespace=self.namespace
)

# Track chunks in MongoDB if configured
if self.chunks_collection is not None:
    await self._track_chunks(embedded_data)
```

### 2. Updated `upload_manifest.py` (`scripts/upload_manifest.py`)

When using Pinecone, now creates MongoDB connection for chunks tracking:
```python
# Get MongoDB connection for chunks tracking
mongo_client = AsyncIOMotorClient(
    settings.mongodb_uri, tlsCAFile=certifi.where(), maxPoolSize=50
)
mongo_db = mongo_client[settings.mongodb_db_name]
chunks_collection = mongo_db["chunks"]

conn = PineconeEmbeddingStore(
    api_key=settings.pinecone_api_key,
    index_name=index_name,
    environment=settings.pinecone_environment,
    namespace=namespace,
    host=settings.pinecone_host,
    chunks_collection=chunks_collection,  # NEW
)
```

### 3. Updated `main.py` (`app/main.py`)

When initializing Pinecone connection, now sets up chunks collection:
```python
# Get chunks collection for tracking (even when using Pinecone)
chunks_collection = db["chunks"]
logger.info(f"MongoDB chunks collection: chunks (for Pinecone chunk tracking)")

embedding_connection = PineconeEmbeddingStore(
    api_key=settings.pinecone_api_key,
    index_name=index_name,
    environment=settings.pinecone_environment,
    namespace=namespace,
    host=settings.pinecone_host,
    chunks_collection=chunks_collection,  # NEW
)
```

## Architecture

### Data Flow

```
Document Upload
    ↓
Chunks Created (with SHA-256 hash)
    ↓
Embeddings Generated
    ↓
┌─────────────────┬─────────────────┐
│   Pinecone      │   MongoDB       │
│   (vectors)     │   (chunks)      │
├─────────────────┼─────────────────┤
│ • Vector data   │ • _id: hash     │
│ • Metadata      │ • text_id       │
│ • Index/NS      │ • chunk_index   │
└─────────────────┴─────────────────┘
```

### Storage Locations

**Pinecone:**
- **Index**: Based on embedding model (e.g., `openai`, `gemini-embedding-001`)
- **Namespace**: Based on chunking strategy (e.g., `fixed_size`, `divided`)
- **Stores**: Full vectors and metadata including hash

**MongoDB:**
- **Collection**: `chunks`
- **Stores**: Lightweight tracking records
  ```javascript
  {
    "_id": "sha256_hash",      // Unique hash
    "text_id": "uuid",         // Groups related chunks
    "chunk_index": 0           // Global sequential index
  }
  ```

## Benefits

1. **Consistent Tracking**: Chunks are tracked the same way regardless of vector store
2. **Deduplication**: Easily identify duplicate chunks across uploads
3. **Analytics**: Analyze chunk distribution without querying Pinecone
4. **Debugging**: Quick lookups by hash or text_id
5. **Future-Proof**: Foundation for advanced features like smart updates

## Configuration

No additional configuration needed! The system automatically:
1. Connects to MongoDB for chunks tracking
2. Uses `MONGODB_URI` and `MONGODB_DB_NAME` from environment
3. Creates `chunks` collection if it doesn't exist
4. Tracks chunks during insertion

## Usage

### Running Upload Script

```bash
# With Pinecone backend - chunks automatically tracked in MongoDB
export DATABASE_CONFIGURATION=pinecone
python scripts/upload_manifest.py --embedding-model openai --chunking-strategy divided
```

### Checking Chunks

```javascript
// MongoDB shell
use your_database_name

// Count total chunks
db.chunks.countDocuments()

// Find chunks by text_id
db.chunks.find({ "text_id": "your-uuid" }).sort({ "chunk_index": 1 })

// Find chunk by hash
db.chunks.findOne({ "_id": "your-sha256-hash" })

// Check for duplicates
db.chunks.aggregate([
  { $group: { _id: "$_id", count: { $sum: 1 } } },
  { $match: { count: { $gt: 1 } } }
])
```

## Error Handling

- Chunk tracking failures don't block vector insertion
- Duplicate hash insertions are silently ignored
- All errors are logged but not raised
- System remains operational even if MongoDB is unavailable

## Environment Variables Required

When using Pinecone with chunk tracking:

```bash
# Pinecone settings
DATABASE_CONFIGURATION=pinecone
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=optional  # defaults to embedding model
PINECONE_NAMESPACE=optional   # defaults to chunking strategy

# MongoDB settings (for chunks tracking)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=your_database

# Embedding settings
EMBEDDING_CONFIGURATION=openai  # or gemini-embedding-001, cohere
CHUNKING_STRATEGY=divided       # or fixed_size
```

## Backward Compatibility

- Existing Pinecone deployments work unchanged
- `chunks_collection` parameter is optional
- If not provided, chunks are not tracked (Pinecone-only mode)
- No migration needed for existing data

## Future Enhancements

With chunk tracking in place, we can now implement:

1. **Smart Re-embedding**: Only re-embed chunks that changed
2. **Deduplication Service**: Identify and merge duplicate chunks
3. **Content Versioning**: Track chunk changes over time
4. **Cross-Database Sync**: Sync chunks between Pinecone and MongoDB
5. **Analytics Dashboard**: Visualize chunk distribution and overlaps

## Technical Notes

- MongoDB connection is created during startup/upload
- Connection is reused across all chunk operations
- Proper cleanup handled in context managers
- Async operations maintain non-blocking behavior
- Chunk tracking happens after successful vector insertion

