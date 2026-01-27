# Configuration-Scoped Sync

## Overview

The sync service now uses **configuration-scoped synchronization**, where each combination of (embedding_model, chunking_strategy) is treated independently. This allows multiple configurations to coexist for the same document without interference.

## How It Works

### Scoping Key

Every chunk is uniquely identified by:
```
(sanity_slug, embedding_model, chunking_strategy, chunk_hash)
```

### Filter Logic

When syncing a document:

```python
# Get existing hashes for THIS specific configuration
existing_hashes = await get_existing_chunk_hashes_by_slug(
    chunks_collection,
    sanity_slug="introduction-to-halacha",
    embedding_model="openai",           # ← Filters by model
    chunking_strategy="divided",        # ← Filters by strategy
)

# Only returns hashes for: openai + divided
# Ignores chunks with: gemini + divided, openai + fixed_size, etc.
```

## Example Scenario

### Document: "Introduction to Halacha"

#### Multiple Configurations Can Coexist:

```javascript
// MongoDB chunks collection

// Configuration 1: OpenAI + DIVIDED
{
  "_id": "hash1",
  "sanity_slug": "introduction-to-halacha",
  "embedding_model": "openai",
  "chunking_strategy": "divided",
  "chunk_index": 0
}
{
  "_id": "hash2",
  "sanity_slug": "introduction-to-halacha",
  "embedding_model": "openai",
  "chunking_strategy": "divided",
  "chunk_index": 1
}

// Configuration 2: Gemini + FIXED_SIZE
{
  "_id": "hash3",
  "sanity_slug": "introduction-to-halacha",
  "embedding_model": "gemini-embedding-001",
  "chunking_strategy": "fixed_size",
  "chunk_index": 0
}
{
  "_id": "hash4",
  "sanity_slug": "introduction-to-halacha",
  "embedding_model": "gemini-embedding-001",
  "chunking_strategy": "fixed_size",
  "chunk_index": 1
}

// Configuration 3: Cohere + DIVIDED
{
  "_id": "hash5",
  "sanity_slug": "introduction-to-halacha",
  "embedding_model": "cohere",
  "chunking_strategy": "divided",
  "chunk_index": 0
}
```

### Sync Behavior

**Scenario**: Sync with `openai + divided`

```
1. Query existing hashes:
   Filter: sanity_slug="intro..." AND model="openai" AND strategy="divided"
   Returns: {hash1, hash2}

2. Chunk new document with openai + divided:
   Creates: {hash1, hash2, hash6}  # hash6 is new

3. Compare:
   Existing: {hash1, hash2}
   New: {hash1, hash2, hash6}
   
4. Actions:
   - Skip: hash1, hash2 (already exist)
   - Insert: hash6 (new chunk)
   - Remove: none

5. Other configurations UNTOUCHED:
   - gemini + fixed_size chunks remain
   - cohere + divided chunks remain
```

## Benefits

### 1. A/B Testing

Test different embedding models on the same content:

```bash
# Sync with OpenAI
python sync_db.py --embedding-model openai --chunking-strategy divided

# Sync with Gemini (doesn't affect OpenAI chunks)
python sync_db.py --embedding-model gemini-embedding-001 --chunking-strategy divided

# Now you have both in the database for comparison!
```

### 2. Gradual Migration

Migrate from one model to another without deleting old data:

```bash
# Old production: Gemini + FIXED_SIZE
# New production: OpenAI + DIVIDED

# Step 1: Sync new configuration
python sync_db.py --embedding-model openai --chunking-strategy divided

# Step 2: Test and validate

# Step 3: Switch application to use new configuration

# Step 4: Eventually clean up old configuration
db.chunks.deleteMany({
  "embedding_model": "gemini-embedding-001",
  "chunking_strategy": "fixed_size"
})
```

### 3. Strategy Comparison

Compare chunking strategies with the same embedding model:

```bash
# Upload with FIXED_SIZE
python sync_db.py --embedding-model openai --chunking-strategy fixed_size

# Upload with DIVIDED
python sync_db.py --embedding-model openai --chunking-strategy divided

# Both coexist - compare retrieval quality!
```

### 4. Safe Experimentation

Experiment with new configurations without affecting production:

```bash
# Production uses: openai + divided
# Test new model: cohere + divided

python sync_db.py --embedding-model cohere --chunking-strategy divided

# If cohere doesn't work well, just delete those chunks
# Production (openai) chunks remain untouched
```

## Query Examples

### Get All Configurations for a Document

```javascript
db.chunks.aggregate([
  {
    $match: { "sanity_slug": "introduction-to-halacha" }
  },
  {
    $group: {
      _id: {
        model: "$embedding_model",
        strategy: "$chunking_strategy"
      },
      count: { $sum: 1 }
    }
  }
])

// Result:
// { _id: { model: "openai", strategy: "divided" }, count: 80 }
// { _id: { model: "gemini-embedding-001", strategy: "fixed_size" }, count: 120 }
// { _id: { model: "cohere", strategy: "divided" }, count: 85 }
```

### Get Chunks for Specific Configuration

```javascript
db.chunks.find({
  "sanity_slug": "introduction-to-halacha",
  "embedding_model": "openai",
  "chunking_strategy": "divided"
}).sort({ "chunk_index": 1 })
```

### Count Chunks by Configuration

```javascript
db.chunks.aggregate([
  {
    $group: {
      _id: {
        model: "$embedding_model",
        strategy: "$chunking_strategy"
      },
      count: { $sum: 1 },
      documents: { $addToSet: "$sanity_slug" }
    }
  },
  { $sort: { count: -1 } }
])
```

### Compare Storage Across Models

```javascript
db.chunks.aggregate([
  {
    $group: {
      _id: "$embedding_model",
      total_chunks: { $sum: 1 },
      avg_chunk_size: { $avg: "$chunk_size" },
      documents: { $addToSet: "$sanity_slug" }
    }
  }
])
```

## Recommended Indexes

For optimal performance with configuration-scoped queries:

```javascript
// Compound index for configuration-scoped lookups
db.chunks.createIndex({ 
  "sanity_slug": 1, 
  "embedding_model": 1, 
  "chunking_strategy": 1 
})

// Index for configuration-scoped queries with ordering
db.chunks.createIndex({ 
  "sanity_slug": 1, 
  "embedding_model": 1, 
  "chunking_strategy": 1,
  "chunk_index": 1
})

// Individual indexes for analytics
db.chunks.createIndex({ "embedding_model": 1 })
db.chunks.createIndex({ "chunking_strategy": 1 })
```

## Configuration Isolation

### Same Document, Different Configs

```
Document: "introduction-to-halacha"

Configuration 1: openai + divided
├─ 80 chunks
├─ Stored in Pinecone index "openai", namespace "divided"
└─ Tracked in MongoDB chunks collection

Configuration 2: gemini-embedding-001 + fixed_size  
├─ 120 chunks
├─ Stored in Pinecone index "gemini-embedding-001", namespace "fixed_size"
└─ Tracked in MongoDB chunks collection

Both coexist independently!
```

### Sync Operations Are Isolated

```bash
# Sync openai + divided
python sync_db.py --embedding-model openai --chunking-strategy divided

# Query: sanity_slug="intro..." AND model="openai" AND strategy="divided"
# Only compares against openai+divided chunks
# Gemini chunks completely unaffected
```

## Use Cases

### 1. Model Evaluation

```bash
# Generate embeddings with 3 different models
python sync_db.py --embedding-model openai --chunking-strategy divided
python sync_db.py --embedding-model gemini-embedding-001 --chunking-strategy divided
python sync_db.py --embedding-model cohere --chunking-strategy divided

# Compare retrieval quality in production
# Switch between models in application config
```

### 2. Strategy Optimization

```bash
# Test different chunking strategies
python sync_db.py --embedding-model openai --chunking-strategy fixed_size
python sync_db.py --embedding-model openai --chunking-strategy divided

# Analyze which produces better results
```

### 3. Cost Optimization

```bash
# Production: expensive but high-quality
python sync_db.py --embedding-model openai --chunking-strategy divided

# Development: cheap mock model
python sync_db.py --embedding-model mock --chunking-strategy fixed_size

# Both available for different environments
```

### 4. Gradual Rollout

```bash
# Week 1: Start with Gemini
python sync_db.py --embedding-model gemini-embedding-001 --chunking-strategy divided

# Week 2: Add OpenAI for comparison
python sync_db.py --embedding-model openai --chunking-strategy divided

# Week 3: Measure performance, choose winner

# Week 4: Delete losing configuration
```

## Storage Impact

### Example Database State

```javascript
db.chunks.countDocuments()
// Total: 3,200 chunks

db.chunks.aggregate([
  { $group: { _id: "$embedding_model", count: { $sum: 1 } } }
])
// Results:
// { _id: "openai", count: 1,200 }
// { _id: "gemini-embedding-001", count: 1,500 }
// { _id: "cohere", count: 500 }
```

Each configuration adds its own set of chunks, so storage grows linearly with configurations.

## Cleanup

### Remove Specific Configuration

```javascript
// Remove all gemini + fixed_size chunks
db.chunks.deleteMany({
  "embedding_model": "gemini-embedding-001",
  "chunking_strategy": "fixed_size"
})

// Also remove from vector store
// For Pinecone: delete from index "gemini-embedding-001", namespace "fixed_size"
// For MongoDB: delete from collection "gemini_embeddings_v3_fixed_size"
```

### Remove All But One Configuration

```javascript
// Keep only openai + divided, remove everything else
db.chunks.deleteMany({
  $or: [
    { "embedding_model": { $ne: "openai" } },
    { "chunking_strategy": { $ne: "divided" } }
  ]
})
```

## Best Practices

1. **Document your configurations**: Keep track of which configs are in production
2. **Clean up unused configs**: Remove old configurations to save storage
3. **Use indexes**: Create compound indexes for configuration queries
4. **Monitor storage**: Track growth across configurations
5. **Test before deleting**: Validate new configuration before removing old

## Summary

✅ **Independent configurations**: Each (model, strategy) combination is isolated  
✅ **Safe experimentation**: Test new configs without affecting production  
✅ **A/B testing ready**: Compare different approaches side-by-side  
✅ **Gradual migration**: Smooth transition between configurations  
✅ **No cross-contamination**: Syncing one config doesn't affect others  

Configuration-scoped sync is production-ready for multi-configuration environments! 🎯

