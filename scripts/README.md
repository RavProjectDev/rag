# Scripts Documentation

## upload_manifest.py

Upload all documents from the manifest to your vector database with flexible configuration options.

### Purpose

This script performs a **simple upload** (not sync):
- Fetches all documents from the manifest API
- Processes and uploads each document using specified embedding model and chunking strategy
- Supports both MongoDB and Pinecone backends
- Allows runtime configuration via command-line arguments
- **Note**: This does NOT check for existing documents - use the sync service for smart syncing

### Difference: Upload vs Sync

- **`upload_manifest.py`**: Simple, brute-force upload of all manifest documents
- **`sync_service/sync_db.py`**: Smart hash-based sync that only processes changed chunks

### Usage

```bash
# Show help
python scripts/upload_manifest.py --help

# List all available options and current environment settings
python scripts/upload_manifest.py --list-options

# Use environment defaults
python scripts/upload_manifest.py

# Specify embedding model
python scripts/upload_manifest.py --embedding-model openai

# Specify both model and strategy
python scripts/upload_manifest.py \
  --embedding-model cohere \
  --chunking-strategy divided
```

### Command-Line Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--embedding-model` | `gemini-embedding-001`<br>`openai`<br>`cohere`<br>`mock` | Embedding model for document processing |
| `--chunking-strategy` | `fixed_size`<br>`divided` | Strategy for splitting documents into chunks |
| `--list-options` | N/A | Display available options and exit |

### Database Backend Configuration

The script behavior depends on your `DATABASE_CONFIGURATION` setting:

#### Pinecone Backend

- **Index Name**: Uses embedding model value (e.g., `openai`, `cohere`)
- **Namespace**: Uses chunking strategy value (e.g., `divided`)
- Can override with `PINECONE_INDEX_NAME` and `PINECONE_NAMESPACE` environment variables

Example Pinecone structure:
```
openai/                        # Index (uses text-embedding-3-large model)
├── fixed_size/                # Namespace
└── divided/                   # Namespace

cohere/                        # Index (uses embed-multilingual-v3.0 model)
├── fixed_size/                # Namespace
└── divided/                   # Namespace
```

#### MongoDB Backend

- **Collection Name**: Appends chunking strategy to base collection name
- Example: `gemini_embeddings_v2_divided`

### Examples

#### Scenario 1: Test Different Embedding Models

```bash
# Upload with Gemini
python scripts/upload_manifest.py \
  --embedding-model gemini-embedding-001 \
  --chunking-strategy fixed_size

# Upload with OpenAI (uses text-embedding-3-large)
python scripts/upload_manifest.py \
  --embedding-model openai \
  --chunking-strategy fixed_size

# Upload with Cohere (uses embed-multilingual-v3.0)
python scripts/upload_manifest.py \
  --embedding-model cohere \
  --chunking-strategy fixed_size
```

#### Scenario 2: Compare Chunking Strategies

```bash
# Fixed size chunking
python scripts/upload_manifest.py \
  --embedding-model openai \
  --chunking-strategy fixed_size

# Divided chunking
python scripts/upload_manifest.py \
  --embedding-model openai \
  --chunking-strategy divided
```

#### Scenario 3: Production Upload

```bash
# Use production settings from .env
# Ensure these are set:
# - DATABASE_CONFIGURATION=pinecone
# - EMBEDDING_CONFIGURATION=text-embedding-3-large
# - CHUNKING_STRATEGY=divided
python scripts/upload_manifest.py
```

### Environment Variables Required

```bash
# Core Settings
DATABASE_CONFIGURATION=mongo|pinecone
EMBEDDING_CONFIGURATION=gemini-embedding-001|openai|cohere  # Default if not specified via CLI
CHUNKING_STRATEGY=fixed_size|divided                        # Default if not specified via CLI

# MongoDB (if using MongoDB backend)
MONGODB_URI=mongodb://...
MONGODB_DB_NAME=your_db
MONGODB_VECTOR_COLLECTION=your_collection
COLLECTION_INDEX=vector_index
VECTOR_PATH=vector

# Pinecone (if using Pinecone backend)
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=optional_override          # Optional: defaults to embedding model
PINECONE_NAMESPACE=optional_override           # Optional: defaults to chunking strategy
PINECONE_HOST=optional_host                    # Optional: for serverless

# API Keys (based on which embedding model you use)
GEMINI_API_KEY=your_key                        # For gemini-embedding-001
OPENAI_API_KEY=your_key                        # For text-embedding-3-large
COHERE_API_KEY=your_key                        # For embed-multilingual-v3.0
```

### Error Handling

The script includes:
- Graceful keyboard interrupt handling (Ctrl+C)
- Detailed error logging with stack traces
- Validation of required environment variables
- Clear error messages for missing configuration

### Logging

The script logs:
- Configuration being used
- Database backend details
- Progress information
- Any errors encountered

Example output:
```
INFO:__main__:Using embedding model from command line: openai
INFO:__main__:Using chunking strategy from environment: divided
INFO:__main__:Starting manifest upload script...
INFO:__main__:Configuration: backend=pinecone, embedding_model=openai, chunking_strategy=divided
INFO:rag.app.db.pinecone_connection:[PINECONE] Using index=openai (embedding_model=openai), namespace=divided (chunking_strategy=divided)
```

### Tips

1. **Start Small**: Test with a subset of documents first
2. **Check Options**: Always run `--list-options` to see current settings
3. **Monitor Costs**: Different embedding models have different costs
4. **Index Management**: For Pinecone, ensure indexes exist before running
5. **API Keys**: Make sure the correct API key is set for your chosen model

