# Embedding Models Update Summary

## Changes Made

### 1. Added Two New Embedding Models

Added support for **Cohere** and **OpenAI** embedding models alongside the existing Gemini model.

#### Updated Files:
- `rag/app/schemas/data.py` - Added new enum values
- `rag/app/services/embedding.py` - Implemented embedding functions
- `rag/app/core/config.py` - Added Cohere API key configuration
- `requirements.txt` - Added cohere package dependency

#### New Embedding Configurations:

```python
class EmbeddingConfiguration(Enum):
    BERT_SMALL = "all-MiniLM-L6-v2"
    BERT_SMALL_TRANSLATED = "all-MiniLM-L6-v2"
    GEMINI = "gemini-embedding-001"
    COHERE = "cohere"   # NEW - uses embed-multilingual-v3.0 model
    OPENAI = "openai"   # NEW - uses text-embedding-3-large model
    MOCK = "mock"
```

**Note:** The enum values ("cohere", "openai") are used as Pinecone index names, while the actual API calls use the full model names ("embed-multilingual-v3.0", "text-embedding-3-large").

### 2. Implemented Embedding Functions

#### Cohere Embedding (`cohere_embedding`)
- Model: `embed-multilingual-v3.0`
- Supports task types:
  - `RETRIEVAL_QUERY` → maps to `search_query`
  - `RETRIEVAL_DOCUMENT` → maps to `search_document`
- Returns: List[float] with dimension 1024

#### OpenAI Embedding (`openai_embedding`)
- Model: `text-embedding-3-large`
- Uses AsyncOpenAI client for async operations
- Returns: List[float] with dimension 3072

### 3. Updated Pinecone Architecture

Modified Pinecone connection to use:
- **Index name**: Based on embedding model (e.g., "gemini-embedding-001", "text-embedding-3-large")
- **Namespace**: Based on chunking strategy (e.g., "fixed_size", "divided")

#### Updated Files:
- `rag/app/db/pinecone_connection.py` - Added documentation
- `rag/app/main.py` - Updated Pinecone initialization logic

#### Configuration Logic:
```python
# Index name is based on embedding model
index_name = settings.pinecone_index_name or settings.embedding_configuration.value

# Namespace is based on chunking strategy
namespace = settings.pinecone_namespace or settings.chunking_strategy.value
```

### 4. Fixed Query Embedding Task Types

Updated all query embedding calls to use `RETRIEVAL_QUERY` task type for better semantic search performance.

#### Updated Files:
- `rag/app/api/v1/chat.py` - retrieve_relevant_documents function
- `rag/app/api/v1/form.py` - get_chunks and _generate functions
- `rag/app/api/v1/mock.py` - helper function
- `rag/app/services/form/get_all_form_data.py` - generate_prompts function

#### Task Type Usage:
- **Documents/Manifests**: `RETRIEVAL_DOCUMENT` (for indexing)
- **User Queries**: `RETRIEVAL_QUERY` (for searching)

## Environment Variables

### Required for Cohere:
```bash
COHERE_API_KEY=your_cohere_api_key_here
```

### Required for OpenAI:
```bash
OPENAI_API_KEY=your_openai_api_key_here  # Already exists
```

### Pinecone Configuration:
```bash
# Optional - if not set, will use embedding model value as index name
PINECONE_INDEX_NAME=gemini-embedding-001

# Optional - if not set, will use chunking strategy value as namespace
PINECONE_NAMESPACE=fixed_size

# Required for Pinecone
PINECONE_API_KEY=your_pinecone_api_key
```

## Usage Examples

### Switching Embedding Models

Update your `.env` file:

```bash
# Use Gemini (default)
EMBEDDING_CONFIGURATION=gemini-embedding-001

# Use Cohere (uses embed-multilingual-v3.0 model)
EMBEDDING_CONFIGURATION=cohere

# Use OpenAI (uses text-embedding-3-large model)
EMBEDDING_CONFIGURATION=openai
```

### Pinecone Index Organization

With the new architecture:
- Each embedding model gets its own index
- Each chunking strategy gets its own namespace within that index

Example structure:
```
Pinecone Indexes:
├── gemini-embedding-001/
│   ├── namespace: fixed_size
│   └── namespace: divided
├── openai/
│   ├── namespace: fixed_size
│   └── namespace: divided
└── cohere/
    ├── namespace: fixed_size
    └── namespace: divided
```

## Testing

Created test file: `tests/unit_test/test_new_embeddings.py`

Tests cover:
- Cohere embedding generation
- OpenAI embedding generation
- API key validation
- Task type parameter passing
- Configuration enum values

To run tests (requires pytest installation):
```bash
cd rag
python -m pytest tests/unit_test/test_new_embeddings.py -v
```

## Dependencies Added

```
cohere==5.11.4
```

## Migration Notes

### For Existing Deployments:

1. **Add API Keys**: Ensure `COHERE_API_KEY` is set if using Cohere
2. **Pinecone Indexes**: Create new indexes for each embedding model you plan to use
3. **Re-index Data**: Documents indexed with one embedding model cannot be queried with a different model
4. **Gradual Migration**: You can run different embedding models side-by-side using different indexes

### Backward Compatibility:

- Existing Gemini embeddings continue to work without changes
- MongoDB connections are unaffected
- Default configuration remains Gemini if not specified

## Performance Considerations

### Embedding Dimensions:
- **Gemini**: 3072 dimensions
- **OpenAI text-embedding-3-large**: 3072 dimensions
- **Cohere embed-multilingual-v3.0**: 1024 dimensions

### Task Type Optimization:
- Using correct task types (`RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`) improves search relevance
- Cohere explicitly maps these to `search_query` and `search_document`
- OpenAI doesn't use task types but parameter is kept for consistency

## Error Handling

All new embedding functions include:
- API key validation
- Proper exception handling
- Detailed logging
- Consistent error types (EmbeddingException, EmbeddingAPIException)

## Upload Script Updates

The `scripts/upload_manifest.py` script now supports command-line arguments for specifying embedding model and chunking strategy.

### Usage Examples:

```bash
# List available options
python scripts/upload_manifest.py --list-options

# Use default settings from environment
python scripts/upload_manifest.py

# Specify embedding model only
python scripts/upload_manifest.py --embedding-model text-embedding-3-large

# Specify both embedding model and chunking strategy
python scripts/upload_manifest.py --embedding-model embed-multilingual-v3.0 --chunking-strategy divided

# Use Cohere with fixed_size chunking
python scripts/upload_manifest.py --embedding-model embed-multilingual-v3.0 --chunking-strategy fixed_size
```

### Command-Line Arguments:

- `--embedding-model`: Choose from:
  - `gemini-embedding-001` (Google Gemini)
  - `openai` (OpenAI text-embedding-3-large)
  - `cohere` (Cohere embed-multilingual-v3.0)
  - `mock` (Testing)

- `--chunking-strategy`: Choose from:
  - `fixed_size`
  - `divided`

- `--list-options`: Display all available options and current environment settings

### Script Behavior:

1. If arguments are provided, they **override** environment variables
2. If arguments are omitted, falls back to `.env` configuration
3. For **Pinecone**: Automatically uses embedding model as index name and chunking strategy as namespace
4. For **MongoDB**: Appends chunking strategy to collection name

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Set up API keys in `.env`
3. Create Pinecone indexes for new embedding models (if using Pinecone)
4. Use the updated upload script to test different combinations:
   ```bash
   # Test Cohere
   python scripts/upload_manifest.py --embedding-model cohere --chunking-strategy divided
   
   # Test OpenAI
   python scripts/upload_manifest.py --embedding-model openai --chunking-strategy fixed_size
   ```
5. Monitor performance and costs across different models

