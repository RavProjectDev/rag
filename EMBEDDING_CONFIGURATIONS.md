# Embedding Configurations Guide

## Overview

This document describes all available embedding configurations in the RAG system.

## Available Embedding Models

### 1. Gemini Embedding Models

#### `GEMINI` (gemini-embedding-001)
- **Provider**: Google Vertex AI
- **Model**: `gemini-embedding-001`
- **Task Type**: `RETRIEVAL_DOCUMENT` (default) or `RETRIEVAL_QUERY`
- **Dimensions**: 784
- **Use Case**: General-purpose retrieval embeddings
- **Configuration**: Uses existing Google Cloud credentials

#### `GEMINI_SEMANTIC_SIMILARITY` (gemini-embedding-001-semantic)
- **Provider**: Google Vertex AI
- **Model**: `gemini-embedding-001`
- **Task Type**: `SEMANTIC_SIMILARITY`
- **Dimensions**: 784
- **Use Case**: Semantic similarity tasks (e.g., semantic chunking)
- **Configuration**: Uses existing Google Cloud credentials

### 2. Cohere Embedding Models

#### `COHERE_V3` (cohere-v3)
- **Provider**: Cohere
- **Model**: `embed-english-v3.0`
- **Task Type**: `RETRIEVAL_DOCUMENT` or `RETRIEVAL_QUERY`
- **Dimensions**: 1024
- **Use Case**: High-quality English embeddings
- **Configuration**: Requires `COHERE_API_KEY` in environment

#### `COHERE_MULTILINGUAL` (embed-multilingual-v3.0)
- **Provider**: Cohere
- **Model**: `embed-multilingual-v3.0`
- **Task Type**: `RETRIEVAL_DOCUMENT` or `RETRIEVAL_QUERY`
- **Dimensions**: 1024
- **Use Case**: Multilingual content embeddings
- **Configuration**: Requires `COHERE_API_KEY` in environment

### 3. OpenAI Embedding Models

#### `OPENAI_TEXT_EMBEDDING_3_LARGE` (text-embedding-3-large)
- **Provider**: OpenAI
- **Model**: `text-embedding-3-large`
- **Task Type**: N/A (same model for queries and documents)
- **Dimensions**: 3072
- **Use Case**: High-quality embeddings with large context
- **Configuration**: Requires `OPENAI_API_KEY` in environment

### 4. Local Models

#### `BERT_SMALL` (all-MiniLM-L6-v2)
- **Provider**: Sentence Transformers (local)
- **Model**: `all-MiniLM-L6-v2`
- **Task Type**: N/A
- **Dimensions**: 384
- **Use Case**: Fast local embeddings (no API calls)
- **Configuration**: No API key needed (runs locally)

#### `MOCK` (mock)
- **Provider**: Mock (testing)
- **Model**: Deterministic hash-based
- **Task Type**: N/A
- **Dimensions**: 784
- **Use Case**: Testing and development
- **Configuration**: No API key needed

## Configuration

### Environment Variables

Add to your `.env` file:

```env
# Required for Gemini models (already configured)
GOOGLE_CLOUD_PROJECT_ID=your-project-id
VERTEX_REGION=us-central1

# Required for Cohere models
COHERE_API_KEY=your-cohere-api-key

# Required for OpenAI models (already configured)
OPENAI_API_KEY=your-openai-api-key
```

### Code Usage

```python
from rag.app.schemas.data import EmbeddingConfiguration
from rag.app.services.embedding import generate_embedding

# Use Gemini with semantic similarity
embedding = await generate_embedding(
    text="Your text here",
    configuration=EmbeddingConfiguration.GEMINI_SEMANTIC_SIMILARITY,
    task_type="SEMANTIC_SIMILARITY",
)

# Use Cohere v3
embedding = await generate_embedding(
    text="Your text here",
    configuration=EmbeddingConfiguration.COHERE_V3,
    task_type="RETRIEVAL_DOCUMENT",
)

# Use Cohere multilingual
embedding = await generate_embedding(
    text="Your multilingual text here",
    configuration=EmbeddingConfiguration.COHERE_MULTILINGUAL,
    task_type="RETRIEVAL_DOCUMENT",
)

# Use OpenAI text-embedding-3-large
embedding = await generate_embedding(
    text="Your text here",
    configuration=EmbeddingConfiguration.OPENAI_TEXT_EMBEDDING_3_LARGE,
)
```

## Model Comparison

| Model | Dimensions | Speed | Quality | Cost | Multilingual |
|-------|-----------|-------|---------|------|--------------|
| Gemini | 784 | Medium | High | Medium | No |
| Gemini Semantic | 784 | Medium | High | Medium | No |
| Cohere v3 | 1024 | Fast | Very High | Medium | No |
| Cohere Multilingual | 1024 | Fast | Very High | Medium | Yes |
| OpenAI 3-Large | 3072 | Medium | Very High | High | No |
| BERT Small | 384 | Very Fast | Medium | Free | No |

## Task Types

### RETRIEVAL_DOCUMENT
Used when embedding documents for storage/indexing. Optimized for document representation.

### RETRIEVAL_QUERY
Used when embedding user queries. Optimized for query representation and matching against documents.

### SEMANTIC_SIMILARITY
Used for semantic similarity tasks (e.g., finding similar sentences, semantic chunking).

## Recommendations

- **General Purpose**: Use `GEMINI` or `COHERE_V3`
- **Multilingual Content**: Use `COHERE_MULTILINGUAL`
- **High Quality**: Use `OPENAI_TEXT_EMBEDDING_3_LARGE` or `COHERE_V3`
- **Fast Local Processing**: Use `BERT_SMALL`
- **Semantic Chunking**: Use `GEMINI_SEMANTIC_SIMILARITY` or `BERT_SMALL`
- **Cost-Conscious**: Use `BERT_SMALL` (free) or `GEMINI`

## Dependencies

All required packages are in `requirements.txt`:
- `cohere>=4.0.0` - For Cohere embeddings
- `openai==1.91.0` - For OpenAI embeddings (already installed)
- `sentence-transformers>=2.2.0` - For BERT Small (already installed)
- Google Cloud libraries - For Gemini (already installed)

## Installation

```bash
pip install -r requirements.txt
```

## Notes

- Cohere models automatically map `RETRIEVAL_QUERY` to `search_query` and `RETRIEVAL_DOCUMENT` to `search_document`
- OpenAI models use the same embedding for both queries and documents
- Gemini models support different task types for optimization
- All models return normalized vectors suitable for cosine similarity

