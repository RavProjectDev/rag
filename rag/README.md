# RAG (Retrieval-Augmented Generation) System

A sophisticated FastAPI-based RAG system designed for processing and querying transcript data with advanced embedding and LLM capabilities. This system specializes in handling subtitle (.srt) files and provides intelligent chat responses based on vector similarity search.

## 🚀 Features

### Core Functionality
- **Document Processing**: Upload and process subtitle (.srt) files
- **Vector Embeddings**: Generate embeddings using Google's Gemini model
- **Semantic Search**: Retrieve relevant document chunks based on vector similarity
- **Chat Interface**: Interactive chat with streaming and non-streaming responses
- **LLM Integration**: Powered by OpenAI GPT-4 for intelligent responses
- **Form-based Interface**: Web form for easy interaction and ratings collection

### Technical Features
- **Async Architecture**: Built with FastAPI for high-performance async operations
- **MongoDB Integration**: Vector storage with MongoDB for scalable data management
- **Metrics & Monitoring**: Comprehensive logging and performance metrics
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **CORS Support**: Cross-origin resource sharing enabled
- **Health Monitoring**: Built-in health check endpoints
- **Scheduler**: Background task processing for data synchronization

## 🏗️ Architecture

```
rag/
├── app/
│   ├── api/v1/           # API endpoints
│   │   ├── chat.py       # Chat functionality
│   │   ├── data_management.py  # Document upload/management
│   │   ├── health.py     # Health checks
│   │   ├── form.py       # Form interface
│   │   └── docs.py       # Documentation
│   ├── core/             # Core configuration
│   ├── db/               # Database connections
│   ├── models/           # Data models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   │   ├── embedding.py  # Embedding generation
│   │   ├── llm.py        # LLM integration
│   │   └── data_upload_service.py  # Document processing
│   └── main.py           # FastAPI application
├── tests/                # Test suite
├── docker-compose.yml    # Docker orchestration
├── Dockerfile           # Container configuration
└── requirements.txt     # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.13+
- MongoDB instance
- OpenAI API key
- Google Cloud credentials (for Gemini embeddings)

### Environment Variables
Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID=your_project_id
GEMINI_API_KEY=your_gemini_api_key
VERTEX_REGION=us-central1

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=rag_database
MONGODB_VECTOR_COLLECTION=embeddings
COLLECTION_INDEX=vector_index

# Optional Configuration
EXTERNAL_API_TIMEOUT=60
ENVIRONMENT=PRD  # or TEST
```

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build manually**
   ```bash
   docker build -t rag-system .
   docker run -p 8000:8000 rag-system
   ```

## 📚 API Documentation

### Base URL
- Local: `http://localhost:8000`
- Docker: `http://localhost:8000`

### Interactive API Docs
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔌 API Endpoints

### Chat Endpoints

#### POST `/api/v1/chat/`
Process chat requests with streaming or full responses.

**Request Body:**
```json
{
  "question": "What are the main themes in Rav Soloveitchik's teachings?",
  "type_of_request": "STREAM",  // or "FULL"
  "name_spaces": ["optional_namespace"]
}
```

**Response:**
- **Streaming**: Server-Sent Events (SSE) with real-time response chunks
- **Full**: Complete JSON response with transcript data and AI response

### Document Management

#### POST `/api/v1/upload/create`
Upload and process subtitle files.

**Request Body:**
```json
{
  "_id": "55806772-3246-4eaf-88a3-4448eb39846e",
  "_updatedAt": "2025-07-15T20:31:24Z",
  "slug": "kedusha-and-malchus",
  "title": "Kedusha and Malchus",
  "transcriptURL": "https://cdn.sanity.io/files/.../transcript.srt"
}
```

#### PATCH `/api/v1/upload/update`
Update existing documents.

#### DELETE `/api/v1/upload/delete`
Delete documents from the system.

### Health & Monitoring

#### GET `/api/v1/health/`
Health check endpoint for monitoring system status.

### Form Interface

#### GET `/form/{question}`
Retrieve relevant document chunks for a given question.

**Response:**
```json
{
  "embedding_type": "gemini_embeddings_v2",
  "documents": [
    {
      "text": "Document chunk content...",
      "metadata": {...},
      "score": 0.95
    }
  ]
}
```

#### POST `/form/upload_ratings`
Upload user ratings for document relevance.

## 🔧 Configuration

### Embedding Models
- **Gemini**: Google's Gemini embedding model (default)
- **Mock**: For testing purposes

### LLM Models
- **GPT-4**: OpenAI's GPT-4 model (default)
- **Mock**: For testing purposes

### Vector Storage
- **MongoDB**: Primary vector database
- **Collections**: Configurable collection names for different embedding types
- **Indexing**: Vector similarity search with configurable thresholds

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit_test/
pytest tests/test_db/
```

## 📊 Monitoring & Metrics

The system includes comprehensive monitoring:

- **Request Metrics**: Endpoint timing and performance data
- **Exception Logging**: Detailed error tracking
- **Embedding Metrics**: Vector generation performance
- **LLM Metrics**: Response times and token usage

## 🔒 Security

- **CORS Configuration**: Configurable cross-origin policies
- **API Key Management**: Secure handling of external API credentials
- **Input Validation**: Comprehensive request validation with Pydantic
- **Error Handling**: Structured error responses without sensitive data exposure

## 🚀 Performance Features

- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Optimized database connections
- **Caching**: LRU cache for configuration and model instances
- **Streaming Responses**: Real-time chat responses
- **Background Tasks**: Scheduled data synchronization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

[Add your license information here]

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the health endpoint at `/api/v1/health/`

---

**Note**: This system is designed for processing transcript data and providing intelligent responses based on Rav Soloveitchik's teachings. Ensure you have proper permissions and comply with relevant data protection regulations when deploying in production environments.
