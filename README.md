# 🇨🇴 gAItan - RAG Colombia Chatbot

**gAItan** is a specialized RAG (Retrieval-Augmented Generation) system that provides intelligent responses about Colombia based on Wikipedia data. The name "gAItan" is a play on words combining "AI" with "Gaitán," referencing Jorge Eliécer Gaitán, the influential Colombian political leader, making it a fitting tribute to Colombian history and artificial intelligence.

## 🎯 Overview

This project implements a complete RAG pipeline that:
- Extracts and processes information from Colombia's Wikipedia page
- Creates vector embeddings for semantic search
- Provides contextual answers about Colombian geography, history, culture, and more
- Offers both API endpoints and a web interface for interaction

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
gAItan/
├── index/
│   ├── extractor.py      # Web content extraction and cleaning
│   ├── splitter.py       # Text splitting and chunking
│   ├── vector_store.py   # Vector database operations
│   ├── llm_service.py    # LLM response generation
│   └── index.py          # Main indexing service
├── api.py                # FastAPI REST endpoints
├── main.py               # CLI interface and indexing pipeline
├── index.html            # Web chat interface
└── requirements.txt      # Dependencies
```

## 🚀 Features

### Core RAG Pipeline
- **Extraction**: Automated content extraction from Wikipedia using Firecrawl
- **Processing**: Intelligent text splitting with markdown header awareness
- **Embeddings**: Multilingual sentence transformers for semantic understanding
- **Vector Storage**: Pinecone cloud database for scalable vector operations
- **Generation**: Local LLM using Ollama or hosted models via OpenRouter

### API Capabilities
- RESTful endpoints with FastAPI
- Health monitoring and system statistics
- CORS-enabled for web integration
- Comprehensive error handling

### Web Interface
- Modern, responsive chat interface
- Real-time typing indicators
- Connection status monitoring
- Mobile-friendly design

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Framework** | FastAPI |
| **RAG Library** | LangChain |
| **Vector DB** | Pinecone |
| **Embeddings** | HuggingFace Transformers |
| **LLM** | Ollama (local) / OpenRouter (hosted) |
| **Web Extraction** | Firecrawl |
| **Frontend** | Vanilla HTML/CSS/JavaScript |

## 📋 Prerequisites

- Python 3.8+
- Ollama (for local LLM) or OpenRouter API key
- Pinecone account and API key
- Firecrawl API key

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/gaitan-rag-colombia.git
cd gaitan-rag-colombia
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:

```env
# Required
PINECONE_API_KEY=your_pinecone_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# Optional (for hosted LLM)
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Ollama Setup (Local LLM)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required model
ollama pull gemma3n:e2b
```

## 🚀 Quick Start

### 1. Index Colombian Wikipedia Data
```bash
python main.py
```

This will:
- Extract content from Colombia's Wikipedia page
- Process and chunk the text
- Create vector embeddings
- Store in Pinecone database

### 2. Start the API Server
```bash
python api.py
```

The API will be available at `http://localhost:8000`

### 3. Access the Web Interface
Open your browser and navigate to `http://localhost:8000`

## 📡 API Endpoints

### Health Check
```http
GET /health
```

### Ask Questions
```http
POST /ask
Content-Type: application/json

{
  "question": "¿Cuál es la capital de Colombia?"
}
```

### Get Statistics
```http
GET /stats
```

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🐳 Docker Deployment

### Build the Image
```bash
docker build -t gaitan-rag .
```

### Run the Container
```bash
docker run -d \
  --name gaitan \
  -p 8000:8000 \
  --env-file .env \
  gaitan-rag
```

### Docker Compose (Recommended)
```yaml
version: '3.8'
services:
  gaitan:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
```

## 🔍 Usage Examples

### Python API Client
```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "¿Cuáles son los departamentos de Colombia?"}
)

answer = response.json()
print(answer['answer'])
```

### cURL
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about Colombian coffee"}'
```

## 🎛️ Configuration

### Model Configuration
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM Model**: `gemma3n:e2b` (Ollama) or `google/gemma-3n-e4b-it:free` (OpenRouter)
- **Vector Dimensions**: 384
- **Chunk Size**: 1000 tokens
- **Chunk Overlap**: 200 tokens

### Pinecone Settings
- **Index Name**: `colombia-rag`
- **Metric**: Cosine similarity
- **Cloud**: AWS (us-east-1)

## 🧪 Testing

### Run Unit Tests
```bash
python -m pytest tests/
```

### Test the Pipeline
```bash
python main.py  # Full indexing pipeline test
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Sample question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es la Cordillera de los Andes?"}'
```

## 📊 Performance Metrics

- **Index Size**: ~2,000-3,000 vector embeddings
- **Response Time**: < 2 seconds average
- **Similarity Threshold**: 0.25 for relevance filtering
- **Context Window**: Up to 10 relevant chunks per query

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Jorge Eliécer Gaitán**: Inspiration for the project name
- **LangChain**: For the excellent RAG framework
- **Pinecone**: For vector database services
- **Ollama**: For local LLM capabilities
- **Wikipedia**: For the comprehensive Colombian data

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: [alvarojcabrera8@gmail.com]

---

**Made with ❤️ for Colombia** 🇨🇴
