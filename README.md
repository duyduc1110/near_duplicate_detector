# Near Duplicate Detection API

A production-ready FastAPI service for detecting near-duplicate documents using TF-IDF and semantic embeddings with clustering optimization.

## 🚀 Quick Start wi}
```

## 🛠️ Developmentrequisites
- Docker and Docker Compose installed
- Git

### Setup and Run

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd happeo
   ```

2. **Download the dataset**
   - Go to [Kaggle Near Duplicate Detection Competition](https://www.kaggle.com/competitions/near-duplicates/data)
   - Download the dataset files
   - Extract and place all `.txt` files into the `data/all_docs/` folder

3. **Start the API service**
   ```bash
   docker-compose up --build
   ```

4. **Access the API**
   - API Base URL: `http://localhost:8000`
   - Interactive Docs: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

### First Run Initialization
The service will automatically:
- Load documents from `data/all_docs/` (ensure you've downloaded the Kaggle dataset)
- Create TF-IDF vectors (~30 seconds)
- Generate document clusters for optimization
- Create semantic embeddings (if enabled)
- Cache all models for fast subsequent startups

## � API Endpoints

### Document Similarity
```bash
# Find similar documents using TF-IDF
curl "http://localhost:8000/documents/123/similar?method=tfidf&threshold=0.7&limit=10"

# Find similar documents using embeddings  
curl "http://localhost:8000/documents/123/similar?method=embedding&threshold=0.8&limit=5"
```

### Document Retrieval
```bash
# Get document content
curl "http://localhost:8000/documents/123"
```

### System Status
```bash
# Check API health and model status
curl "http://localhost:8000/health"
```

## 🏗️ Architecture Overview

### Project Structure
```
├── main.py                    # FastAPI application entry point
├── docker-compose.yml         # Container orchestration
├── Dockerfile                 # Container definition
├── requirements.txt           # Python dependencies
├── data/all_docs/             # Document dataset (download from Kaggle)
├── api/                       # FastAPI application layer
│   ├── models.py             # Pydantic data models
│   ├── routes.py             # API endpoints
│   ├── services.py           # Business logic
│   └── lifespan.py           # Application lifecycle & model loading
└── scripts/                   # Core processing modules
    ├── read_data.py          # Document loading
    ├── tfidf.py              # TF-IDF vectorization (sklearn-optimized)
    ├── clustering.py         # K-means clustering for optimization
    ├── embedding.py          # Semantic embeddings (sentence-transformers)
    └── finding_similar_doc.py # Similarity computation
```

### API Layer (`api/`)

**`lifespan.py`** - Application Lifecycle
- Loads documents on startup
- Creates/loads TF-IDF vectors, clusters, embeddings
- Provides global access to models
- Handles graceful shutdown

**`routes.py`** - API Endpoints
- Document retrieval endpoints
- Similarity search endpoints
- Health check endpoints
- Request validation

**`services.py`** - Business Logic  
- Implements similarity algorithms
- Clustering optimization (44x speedup)
- Handles both TF-IDF and embedding methods
- Result formatting and ranking

**`models.py`** - Data Models
- Pydantic models for request/response validation
- Type safety and API documentation

### Scripts Layer (`scripts/`)

**`read_data.py`** - Document Loading
- Reads text files from data directory
- Text preprocessing and validation
- Memory-efficient document streaming

**`tfidf.py`** - TF-IDF Vectorization
- sklearn TfidfVectorizer integration
- Optimized parameters (min_df=2, max_df=0.8)
- Vector caching and persistence

**`clustering.py`** - Document Clustering
- K-means clustering for performance optimization
- Reduces similarity comparisons by 98%
- Cluster-based similarity search

**`embedding.py`** - Semantic Embeddings
- sentence-transformers integration
- all-MiniLM-L6-v2 model (384 dimensions)
- Batch processing optimization

**`finding_similar_doc.py`** - Similarity Computation
- Optimized cosine similarity (sklearn)
- Clustered vs. brute-force approaches
- Result ranking and filtering

## ⚡ Performance Optimization

### Clustering Strategy
- **Without clusters**: 9.8M comparisons (O(n²))
- **With clusters**: ~200K comparisons (98% reduction)
- **Speedup**: 44x faster similarity search

### Caching Strategy
- TF-IDF vectors cached as `models/tfidf_vectors.pkl`
- Clusters cached as `models/clusters.pkl`  
- Embeddings cached as `models/embeddings.pkl`
- Instant startup after first initialization

## � Configuration

### Environment Variables
```bash
# Optional: Customize in docker-compose.yml
LOG_LEVEL=INFO
MAX_FEATURES=5000
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Similarity Thresholds
- **TF-IDF**: 0.7 (recommended) - lexical similarity
- **Embeddings**: 0.8 (recommended) - semantic similarity

## 📊 API Response Examples

### Similarity Search Response
```json
{
  "query_document_id": "123",
  "similar_documents": [
    {"id": "456", "similarity_score": 0.8542},
    {"id": "789", "similarity_score": 0.7891}
  ],
  "method": "tfidf"
}
```

### Health Check Response
```json
{
  "status": "healthy",
  "models_loaded": {
    "documents": true,
    "tfidf_vectors": true, 
    "clusters": true,
    "embeddings": true
  },
  "counts": {
    "documents": 4419,
    "tfidf_vectors": 4419,
    "clusters": 4419,
    "embeddings": 4419
  }
}
```