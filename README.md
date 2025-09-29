# 🎓 TutorlyAI - AI-Powered Educational Platform

**TutorlyAI** is an advanced AI-powered educational platform that combines RAG (Retrieval-Augmented Generation), intelligent caching, quiz generation, English learning, and image generation capabilities. Built with FastAPI and powered by Fal.ai's Gemini 2.5 Flash model.

---

## 🌟 Features

### 🧠 Core AI Capabilities
- **Advanced RAG System** - Hybrid search combining semantic (embedding-based) and keyword-based (BM25) retrieval
- **Intelligent Query Planning** - Automatic query understanding and filter extraction
- **Smart Caching** - Multi-layer Redis caching with semantic similarity matching
- **Streaming Responses** - Real-time text generation with Server-Sent Events

### 📚 Educational Features
- **Quiz Generation** - Multiple choice and open-ended questions with difficulty levels
- **English Learning** - Level-based English teaching (A1, A2, B1, B2, C1, C2)
- **Lesson Planning** - Automated lesson plan generation
- **Image Generation** - Educational content visualization
- **Grade-Based Content** - Supports curriculum for grades 9-12

### 🚀 Performance & Reliability
- **Redis Cache Integration** - Query cache, performance cache, BM25 cache, session cache
- **Similarity-Based Cache** - Fuzzy cache matching with 80% similarity threshold
- **Circuit Breaker Pattern** - Automatic failure handling and recovery
- **Retry Mechanism** - Exponential backoff with jitter
- **Memory Monitoring** - Real-time memory usage tracking
- **Rate Limiting** - Concurrent request limiting per user

### 🔒 Security
- **API Key Authentication** - Secure API access control
- **Input Sanitization** - SQL injection, XSS, command injection protection
- **Request Size Limits** - Prevention of DoS attacks
- **Security Headers** - CSP, HSTS, X-Frame-Options
- **Violation Tracking** - Security event logging and monitoring

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Endpoints](#-api-endpoints)
- [Usage Examples](#-usage-examples)
- [RAG System](#-rag-system-deep-dive)
- [Caching Strategy](#-caching-strategy)
- [Security](#-security-features)
- [Performance Tuning](#-performance-tuning)
- [Troubleshooting](#-troubleshooting)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Application                   │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Security   │  │ Performance  │  │  Resilience  │      │
│  │  Middleware  │  │  Middleware  │  │   Patterns   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────▼─────┐        ┌────▼─────┐        ┌────▼─────┐
    │   RAG    │        │  Redis   │        │  Fal.ai  │
    │  Engine  │        │  Cache   │        │  Model   │
    └──────────┘        └──────────┘        └──────────┘
         │                    │
    ┌────▼─────┐        ┌────▼─────┐
    │ ChromaDB │        │ Similarity│
    │ Vector   │        │  Cache    │
    │   Store  │        └───────────┘
    └──────────┘
```

### Key Components

1. **RAG System** (`tools/initalize_rag_system.py`)
   - ChromaDB vector store for semantic search
   - BM25 index for keyword-based search
   - Hybrid retriever combining both approaches

2. **Cache Layer** (`tools/redis_cache_adapters.py`, `tools/similarity_cache.py`)
   - Multi-database Redis architecture (query, performance, BM25, session)
   - Semantic similarity cache with 80% threshold
   - Query deduplication and result caching

3. **Security Layer** (`tools/security_utils.py`)
   - API key validation
   - Input sanitization (SQL, XSS, command injection)
   - Rate limiting and violation tracking

4. **Resilience System** (`tools/resilience_utils.py`)
   - Circuit breaker with configurable thresholds
   - Retry with exponential backoff
   - Fallback responses

5. **Performance Monitoring** (`tools/performance_utils.py`)
   - Memory usage tracking
   - Concurrent request limiting
   - Performance statistics

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Redis server (local or remote)
- Fal.ai API key

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/TutorlyAI.git
cd TutorlyAI
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies Overview
```
Core Framework:
├── fastapi>=0.104.0           # Web framework
├── uvicorn[standard]>=0.24.0  # ASGI server
└── pydantic>=2.0.0            # Data validation

AI & LLM:
├── fal-client>=0.4.0          # Fal.ai integration
├── google-generativeai>=0.3.0 # Gemini model
└── tiktoken>=0.5.0            # Token counting

RAG System:
├── chromadb>=0.4.0            # Vector database
├── sentence-transformers>=2.2.0 # Embeddings
├── langchain>=0.1.0           # RAG framework
├── langchain-community>=0.0.13
├── langchain-chroma>=0.1.0
├── langchain-huggingface>=0.0.13
└── PyPDF2>=3.0.0              # PDF processing

Hybrid Search:
├── rank-bm25>=0.2.2           # BM25 algorithm
├── scikit-learn>=1.3.0        # ML utilities
└── numpy>=1.24.0              # Numerical computing

Cache & Performance:
└── redis[asyncio]>=5.0.0      # Redis client with async
```

### Step 3: Setup Redis
```bash
# Option 1: Docker (Recommended)
docker-compose up -d

# Option 2: Local Redis Installation
# Windows: https://redis.io/docs/getting-started/installation/install-redis-on-windows/
# Linux: sudo apt-get install redis-server
# macOS: brew install redis
```

### Step 4: Prepare Books Directory
```bash
mkdir -p books
# Add your PDF textbooks (grades 9-12) to the books/ directory
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# ================================
# REQUIRED - Fal.ai Configuration
# ================================
FAL_KEY=your_fal_api_key_here

# ================================
# REQUIRED - API Security
# ================================
VALIDATE_API_KEY=your_secure_api_key_here

# ================================
# Redis Configuration
# ================================
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=                    # Optional
REDIS_SSL=false

# Redis Database Selection (0-15)
REDIS_QUERY_CACHE_DB=0
REDIS_PERFORMANCE_CACHE_DB=1
REDIS_BM25_CACHE_DB=2
REDIS_SESSION_CACHE_DB=3

# Redis TTL Settings (seconds)
REDIS_QUERY_CACHE_TTL=300          # 5 minutes
REDIS_PERFORMANCE_CACHE_TTL=60     # 1 minute
REDIS_BM25_CACHE_TTL=3600          # 1 hour
REDIS_SESSION_CACHE_TTL=1800       # 30 minutes

# ================================
# Similarity Cache Configuration
# ================================
REDIS_SIMILARITY_THRESHOLD=0.80    # 80% similarity for cache hit
REDIS_MAX_SIMILAR_QUERIES=500      # Max cached queries
REDIS_ENABLE_SIMILARITY_CACHE=true
REDIS_FINAL_RESPONSE_TTL=900       # 15 minutes

# ================================
# RAG System Configuration
# ================================
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
SEARCH_K=5
SCORE_THRESHOLD=0.25

# Hybrid Search Weights
SEMANTIC_WEIGHT=0.7
KEYWORD_WEIGHT=0.3

# ================================
# Performance Configuration
# ================================
MAX_CONCURRENT_REQUESTS_PER_USER=5
MAX_CONCURRENT_REQUESTS_GLOBAL=50
MEMORY_WARNING_THRESHOLD=80        # %
MEMORY_CRITICAL_THRESHOLD=90       # %

# ================================
# Resilience Configuration
# ================================
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_SUCCESS_THRESHOLD=2
CIRCUIT_BREAKER_TIMEOUT=60         # seconds
MAX_RETRY_ATTEMPTS=3
RETRY_BASE_DELAY=1                 # seconds
RETRY_MAX_DELAY=10                 # seconds
```

### Getting API Keys

#### Fal.ai API Key
1. Visit [fal.ai](https://fal.ai)
2. Sign up or log in
3. Go to Dashboard → API Keys
4. Create a new key
5. Copy and add to `.env` as `FAL_KEY`

---

## 📡 API Endpoints

### Health & Status

#### GET `/health`
Health check endpoint
```json
{
  "status": "healthy",
  "rag_active": true,
  "timestamp": "2024-01-01T12:00:00",
  "performance": {
    "concurrent_requests": 3,
    "memory_usage_mb": 512.5,
    "cache_hit_rate": 85.2
  },
  "android_compatible": true,
  "api_version": "2.2.0"
}
```

#### GET `/rag-status`
RAG system status
```json
{
  "status": "active",
  "documents_count": 1250,
  "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "vectorstore": "ChromaDB",
  "sample_metadata": [...]
}
```

### 🎯 Core RAG API

#### POST `/generate`
Generate AI responses with RAG support

**Request:**
```json
{
  "prompt": "9. sınıf matematik fonksiyonlar hakkında bilgi ver",
  "max_tokens": 1000,
  "temperature": 0.7,
  "use_hybrid": true,
  "semantic_weight": 0.7,
  "keyword_weight": 0.3,
  "score_threshold": 0.25,
  "search_k": 5
}
```

**Response:**
```json
{
  "generated_text": "Fonksiyonlar, matematiksel ilişkilerin...",
  "search_plan": {
    "query": "fonksiyon tanımı örnekler",
    "filters": {
      "sinif": 9,
      "ders": "matematik",
      "konu_slug": "fonksiyonlar"
    }
  },
  "found_documents": 5,
  "sources": ["matematik_9_sinif.pdf", "..."],
  "search_method": "hybrid_search",
  "search_config": {
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "score_threshold": 0.25,
    "search_k": 5,
    "hybrid_enabled": true
  },
  "search_details": {
    "average_hybrid_score": 0.85,
    "average_semantic_score": 0.82,
    "average_bm25_score": 0.78,
    "score_breakdown": [...]
  }
}
```

#### POST `/generate/stream`
Streaming text generation
```bash
curl -X POST "http://localhost:8000/generate/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "10. sınıf fizik yasaları",
    "max_tokens": 800
  }'
```

#### POST `/search`
Manual search endpoint
```json
{
  "query": "atom nedir",
  "filters": {
    "sinif": 9,
    "ders": "kimya"
  },
  "k": 5,
  "score_threshold": 0.3,
  "use_hybrid": true,
  "semantic_weight": 0.7,
  "keyword_weight": 0.3
}
```

### 📝 Quiz Generation

#### POST `/quiz/generate`
Generate quiz questions

**Request:**
```json
{
  "sinif": 10,
  "ders": "fizik",
  "konu": "Hareket ve Kuvvet",
  "soru_sayisi": 5,
  "soru_tipi": "coktan_secmeli",
  "zorluk": "orta",
  "dil": "tr"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Quiz başarıyla oluşturuldu",
  "data": {
    "quiz_info": {
      "sinif": 10,
      "ders": "fizik",
      "konu": "Hareket ve Kuvvet",
      "zorluk": "orta"
    },
    "sorular": [
      {
        "soru": "Newton'un birinci yasası nedir?",
        "a": "F = ma",
        "b": "Bir cisim dengede ise net kuvvet sıfırdır",
        "c": "Her etkiye eşit ve zıt bir tepki vardır",
        "d": "Momentumun korunumu",
        "cevap": "b",
        "aciklama": "Newton'un birinci yasası..."
      }
    ],
    "total_soru": 5,
    "soru_tipi": "coktan_secmeli",
    "created_at": "2024-01-01T12:00:00"
  }
}
```

#### GET `/quiz/info`
Quiz system information

### 🌐 English Learning

#### POST `/english/generate`
Generate English learning content with level detection

**Request:**
```json
{
  "prompt": "ingilizce seviyesi: b2. Explain the difference between 'affect' and 'effect'",
  "max_tokens": 800,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "generated_text": "Great question! Let's clarify...",
  "detected_level": "B2",
  "system_prompt_used": "B2 seviyesi için özelleştirilmiş prompt",
  "clean_prompt": "Explain the difference between 'affect' and 'effect'"
}
```

#### POST `/english/stream`
Streaming English content generation

#### GET `/english/levels`
List supported English levels (A1-C2)

### 🎨 Image Generation

#### POST `/generate/image`
Generate educational images

**Request:**
```json
{
  "prompt": "Bir hücre yapısı diyagramı çiz, etiketli",
  "workflow_id": "workflows/halillllibrahim58/teach-img-model",
  "max_tokens": 1000,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "image_url": "https://fal.media/files/...",
  "workflow_id": "workflows/halillllibrahim58/teach-img-model",
  "prompt": "Bir hücre yapısı diyagramı çiz...",
  "error_message": null,
  "generated_at": "2024-01-01T12:00:00",
  "all_images": ["https://fal.media/files/..."],
  "total_images": 1
}
```

#### POST `/generate/image/stream`
Streaming image generation

#### GET `/generate/image/info`
Image generation system info

### 📋 Lesson Planning

#### POST `/lesson-plan/generate`
Generate lesson plans

**Request:**
```json
{
  "prompt": "5. sınıf matematik üslü sayılar konusunda 45 dakikalık ders planı"
}
```

**Response:**
```json
{
  "success": true,
  "lesson_plan": "# Ders Planı\n\n## Konu: Üslü Sayılar\n...",
  "error_message": null
}
```

#### POST `/lesson-plan/stream`
Streaming lesson plan generation

#### GET `/lesson-plan/info`
Lesson planning system info

### 📊 Performance & Monitoring

#### GET `/performance/stats`
Detailed performance statistics
```json
{
  "service": "TutorlyAI Performance Monitoring",
  "version": "1.0.0",
  "timestamp": "2024-01-01T12:00:00",
  "performance": {
    "memory": {
      "current_memory_mb": 512.5,
      "peak_memory_mb": 768.2,
      "current_memory_percent": 15.3
    },
    "concurrency": {
      "concurrent_requests": 3,
      "max_concurrent_requests": 50,
      "total_requests": 1250,
      "rejected_requests": 5
    },
    "cache": {
      "hits": 850,
      "misses": 400,
      "hit_rate_percent": 68.0
    }
  },
  "database": {
    "query_cache": {...},
    "bm25_cache": {...}
  }
}
```

#### GET `/performance/memory`
Memory usage statistics

#### GET `/performance/concurrency`
Concurrent request statistics

### 🛡️ Security Management

#### GET `/security/violations`
Security violations log (requires admin key)

#### POST `/security/api-keys/generate`
Generate new API key (requires admin key)

#### GET `/security/config`
Security configuration details (requires admin key)

### 🔧 Resilience Management

#### GET `/resilience/stats`
Circuit breaker and resilience statistics

#### POST `/resilience/reset`
Reset circuit breaker (emergency use)

#### GET `/resilience/config`
Resilience configuration details

---

## 💡 Usage Examples

### Python Examples

#### 1. Basic Text Generation
```python
import requests

url = "http://localhost:8000/generate"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

data = {
    "prompt": "Python programlama dilinin avantajları nelerdir?",
    "max_tokens": 500,
    "temperature": 0.8
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(result["generated_text"])
```

#### 2. RAG with Custom Search Parameters
```python
import requests

url = "http://localhost:8000/generate"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

data = {
    "prompt": "9. sınıf kimya atom modelleri",
    "max_tokens": 1000,
    "temperature": 0.7,
    # Custom hybrid search parameters
    "use_hybrid": True,
    "semantic_weight": 0.8,  # More weight on semantic search
    "keyword_weight": 0.2,
    "score_threshold": 0.3,  # Higher threshold for quality
    "search_k": 10           # More results
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

print(f"Found {result['found_documents']} documents")
print(f"Search method: {result['search_method']}")
print(f"Answer: {result['generated_text']}")
```

#### 3. Quiz Generation
```python
import requests

url = "http://localhost:8000/quiz/generate"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

data = {
    "sinif": 10,
    "ders": "matematik",
    "konu": "Trigonometri",
    "soru_sayisi": 3,
    "soru_tipi": "coktan_secmeli",
    "zorluk": "zor",
    "dil": "tr"
}

response = requests.post(url, json=data, headers=headers)
quiz = response.json()

for i, soru in enumerate(quiz["data"]["sorular"], 1):
    print(f"\nSoru {i}: {soru['soru']}")
    print(f"A) {soru['a']}")
    print(f"B) {soru['b']}")
    print(f"C) {soru['c']}")
    print(f"D) {soru['d']}")
    print(f"Cevap: {soru['cevap'].upper()}")
```

#### 4. English Learning with Level Detection
```python
import requests

url = "http://localhost:8000/english/generate"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

data = {
    "prompt": "ingilizce seviyesi: c1. Explain the nuances between 'imply' and 'infer'",
    "max_tokens": 800,
    "temperature": 0.7
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

print(f"Detected Level: {result['detected_level']}")
print(f"Content: {result['generated_text']}")
```

#### 5. Streaming Response
```python
import requests

url = "http://localhost:8000/generate/stream"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

data = {
    "prompt": "11. sınıf biyoloji hücre bölünmesi",
    "max_tokens": 800
}

response = requests.post(url, json=data, headers=headers, stream=True)

for line in response.iter_lines():
    if line:
        # Parse SSE format: "data: {...}"
        if line.startswith(b"data: "):
            print(line.decode()[6:], end='', flush=True)
```

### cURL Examples

#### 1. RAG Query
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "10. sınıf fizik Newton yasaları",
    "max_tokens": 800,
    "use_hybrid": true,
    "semantic_weight": 0.7,
    "keyword_weight": 0.3
  }'
```

#### 2. Quiz Generation
```bash
curl -X POST "http://localhost:8000/quiz/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "sinif": 9,
    "ders": "matematik",
    "konu": "İkinci Dereceden Denklemler",
    "soru_sayisi": 5,
    "soru_tipi": "coktan_secmeli",
    "zorluk": "orta"
  }'
```

#### 3. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

#### 4. Performance Stats
```bash
curl -X GET "http://localhost:8000/performance/stats"
```

### JavaScript/TypeScript Example

```typescript
// Using fetch API
async function generateWithRAG(prompt: string) {
  const response = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer YOUR_API_KEY',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      prompt: prompt,
      max_tokens: 1000,
      temperature: 0.7,
      use_hybrid: true,
      semantic_weight: 0.7,
      keyword_weight: 0.3
    })
  });

  const result = await response.json();
  return result;
}

// Usage
generateWithRAG("9. sınıf kimya atom yapısı")
  .then(result => {
    console.log('Found documents:', result.found_documents);
    console.log('Answer:', result.generated_text);
  });
```

---

## 🧠 RAG System Deep Dive

### Hybrid Search Architecture

TutorlyAI uses a **hybrid search approach** combining:

1. **Semantic Search** (Embedding-based)
   - Uses `paraphrase-multilingual-MiniLM-L12-v2` model
   - Understands context and meaning
   - Great for conceptual queries
   - Score range: 0.0 - 1.0

2. **Keyword Search** (BM25)
   - Traditional information retrieval algorithm
   - Excellent for exact term matching
   - Fast and efficient
   - Score range: normalized to 0.0 - 1.0

3. **Hybrid Scoring**
   ```python
   hybrid_score = (semantic_weight × semantic_score) +
                  (keyword_weight × bm25_score)

   # Default: 0.7 × semantic + 0.3 × keyword
   ```

### Query Planning

The system automatically analyzes queries to extract:
- **Grade level** (sınıf: 9-12)
- **Subject** (ders: matematik, fizik, kimya, biyoloji, etc.)
- **Topic** (konu: specific topic within the subject)
- **Search terms** (optimized keywords)

Example:
```
User: "9. sınıf kimya atom nedir"

Planner Output:
{
  "query": "atom nedir yapısı özellikleri temel tanımı",
  "filters": {
    "sinif": 9,
    "ders": "kimya",
    "konu_slug": "atom-yapisi"
  }
}
```

### Document Processing Pipeline

```
PDF Files → Text Extraction → Chunking → Metadata Extraction
                                           │
                                           ▼
                              ┌───────────────────────┐
                              │ Embedding Generation  │
                              └───────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                ┌─────────────┐                      ┌──────────────┐
                │  ChromaDB   │                      │  BM25 Index  │
                │   Vectors   │                      │   Keywords   │
                └─────────────┘                      └──────────────┘
```

### Metadata Structure

Each document chunk contains:
```json
{
  "source": "matematik_9_sinif.pdf",
  "sinif": 9,
  "ders": "matematik",
  "konu": "Fonksiyonlar",
  "konu_slug": "fonksiyonlar",
  "page": 42,
  "chunk_id": "abc123...",
  "content_hash": "def456..."
}
```

### Search Optimization

- **Score Threshold**: Documents below threshold are filtered (default: 0.25)
- **Top-K Results**: Returns best K documents (default: 5)
- **Deduplication**: Removes duplicate chunks
- **Relevance Ranking**: Combined semantic + keyword scores

---

## 🗄️ Caching Strategy

### Multi-Layer Cache Architecture

```
┌─────────────────────────────────────────────────┐
│              Client Request                      │
└─────────────────┬───────────────────────────────┘
                  │
    ┌─────────────▼──────────────┐
    │  L1: Similarity Cache      │ (TTL: 15m)
    │  - Fuzzy query matching    │
    │  - 80% similarity threshold│
    └─────────────┬──────────────┘
                  │ Cache Miss
    ┌─────────────▼──────────────┐
    │  L2: Query Cache           │ (TTL: 5m)
    │  - Exact query matching    │
    │  - Includes filters        │
    └─────────────┬──────────────┘
                  │ Cache Miss
    ┌─────────────▼──────────────┐
    │  L3: Performance Cache     │ (TTL: 1m)
    │  - Intermediate results    │
    └─────────────┬──────────────┘
                  │ Cache Miss
    ┌─────────────▼──────────────┐
    │  L4: BM25 Cache            │ (TTL: 1h)
    │  - BM25 index              │
    │  - Document corpus         │
    └─────────────┬──────────────┘
                  │ Cache Miss
    ┌─────────────▼──────────────┐
    │    Execute Full Search     │
    └────────────────────────────┘
```

### Similarity-Based Caching

**How it works:**
1. Query is embedded using the same model as documents
2. Cosine similarity calculated with cached queries
3. If similarity ≥ 80%, cached result is returned
4. **Filters must match exactly** (grade, subject, topic)

Example:
```python
Query 1: "atom nedir yapısı özellikleri"
Query 2: "atom yapısı ve özellikleri nedir"

Similarity: 0.87 (87%) → Cache Hit! ✓
But only if filters also match:
- Grade: 9 = 9 ✓
- Subject: kimya = kimya ✓
- Topic: atom-yapisi = atom-yapisi ✓
```

### Cache Key Strategy

```python
# Similarity Cache Key
f"tutorlyai:similarity:final_response:metadata"

# Query Cache Key
f"tutorlyai:query_cache:{query_hash}:{filter_hash}"

# Performance Cache Key
f"tutorlyai:performance:{operation}:{hash}"

# BM25 Cache Key
f"tutorlyai:bm25_index:main"
```

### Redis Database Allocation

- **DB 0**: Query Cache
- **DB 1**: Performance Cache
- **DB 2**: BM25 Index Cache
- **DB 3**: Session Cache (similarity metadata)

### Cache Statistics

Monitor cache performance:
```bash
curl http://localhost:8000/performance/stats
```

```json
{
  "cache": {
    "hits": 850,
    "misses": 400,
    "hit_rate_percent": 68.0,
    "evictions": 10,
    "current_size": 1250
  }
}
```

---

## 🔒 Security Features

### API Key Authentication

All endpoints (except health checks) require API key:
```bash
# Header-based (Recommended)
Authorization: Bearer YOUR_API_KEY

# Alternative: X-API-Key header
X-API-Key: YOUR_API_KEY

# Development only: Query parameter
?api_key=YOUR_API_KEY
```

### Input Sanitization

Protected against:
- **SQL Injection**: Pattern detection and blocking
- **XSS Attacks**: HTML/JavaScript tag removal
- **Command Injection**: Shell command pattern blocking
- **Path Traversal**: Directory traversal prevention

### Security Middleware

```python
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # 1. Client IP extraction
    # 2. Request size validation (max 10MB)
    # 3. Rate limiting check
    # 4. Input sanitization
    # 5. Security header injection
    # 6. Violation logging
```

### Security Headers

Automatically added to all responses:
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
```

### Violation Tracking

View security events:
```bash
curl -X GET "http://localhost:8000/security/violations?hours=24" \
  -H "Authorization: Bearer ADMIN_KEY"
```

### API Key Management

```bash
# Generate new key
curl -X POST "http://localhost:8000/security/api-keys/generate" \
  -H "Authorization: Bearer ADMIN_KEY" \
  -d '{"user_id": "team_member_1"}'

# Rotate key
curl -X POST "http://localhost:8000/security/api-keys/rotate" \
  -H "Authorization: Bearer ADMIN_KEY" \
  -d '{"old_key": "old_key_here"}'

# Revoke key
curl -X DELETE "http://localhost:8000/security/api-keys/{api_key}" \
  -H "Authorization: Bearer ADMIN_KEY"
```

---

## ⚡ Performance Tuning

### Memory Management

**Configuration:**
```bash
# .env
MEMORY_WARNING_THRESHOLD=80    # Warning at 80% usage
MEMORY_CRITICAL_THRESHOLD=90   # Reject requests at 90%
```

**Monitoring:**
```bash
curl http://localhost:8000/performance/memory
```

### Concurrency Limits

**Per-User Limits:**
```bash
MAX_CONCURRENT_REQUESTS_PER_USER=5
```

**Global Limits:**
```bash
MAX_CONCURRENT_REQUESTS_GLOBAL=50
```

### Circuit Breaker Configuration

```bash
# .env
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5    # Open after 5 failures
CIRCUIT_BREAKER_SUCCESS_THRESHOLD=2    # Close after 2 successes
CIRCUIT_BREAKER_TIMEOUT=60             # Try again after 60s
```

**States:**
- **CLOSED**: Normal operation
- **OPEN**: Failing, requests rejected immediately
- **HALF_OPEN**: Testing if service recovered

### Retry Configuration

```bash
MAX_RETRY_ATTEMPTS=3
RETRY_BASE_DELAY=1      # Start with 1 second
RETRY_MAX_DELAY=10      # Max 10 seconds
```

**Backoff Formula:**
```python
delay = min(base_delay * (2 ** attempt) + random_jitter, max_delay)

# Example:
# Attempt 1: 1s + jitter
# Attempt 2: 2s + jitter
# Attempt 3: 4s + jitter
```

### RAG Performance Tips

**1. Adjust Search Parameters**
```json
{
  "search_k": 3,              // Fewer results = faster
  "score_threshold": 0.4,     // Higher threshold = fewer results
  "use_hybrid": false         // Semantic only = faster
}
```

**2. Optimize Semantic Weight**
```json
{
  "semantic_weight": 0.9,     // Favor semantic (slower but accurate)
  "keyword_weight": 0.1       // Less keyword (faster)
}
```

**3. Cache Tuning**
```bash
# Increase cache TTL for stable data
REDIS_QUERY_CACHE_TTL=600        # 10 minutes
REDIS_BM25_CACHE_TTL=7200        # 2 hours

# Increase similarity cache size
REDIS_MAX_SIMILAR_QUERIES=1000
```

### Database Optimization

**ChromaDB:**
- Stored on disk for persistence
- Automatic indexing
- No manual optimization needed

**Redis:**
```bash
# Check memory usage
redis-cli info memory

# Check database sizes
redis-cli -n 0 DBSIZE  # Query cache
redis-cli -n 1 DBSIZE  # Performance cache
redis-cli -n 2 DBSIZE  # BM25 cache
redis-cli -n 3 DBSIZE  # Session cache
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. FAL_KEY Error
```
ValueError: FAL_KEY ortam değişkeni ayarlanmamış!
```

**Solution:**
```bash
# Check .env file
cat .env | grep FAL_KEY

# Verify environment variable
echo $FAL_KEY  # Linux/macOS
echo %FAL_KEY%  # Windows CMD
```

#### 2. Redis Connection Error
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solutions:**
```bash
# Check Redis is running
redis-cli ping  # Should return "PONG"

# Check configuration
cat .env | grep REDIS

# Restart Redis
# Docker: docker-compose restart redis
# Service: sudo systemctl restart redis
```

#### 3. RAG System Not Active
```
{"status": "inactive", "message": "RAG sistemi başlatılmamış"}
```

**Solutions:**
```bash
# 1. Check if books exist
ls -la books/

# 2. Load books manually
curl -X POST http://localhost:8000/load-books

# 3. Check embedding model
# Should download automatically on first run
# Check: ~/.cache/torch/sentence_transformers/

# 4. Clear and rebuild ChromaDB
rm -rf chroma_db_v2/
python fal_api.py
```

#### 4. High Memory Usage
```
{"status": "warning", "current_memory_percent": 85}
```

**Solutions:**
```bash
# 1. Check current stats
curl http://localhost:8000/performance/memory

# 2. Clear Redis cache
redis-cli FLUSHALL

# 3. Restart application
# This releases memory from ChromaDB

# 4. Reduce chunk size in .env
CHUNK_SIZE=300  # Smaller chunks = less memory
```

#### 5. Circuit Breaker Open
```
{"error": "Circuit breaker is OPEN"}
```

**Solutions:**
```bash
# 1. Check resilience stats
curl http://localhost:8000/resilience/stats

# 2. Wait for timeout (default 60s)
# Or reset manually:
curl -X POST http://localhost:8000/resilience/reset

# 3. Check Fal.ai service status
# Visit https://status.fal.ai
```

#### 6. Slow Query Performance
```
X-Process-Time: 5.234  # Too slow
```

**Solutions:**
```json
// 1. Reduce search_k
{"search_k": 3}

// 2. Increase score_threshold
{"score_threshold": 0.4}

// 3. Use semantic only
{"use_hybrid": false}

// 4. Check cache hit rate
curl http://localhost:8000/performance/stats
// If hit_rate < 50%, increase TTL
```

#### 7. Cache Not Working
```
"cache_hits": 0, "cache_misses": 100
```

**Solutions:**
```bash
# 1. Check Redis connection
redis-cli ping

# 2. Check similarity cache config
cat .env | grep SIMILARITY

# 3. Enable similarity cache
REDIS_ENABLE_SIMILARITY_CACHE=true

# 4. Clear corrupted cache
redis-cli FLUSHALL
```

### Debug Mode

Enable detailed logging:
```python
# fal_api.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

View logs:
```bash
tail -f logs/tutorlyai.log
```

### Performance Profiling

```python
# Add timing decorators
@timer
def slow_function():
    # Function code
    pass
```

Check process time headers:
```bash
curl -I http://localhost:8000/generate
# X-Process-Time: 0.234
# X-Concurrent-Requests: 3
```

---

## 🚢 Deployment

### Docker Deployment

#### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Manual Docker Build

```bash
# Build image
docker build -t tutorlyai:latest .

# Run container
docker run -d \
  --name tutorlyai \
  -p 8000:8000 \
  -v $(pwd)/books:/app/books \
  -v $(pwd)/chroma_db_v2:/app/chroma_db_v2 \
  --env-file .env \
  tutorlyai:latest
```

### Production Checklist

- [ ] Set strong `VALIDATE_API_KEY`
- [ ] Configure Redis with authentication
- [ ] Enable Redis persistence (AOF/RDB)
- [ ] Set up HTTPS/SSL
- [ ] Configure CORS for specific origins
- [ ] Enable rate limiting
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log rotation
- [ ] Set memory limits
- [ ] Enable Redis clustering (for high load)
- [ ] Set up backup for ChromaDB
- [ ] Configure health check monitoring
- [ ] Set up alerting for circuit breaker events

### Environment-Specific Settings

**Development:**
```bash
REDIS_QUERY_CACHE_TTL=60        # Short TTL for testing
REDIS_ENABLE_SIMILARITY_CACHE=true
CIRCUIT_BREAKER_FAILURE_THRESHOLD=10  # More tolerant
```

**Production:**
```bash
REDIS_QUERY_CACHE_TTL=600       # Longer TTL
REDIS_ENABLE_SIMILARITY_CACHE=true
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5   # Stricter
MEMORY_CRITICAL_THRESHOLD=85    # More conservative
```

---

## 📚 Documentation

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Project Structure

```
TutorlyAI/
├── fal_api.py                 # Main FastAPI application
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Docker services configuration
├── .env                       # Environment variables
├── README.md                  # This file
│
├── books/                     # PDF textbooks (9-12 grades)
│   ├── matematik_9_sinif.pdf
│   ├── fizik_10_sinif.pdf
│   └── ...
│
├── chroma_db_v2/             # ChromaDB persistent storage
│   └── ...
│
├── logs/                      # Application logs
│   └── tutorlyai.log
│
└── tools/                     # Core modules
    ├── initalize_rag_system.py      # RAG initialization
    ├── hybrid_retriever.py          # Hybrid search implementation
    ├── similarity_cache.py          # Similarity-based cache
    ├── redis_client.py              # Redis client wrapper
    ├── redis_cache_adapters.py      # Cache adapters
    ├── get_search_plan.py           # Query planning
    ├── classes.py                   # Pydantic models
    ├── generate_quiz.py             # Quiz generation
    ├── generate_stream.py           # Streaming utilities
    ├── resilience_utils.py          # Circuit breaker, retry
    ├── security_utils.py            # Security features
    ├── performance_utils.py         # Performance monitoring
    ├── database_pool.py             # Database pooling
    ├── file_processing.py           # PDF processing
    ├── subject_normalizer.py        # Subject name normalization
    ├── parse_filename.py            # PDF filename parsing
    ├── system_prompt.py             # System prompts
    └── quiz_prompts.py              # Quiz generation prompts
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update README for API changes

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Fal.ai** for providing the AI model gateway
- **Google** for Gemini 2.5 Flash model
- **LangChain** for RAG framework
- **ChromaDB** for vector database
- **FastAPI** for web framework
- **Redis** for caching infrastructure

---

## 📞 Support

For issues, questions, or suggestions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/TutorlyAI/issues)
- Email: support@tutorlyai.com

---

## 🗺️ Roadmap

### v2.3.0 (Planned)
- [ ] PostgreSQL integration for metadata
- [ ] Multi-language support (English, Arabic)
- [ ] Voice input/output support
- [ ] Mobile app (React Native)
- [ ] Admin dashboard

### v2.4.0 (Future)
- [ ] Fine-tuned models for Turkish education
- [ ] Personalized learning paths
- [ ] Student progress tracking
- [ ] Interactive exercises
- [ ] Video content generation

---

**Built with ❤️ for education | TutorlyAI v2.2.0**