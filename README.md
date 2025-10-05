# 🎓 TutorlyAI - AI Educational Backend for Android

**TutorlyAI**, Android eğitim uygulamanız için geliştirilmiş yapay zeka destekli backend sistemidir. RAG (Retrieval-Augmented Generation) teknolojisi, akıllı önbellekleme, quiz oluşturma, İngilizce öğrenme ve görsel üretimi özellikleri sunar.

---

## 🌟 Sistem Mimarisi

### 📚 Kitap İşleme Sistemi

#### 1. PDF Kitapların Yüklenmesi
```
books/
├── 9_sinif_matematik_unite_1_sayilar.pdf
├── 9_sinif_fizik_unite_2_kuvvet_ve_hareket.pdf
├── 10_sinif_kimya_unite_1_etkilesim.pdf
└── ...
```

**Dosya Adı Formatı:**
- `{sinif}_sinif_{ders}_unite_{unite_no}_{konu}.pdf`
- Örnek: `9_sinif_matematik_unite_1_sayilar.pdf`

**İşleme Süreci:**
1. **PDF Okuma**: PyPDF2 ile her sayfa okunur
2. **Metin Çıkarma**: Sayfa içeriği text'e dönüştürülür
3. **Metadata Çıkarma**: Dosya adından sınıf, ders, ünite, konu bilgisi alınır
4. **Chunk'lara Bölme**: 800 karakter chunk'lar, 100 karakter overlap ile
5. **Embedding Oluşturma**: Her chunk için 384 boyutlu vektör
6. **ChromaDB'ye Kaydetme**: Vector database'e indexleme

**Örnek Metadata:**
```python
{
    "source": "9_sinif_matematik_unite_1_sayilar",
    "filename": "9_sinif_matematik_unite_1_sayilar.pdf",
    "sinif": 9,
    "ders": "matematik",
    "unite": 1,
    "konu": "sayilar",
    "konu_slug": "sayilar",
    "chunk_id": 0,
    "chunk_length": 756
}
```

---

### 🧠 RAG (Retrieval-Augmented Generation) Sistemi

#### Hibrit Arama Stratejisi

**1. Semantic Search (Anlam Bazlı Arama)**
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Embedding Boyutu**: 384 dimension
- **Benzerlik Metriği**: Cosine similarity
- **Nasıl Çalışır**: 
  - Kullanıcı sorusu embedding'e dönüştürülür
  - ChromaDB'de en yakın vektörler bulunur
  - Anlamsal olarak benzer içerikler döner

**2. BM25 Search (Kelime Bazlı Arama)**
- **Algoritma**: Best Matching 25 (Okapi BM25)
- **Nasıl Çalışır**:
  - Kelime frekansı analizi
  - TF-IDF benzeri skorlama
  - Nadir kelimelere daha fazla ağırlık
  - Tam kelime eşleşmeleri için ideal

**3. Hibrit Skorlama**
```python
hybrid_score = (semantic_weight * semantic_score) + (keyword_weight * bm25_score)
# Varsayılan: 0.7 * semantic + 0.3 * bm25
```

**Örnek Arama Akışı:**
```
Kullanıcı Sorusu: "atom nedir ve yapısı nasıldır?"

1. Query Planning (API-1):
   - Soru analiz edilir
   - Filtreler çıkarılır: {sinif: 9, ders: "kimya"}
   - Optimize edilmiş query: "atom yapısı"

2. Semantic Search:
   - Query embedding: [0.23, -0.45, 0.67, ...]
   - ChromaDB'de arama
   - Sonuç: 15 doküman (score: 0.85-0.45)

3. BM25 Search:
   - Kelime analizi: ["atom", "yapısı"]
   - BM25 skorlama
   - Sonuç: 12 doküman (score: 8.5-2.1)

4. Hibrit Birleştirme:
   - Her doküman için: 0.7*semantic + 0.3*bm25
   - Threshold filtreleme: score >= 0.25
   - Final: 8 doküman

5. Context Oluşturma:
   - En iyi 5 doküman seçilir
   - Kaynak bilgileriyle birleştirilir
   - AI modeline gönderilir

6. Cevap Üretimi (API-2):
   - Fal.ai Gemini 2.5 Flash
   - Context + Soru → Cevap
   - Kaynak referanslarıyla döner
```

---

### 💾 Akıllı Önbellekleme Sistemi

#### 1. Redis Multi-Database Mimarisi

**4 Farklı Cache Database:**

```
Redis DB 0: Query Cache
├── Key: "query:{hash}"
├── Value: Arama sonuçları (dokümanlar)
├── TTL: 300 saniye (5 dakika)
└── Amaç: Aynı sorguları tekrar aramaktan kaçınma

Redis DB 1: Performance Cache
├── Key: "generate_response:{hash}"
├── Value: Final AI cevabı
├── TTL: 60 saniye (1 dakika)
└── Amaç: Aynı soruya hızlı cevap

Redis DB 2: BM25 Cache
├── Key: "bm25_index"
├── Value: BM25 index (pickle)
├── TTL: 3600 saniye (1 saat)
└── Amaç: BM25 index'i her seferinde oluşturmama

Redis DB 3: Session Cache
├── Key: "similarity:{cache_type}:metadata"
├── Value: Query metadata + embeddings
├── TTL: 900 saniye (15 dakika)
└── Amaç: Benzer sorular için cache hit
```

#### 2. Similarity-Based Cache (Benzerlik Önbelleği)

**En İleri Özellik!**

**Nasıl Çalışır:**
```python
# Örnek 1: İlk Soru
Soru: "atom nedir?"
→ Cache'de yok
→ RAG sistemi çalışır
→ Cevap üretilir
→ Cache'e kaydedilir:
   - Query: "atom nedir"
   - Embedding: [0.23, -0.45, 0.67, ...]
   - Response: "Atom, maddenin en küçük..."
   - Filters: {sinif: 9, ders: "kimya"}

# Örnek 2: Benzer Soru (5 dakika sonra)
Soru: "atom ne demek?"
→ Cache'de exact match yok
→ Similarity search başlar:
   - Yeni soru embedding: [0.24, -0.44, 0.68, ...]
   - Cosine similarity: 0.87 (threshold: 0.80)
   - CACHE HIT! ✅
→ Önceki cevap döner (RAG + AI çalışmaz!)

# Örnek 3: Farklı Soru
Soru: "hücre nedir?"
→ Similarity: 0.23 (threshold altı)
→ CACHE MISS
→ RAG sistemi çalışır
```

**Avantajları:**
- %80 benzerlik yeterli (exact match gerekmez)
- "atom nedir" = "atom ne demek" = "atomun tanımı" → Hepsi aynı cache
- Farklı filtreler için ayrı cache (9. sınıf ≠ 10. sınıf)
- Embedding tabanlı, dil bağımsız

**Cache Hit Senaryoları:**
```
✅ "fonksiyon nedir" → "fonksiyon ne demek" (similarity: 0.92)
✅ "newton yasaları" → "newton'un hareket yasaları" (similarity: 0.85)
✅ "present perfect" → "present perfect tense" (similarity: 0.88)
❌ "atom" → "hücre" (similarity: 0.31)
❌ "9. sınıf matematik" → "10. sınıf matematik" (farklı filtre)
```

#### 3. Cache Akış Diyagramı

```
┌─────────────────────────────────────────────────────────────┐
│                    Kullanıcı Sorusu                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  1. Exact Match Cache Check   │
         │  (Redis DB 1)                 │
         └───────────┬───────────────────┘
                     │
            ┌────────┴────────┐
            │                 │
         HIT│                 │MISS
            ▼                 ▼
    ┌──────────────┐  ┌──────────────────────┐
    │ Return Cache │  │ 2. Similarity Check  │
    │   Response   │  │    (Redis DB 3)      │
    └──────────────┘  └──────┬───────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                 HIT│                 │MISS
                    ▼                 ▼
            ┌──────────────┐  ┌──────────────────┐
            │ Return Cache │  │ 3. Query Cache   │
            │   Response   │  │    (Redis DB 0)  │
            └──────────────┘  └──────┬───────────┘
                                     │
                            ┌────────┴────────┐
                            │                 │
                         HIT│                 │MISS
                            ▼                 ▼
                    ┌──────────────┐  ┌──────────────────┐
                    │ Use Cached   │  │ 4. RAG Search    │
                    │  Documents   │  │    (ChromaDB)    │
                    └──────┬───────┘  └──────┬───────────┘
                           │                 │
                           └────────┬────────┘
                                    ▼
                        ┌───────────────────────┐
                        │ 5. AI Model (Fal.ai) │
                        │    Generate Answer    │
                        └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │ 6. Cache Response     │
                        │    (All Layers)       │
                        └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   Return to User      │
                        └───────────────────────┘
```

---

### 🔒 Güvenlik Sistemi

#### 1. API Key Doğrulama
```python
# Her istekte kontrol edilir
X-API-Key: api_key

# .env dosyasındaki VALIDATE_API_KEY ile karşılaştırılır
# Eşleşmezse: 401 Unauthorized
```

#### 2. Input Sanitization (Girdi Temizleme)

**SQL Injection Koruması:**
```python
# Tehlikeli: "SELECT * FROM users WHERE id=1"
# Engellenir: union, select, drop, delete, exec
```

**XSS Koruması:**
```python
# Tehlikeli: "<script>alert('xss')</script>"
# Engellenir: <script>, javascript:, onerror=
```

**Command Injection Koruması:**
```python
# Tehlikeli: "rm -rf /"
# Engellenir: ;, |, &&, rm, del, wget
```

**AI Prompt için Özel Sanitization:**
- SQL/Command injection kontrol YOK (veritabanı/sistem komutu yok)
- Sadece açık script tag'leri engellenir
- Kullanıcı soruları doğal dilde olabilir

#### 3. Rate Limiting

**Kullanıcı Bazlı:**
- Her kullanıcı: Max 5 concurrent request
- Global: Max 50 concurrent request

**IP Bazlı:**
- Dakikada 100 request (opsiyonel)

#### 4. Security Headers
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
```

---

### ⚡ Performans Optimizasyonları

#### 1. Memory Monitoring
```python
# Sürekli izlenir
Current: 512 MB
Warning: 80% (640 MB)
Critical: 90% (720 MB)

# Critical'de yeni request reddedilir
```

#### 2. Circuit Breaker Pattern

**Fal.ai API Koruması:**
```
Normal (CLOSED):
├── Her istek gönderilir
└── Başarı sayılır

Failure Threshold (5 hata):
├── Circuit OPEN olur
├── 60 saniye bekler
└── Fallback response döner

Recovery (HALF_OPEN):
├── Test istekleri gönderilir
├── 2 başarı → CLOSED
└── 1 hata → OPEN
```

**Fallback Responses:**
```json
{
  "generated_text": "Şu anda AI servisi kullanılamıyor. Lütfen daha sonra tekrar deneyin.",
  "fallback": true,
  "circuit_state": "open"
}
```

#### 3. Retry Mechanism

**Exponential Backoff:**
```
1. Deneme: Hemen
2. Deneme: 1 saniye sonra
3. Deneme: 2 saniye sonra
4. Deneme: 4 saniye sonra
Max: 10 saniye

# Jitter eklenir (rastgele gecikme)
# Thundering herd problemini önler
```

---

### 📊 İstatistikler ve Monitoring

#### Performance Stats
```bash
GET /performance/stats

Response:
{
  "concurrency": {
    "concurrent_requests": 3,
    "max_concurrent_per_user": 5,
    "max_concurrent_global": 50
  },
  "memory": {
    "current_memory_mb": 512.5,
    "warning_threshold": 80,
    "critical_threshold": 90
  },
  "cache": {
    "total_hits": 1250,
    "total_misses": 350,
    "hit_rate_percent": 78.1
  }
}
```

#### Resilience Stats
```bash
GET /resilience/stats

Response:
{
  "circuit_breaker": {
    "state": "closed",
    "failure_count": 0,
    "success_count": 1523
  },
  "retry": {
    "total_retries": 45,
    "successful_retries": 42
  },
  "rate_limit": {
    "requests_per_minute": 87,
    "limit": 100
  }
}
```

---

## 🚀 Kurulum ve Çalıştırma

### Adım 1: Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

**Yüklenen Paketler:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `chromadb` - Vector database
- `sentence-transformers` - Embedding model
- `langchain` - RAG framework
- `redis` - Cache sistemi
- `rank-bm25` - BM25 algoritması
- `fal-client` - AI model client

### Adım 2: .env Dosyasını Ayarla
```bash
# Zorunlu
FAL_KEY=your_fal_api_key_here
VALIDATE_API_KEY=your_api_key_for_android

# Redis 
REDIS_HOST=localhost
REDIS_PORT=6379

# Cache ayarları
REDIS_SIMILARITY_THRESHOLD=0.80
REDIS_QUERY_CACHE_TTL=300
REDIS_FINAL_RESPONSE_TTL=900

# RAG ayarları
SEMANTIC_WEIGHT=0.7
KEYWORD_WEIGHT=0.3
SCORE_THRESHOLD=0.25
```

### Adım 3: Redis'i Başlat (Opsiyonel)
```bash
# Docker ile
docker-compose up -d redis

# Manuel
redis-server

# Redis olmadan da çalışır (cache devre dışı)
```

### Adım 4: Kitapları Hazırla
```bash
# books/ klasörüne PDF'leri koy
# Format: 9_sinif_matematik_unite_1_sayilar.pdf
```

### Adım 5: Uygulamayı Çalıştır
```bash
python fal_api.py
```

**İlk Çalıştırma:**
```
[RAG] Sistem başlatılıyor...
[RAG] Embedding model yüklendi
[RAG] ChromaDB bağlandı
[RAG] Hibrit retriever başlatıldı
[RAG] 75 PDF dosyası bulundu
[RAG] İşleniyor: 9_sinif_matematik_unite_1_sayilar.pdf
[RAG] 9_sinif_matematik_unite_1_sayilar.pdf: 45 chunk oluşturuldu
...
[RAG] BAŞARILI: 75 dosya, 3420 chunk yüklendi
[✓] RAG sistemi başlatıldı
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## 📡 API Kullanımı

### 1. RAG Soru-Cevap

**Request:**
```bash
POST /generate
Content-Type: application/json
X-API-Key: api_key

{
  "prompt": "atom nedir ve yapısı nasıldır?",
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
  "generated_text": "Atom, maddenin en küçük yapı taşıdır...",
  "search_plan": {
    "query": "atom yapısı",
    "filters": {
      "sinif": 9,
      "ders": "kimya"
    }
  },
  "found_documents": 5,
  "sources": [
    "9_sinif_kimya_unite_1_etkilesim",
    "9_sinif_kimya_unite_2_cesitlilik"
  ],
  "search_method": "hybrid_search",
  "search_config": {
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "score_threshold": 0.25
  },
  "search_details": {
    "average_hybrid_score": 0.78,
    "average_semantic_score": 0.82,
    "average_bm25_score": 0.65,
    "score_breakdown": [
      {
        "source": "9_sinif_kimya_unite_1_etkilesim",
        "hybrid_score": 0.85,
        "semantic_score": 0.89,
        "bm25_score": 0.72
      }
    ]
  }
}
```

### 2. Quiz Oluşturma

**Request:**
```bash
POST /quiz/generate
X-API-Key: api_key

{
  "sinif": 9,
  "ders": "matematik",
  "konu": "Fonksiyonlar",
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
    "sorular": [
      {
        "soru_metni": "f(x) = 2x + 3 fonksiyonunda f(5) değeri kaçtır?",
        "secenekler": ["11", "13", "15", "17"],
        "dogru_cevap": "B",
        "aciklama": "f(5) = 2(5) + 3 = 13"
      }
    ],
    "toplam_soru": 5,
    "zorluk_seviyesi": "orta"
  }
}
```

### 3. İngilizce Öğrenme

**Seviye Sistemi:**
- **A1**: Başlangıç (temel kelimeler, present simple)
- **A2**: Temel (günlük rutinler, past simple)
- **B1**: Orta (present perfect, conditionals)
- **B2**: Üst-Orta (mixed conditionals, idioms)
- **C1**: İleri (inversion, cleft sentences)
- **C2**: Usta (native speaker seviyesi)

**Request:**
```bash
POST /english/generate
X-API-Key: api_key

{
  "prompt": "ingilizce seviyesi: b1. Explain the difference between present perfect and past simple",
  "max_tokens": 800
}
```

**Response:**
```json
{
  "generated_text": "Present Perfect ve Past Simple arasındaki fark...",
  "detected_level": "B1",
  "system_prompt_used": "B1 seviyesi için özelleştirilmiş prompt",
  "clean_prompt": "Explain the difference between present perfect and past simple"
}
```

### 4. Görsel Oluşturma

**Request:**
```bash
POST /generate/image
X-API-Key: api_key

{
  "prompt": "matematik fonksiyon grafiği y=2x+3",
  "workflow_id": "workflows/halillllibrahim58/teach-img-model"
}
```

**Response:**
```json
{
  "success": true,
  "image_url": "https://fal.media/files/...",
  "workflow_id": "workflows/halillllibrahim58/teach-img-model",
  "prompt": "matematik fonksiyon grafiği y=2x+3",
  "generated_at": "2024-01-01T12:00:00"
}
```

---

## 📱 Android Entegrasyonu

### Retrofit Interface
```kotlin
interface TutorlyAPI {
    @POST("generate")
    suspend fun generateAnswer(
        @Header("X-API-Key") apiKey: String,
        @Header("X-User-ID") userId: String,
        @Body request: GenerateRequest
    ): GenerateResponse
    
    @POST("quiz/generate")
    suspend fun generateQuiz(
        @Header("X-API-Key") apiKey: String,
        @Body request: QuizRequest
    ): QuizResponse
    
    @POST("english/generate")
    suspend fun generateEnglish(
        @Header("X-API-Key") apiKey: String,
        @Body request: EnglishRequest
    ): EnglishResponse
    
    @POST("generate/image")
    suspend fun generateImage(
        @Header("X-API-Key") apiKey: String,
        @Body request: ImageRequest
    ): ImageResponse
}
```

### Kullanım Örneği
```kotlin
// Firebase user ID ile
val userId = FirebaseAuth.getInstance().currentUser?.uid ?: ""

// RAG soru-cevap
val response = api.generateAnswer(
    apiKey = "api_key",
    userId = userId,
    request = GenerateRequest(
        prompt = "atom nedir?",
        maxTokens = 1000,
        useHybrid = true
    )
)

// Quiz oluştur
val quiz = api.generateQuiz(
    apiKey = "api_key",
    request = QuizRequest(
        sinif = 9,
        ders = "matematik",
        konu = "Fonksiyonlar",
        soruSayisi = 5,
        soruTipi = "coktan_secmeli"
    )
)
```

---

## 🔧 Gelişmiş Özellikler

### 1. Manuel Arama
```bash
POST /search
X-API-Key: api_key

{
  "query": "atom yapısı",
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

### 2. RAG Durumu
```bash
GET /rag-status

Response:
{
  "status": "active",
  "documents_count": 3420,
  "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "vectorstore": "ChromaDB",
  "sample_metadata": [...]
}
```

### 3. Debug Bilgileri
```bash
GET /debug

Response:
{
  "total_documents": 3420,
  "class_distribution": {
    "9": 1250,
    "10": 1180,
    "11": 990
  },
  "subject_distribution": {
    "matematik": 850,
    "fizik": 720,
    "kimya": 680
  }
}
```

---

## 🐛 Sorun Giderme

### Redis Bağlantı Hatası
```bash
# Redis opsiyonel, uygulama çalışır
# Sadece cache devre dışı kalır

# Redis başlatmak için:
docker-compose up -d redis
```

### Port 8000 Kullanımda
```bash
# Farklı port kullan
uvicorn fal_api:app --port 8001
```

### Kitaplar Yüklenmiyor
```bash
# Dosya adı formatını kontrol et
# Doğru: 9_sinif_matematik_unite_1_sayilar.pdf
# Yanlış: matematik_9_sinif.pdf

# Log'ları kontrol et
tail -f logs/tutorly_api.log
```

### Memory Hatası
```bash
# Chunk size'ı küçült
# tools/initalize_rag_system.py
chunk_size = 500  # 800'den 500'e düşür
```

---

## 📊 Performans Metrikleri

### Cache Hit Rates
- **Similarity Cache**: %85-90
- **Query Cache**: %70-75
- **Response Cache**: %60-65
- **BM25 Cache**: %95+

### Response Times
- **Cache Hit**: 50-100ms
- **Cache Miss + RAG**: 500-1000ms
- **Cache Miss + RAG + AI**: 2000-4000ms

### Memory Usage
- **Idle**: 200-300 MB
- **Active (10 users)**: 500-700 MB
- **Peak (50 users)**: 1-1.5 GB

---

## 📄 Lisans

Apache License 2.0

---

## 🙏 Teşekkürler

- **Fal.ai** - AI model gateway
- **Google** - Gemini 2.5 Flash model
- **LangChain** - RAG framework
- **ChromaDB** - Vector database
- **Redis** - Cache infrastructure

---

**Android eğitim uygulamanız için hazır! 🚀**

**API Docs:** http://localhost:8000/docs
