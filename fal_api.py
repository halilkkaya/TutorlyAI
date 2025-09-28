from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError
import os
import asyncio
from dotenv import load_dotenv
import traceback
import fal_client
from datetime import datetime
import glob
import re
import logging
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from tools.get_search_plan import get_search_plan
from tools.generate_stream import generate_stream
from tools.generate_quiz import generate_quiz
from tools.classes import TextGenerationRequest, QuizRequest, EnglishLearningRequest, EnglishLearningResponse, ImageGenerationRequest, ImageGenerationResponse
from tools.initalize_rag_system import initialize_rag_system, should_load_books, load_books_async, search_books_enhanced, search_books_enhanced_async
from tools.resilience_utils import resilient_client, create_fallback_response, FALLBACK_RESPONSES, CircuitState
from tools.security_utils import (
    security_validator, api_key_manager, create_security_headers,
    get_client_ip, SecurityViolation
)
from tools.performance_utils import (
    request_limiter, memory_monitor, response_cache, get_performance_stats
)
from tools.database_pool import get_database_stats

# Ortam değişkenlerini yükle
load_dotenv()

# FastAPI uygulaması oluştur
app = FastAPI(
    title="TutorlyAI - Enhanced RAG, Quiz & English Learning API",
    description="Gelişmiş RAG sistemi, Quiz oluşturma ve İngilizce öğrenme API'si - fal.ai Gemini 2.5 Flash ile güçlendirilmiş",
    version="2.2.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fal.ai API anahtarını kontrol et
FAL_KEY = os.getenv("FAL_KEY")

if not FAL_KEY:
    raise ValueError("FAL_KEY ortam değişkeni ayarlanmamış!")

# Model adını belirle
MODEL_NAME = "google/gemini-2.5-flash"
FAL_MODEL_GATEWAY = "fal-ai/any-llm"

# Security setup
security = HTTPBearer(auto_error=False)

# Sabit API key .env dosyasından al
VALIDATE_API_KEY = os.getenv("VALIDATE_API_KEY")
if not VALIDATE_API_KEY:
    raise ValueError("VALIDATE_API_KEY ortam değişkeni ayarlanmamış!")

# Test ekipleri için sabit key kullan
DEFAULT_API_KEY = VALIDATE_API_KEY

# RAG sistemi için global değişkenler
vectorstore = None
embedding_model = None
text_splitter = None

# İngilizce seviyeleri için system promptları
LEVEL_SYSTEM_PROMPTS = {
    "a1": """Sen bir İngilizce öğretmenisin ve A1 seviyesindeki öğrencilerle çalışıyorsun. 
A1 seviyesi için:
- Çok basit kelimeler ve cümleler kullan
- Present Simple tense'i tercih et
- Günlük yaşamdan örnekler ver
- Kısa ve net açıklamalar yap
- Temel kelime dağarcığı kullan (merhaba, teşekkürler, aile, renkler, sayılar vb.)
- Karmaşık gramer yapılarından kaçın
Türkçe açıklama yaparken basit dil kullan ve öğrencinin seviyesine uygun örnekler ver.""",

    "a2": """Sen bir İngilizce öğretmenisin ve A2 seviyesindeki öğrencilerle çalışıyorsun.
A2 seviyesi için:
- Temel gramer yapılarını kullan (Present Simple, Present Continuous, Past Simple)
- Günlük rutinler, alışveriş, aile hakkında konuşmayı destekle
- Basit bağlaçlar kullan (and, but, because)
- Sık kullanılan kelimelerle cümleler kur
- Açık ve yavaş bir şekilde açıkla
- Modal fiiller (can, could, should) gibi temel yapıları tanıt
Öğrencinin günlük yaşamından örnekler vererek öğrenmeyi kolaylaştır.""",

    "b1": """Sen bir İngilizce öğretmenisin ve B1 seviyesindeki öğrencilerle çalışıyorsun.
B1 seviyesi için:
- Orta seviye gramer yapılarını kullan (Present Perfect, Future tenses, Conditionals)
- Soyut konuları açıklayabilir
- Bağlaçları çeşitlendir (although, however, therefore)
- Görüş bildirme ve öneri verme ifadelerini kullan
- Karmaşık metinleri anlamaya yardımcı ol
- Passive voice gibi yapıları tanıt
- İş, eğitim, seyahat gibi konularda yardım et
Öğrencinin kendi fikirlerini İngilizce ifade etmesini teşvik et.""",

    "b2": """Sen bir İngilizce öğretmenisin ve B2 seviyesindeki öğrencilerle çalışıyorsun.
B2 seviyesi için:
- Gelişmiş gramer yapılarını kullan (Past Perfect, Mixed Conditionals, Subjunctive)
- Karmaşık metinleri ve soyut kavramları açıkla
- İdiomatic ifadeler ve phrasal verb'ler kullan
- Formal ve informal dil arasındaki farkları göster
- Akademik ve profesyonel konularda destek ver
- Eleştirel düşünceyi teşvik et
- Nüanslı ifadeleri açıkla
Öğrencinin akıcılığını artırmaya ve güvenini geliştirmeye odaklan.""",

    "c1": """Sen bir İngilizce öğretmenisin ve C1 seviyesindeki öğrencilerle çalışıyorsun.
C1 seviyesi için:
- İleri seviye gramer yapılarını ustaca kullan
- Karmaşık ve akademik konuları detaylı açıkla
- Geniş kelime dağarcığı ve idiom kullan
- Inversion, cleft sentences gibi yapıları kullan
- Akademik yazım ve sunum becerilerinde destek ver
- Kültürel referansları açıkla
- İnce nüansları ve çok anlamlı ifadeleri göster
- Professional English kullanımında rehberlik et
Öğrencinin native speaker seviyesine yaklaşmasına yardım et.""",

    "c2": """Sen bir İngilizce öğretmenisin ve C2 seviyesindeki öğrencilerle çalışıyorsun.
C2 seviyesi için:
- Native speaker seviyesinde dil kullan
- En karmaşık gramer yapılarını ve stilistik özelliklerini göster
- Akademik, sanatsal ve profesyonel konularda derinlemesine destek ver
- Çok gelişmiş kelime dağarcığı ve sophisticated ifadeler kullan
- Edebiyat, felsefe, bilim gibi üst düzey konularda tartışma
- Regional dialects ve language varieties hakkında bilgi ver
- Çeviri ve interpretation becerilerinde destek
- Style, register ve tone konularında uzman rehberlik
Öğrencinin dili mükemmel seviyede kullanmasına yardım et."""
}

def detect_english_level(prompt: str) -> str:
    """Prompt'tan İngilizce seviyesini algılar"""
    # "ingilizce seviyesi: c1" formatındaki metni arıyoruz
    level_pattern = r"ingilizce seviyesi:\s*([a-c][1-2])"
    match = re.search(level_pattern, prompt.lower())
    
    if match:
        level = match.group(1).lower()
        logger.info(f"[ENGLISH] Seviye algılandı: {level}")
        return level
    
    # Varsayılan seviye
    logger.warning("[ENGLISH] Seviye algılanamadı, varsayılan B1 kullanılıyor")
    return "b1"

def clean_prompt(prompt: str) -> str:
    """Prompt'tan seviye bilgisini temizler"""
    # "ingilizce seviyesi: c1" kısmını çıkar
    level_pattern = r"ingilizce seviyesi:\s*[a-c][1-2]\.?\s*"
    cleaned = re.sub(level_pattern, "", prompt, flags=re.IGNORECASE)
    return cleaned.strip()

# Security middleware
async def validate_api_key(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """API key validation middleware"""
    client_ip = get_client_ip(request)

    # API key kontrolü
    api_key = None
    if credentials:
        api_key = credentials.credentials

    # Header'dan da kontrol et
    if not api_key:
        api_key = request.headers.get("X-API-Key")

    # Query parameter'dan da kontrol et (development için)
    if not api_key:
        api_key = request.query_params.get("api_key")

    # Sabit key kontrolü (.env dosyasındaki VALIDATE_API_KEY)
    if not api_key or api_key != VALIDATE_API_KEY:
        violation = SecurityViolation(
            violation_type="invalid_api_key",
            details=f"Invalid or missing API key from {client_ip}",
            client_ip=client_ip,
            timestamp=datetime.now(),
            request_path=str(request.url.path),
            severity="medium"
        )
        security_validator.violations.append(violation)
        logger.warning(f"[SECURITY] Invalid API key from {client_ip}: {api_key[:8] if api_key else 'None'}...")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return api_key

async def sanitize_request_data(request: Request):
    """Request data'sını sanitize eder"""
    client_ip = get_client_ip(request)

    # Request size kontrolü
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > security_validator.config.max_request_size:
        violation = SecurityViolation(
            violation_type="request_too_large",
            details=f"Request too large: {content_length} bytes from {client_ip}",
            client_ip=client_ip,
            timestamp=datetime.now(),
            request_path=str(request.url.path),
            severity="medium"
        )
        security_validator.violations.append(violation)
        raise HTTPException(status_code=413, detail="Request too large")

@app.middleware("http")
async def android_compatibility_middleware(request: Request, call_next):
    """Android uygulaması uyumluluğu için özel middleware"""
    client_ip = get_client_ip(request)

    # Android User-Agent kontrolü
    user_agent = request.headers.get("User-Agent", "")
    is_android = "android" in user_agent.lower() or "okhttp" in user_agent.lower()

    # Android uygulamasından gelen istekler için özel handling
    if is_android:
        logger.info(f"[ANDROID] Request from Android app: {user_agent}")

        # Android app'e özel header'lar ekle
        response = await call_next(request)

        # Android uyumlu response headers
        response.headers["X-Android-Compatible"] = "true"
        response.headers["X-API-Version"] = "2.2.0"

        # CORS headers Android için optimize et
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key, X-User-ID"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"

        return response
    else:
        return await call_next(request)

@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Performance and concurrency middleware"""
    client_ip = get_client_ip(request)
    request_id = f"{client_ip}_{int(time.time() * 1000)}"

    # Android uygulamasından User ID'yi al
    user_id = request.headers.get("X-User-ID", client_ip)

    # Alternatif olarak query parameter'dan da al (Android'de header gönderme sorunu varsa)
    if not user_id or user_id == client_ip:
        user_id = request.query_params.get("user_id", client_ip)

    start_time = time.time()

    # Memory kontrolü
    if not memory_monitor.check_memory_limits():
        logger.error(f"[PERFORMANCE] Memory limit exceeded, rejecting request from {client_ip}")
        raise HTTPException(status_code=503, detail="Server overloaded, please try again later")

    # Concurrent request limiting
    try:
        async with request_limiter.limit_request(request_id, user_id):
            response = await call_next(request)

            # Performance headers ekle
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Concurrent-Requests"] = str(request_limiter.stats["concurrent_requests"])

            return response

    except HTTPException as e:
        # Performance related HTTP exceptions
        raise e
    except Exception as e:
        # Diğer hataları logla
        logger.error(f"[PERFORMANCE] Unexpected error in request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Global security middleware"""
    client_ip = get_client_ip(request)

    try:
        # Request data sanitization
        await sanitize_request_data(request)

        response = await call_next(request)

        # Security headers ekle
        security_headers = create_security_headers()
        for key, value in security_headers.items():
            response.headers[key] = value

        return response

    except HTTPException as e:
        # HTTP Exception'ları re-raise et
        raise e
    except Exception as e:
        # Diğer hataları logla ve güvenli hata mesajı döndür
        logger.error(f"[SECURITY] Unexpected error from {client_ip}: {str(e)}")
        violation = SecurityViolation(
            violation_type="unexpected_error",
            details=f"Unexpected error from {client_ip}: {str(e)[:100]}",
            client_ip=client_ip,
            timestamp=datetime.now(),
            request_path=str(request.url.path),
            severity="high"
        )
        security_validator.violations.append(violation)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.options("/{path:path}")
async def handle_options(path: str):
    """CORS preflight requests için OPTIONS handler - Android uyumluluğu"""
    return {
        "message": "CORS preflight OK",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "headers": ["Content-Type", "Authorization", "X-API-Key", "X-User-ID"]
    }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü"""
    performance_stats = get_performance_stats()

    return {
        "status": "healthy",
        "rag_active": vectorstore is not None,
        "timestamp": datetime.now().isoformat(),
        "performance": {
            "concurrent_requests": performance_stats["concurrency"]["concurrent_requests"],
            "memory_usage_mb": performance_stats["memory"]["current_memory_mb"],
            "cache_hit_rate": performance_stats["cache"]["hit_rate_percent"]
        },
        "android_compatible": True,
        "api_version": "2.2.0"
    }

@app.post("/load-books")
async def load_books_endpoint():
    """Kitapları yükler"""
    try:
        success = await load_books_async()
        
        if success:
            return {
                "message": "Kitaplar başarıyla yüklendi",
                "status": "success"
            }
        else:
            return {
                "message": "Kitap yükleme başarısız",
                "status": "failed"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kitap yükleme hatası: {str(e)}")

@app.get("/rag-status")
async def rag_status():
    """RAG sistemi durumu"""
    global vectorstore
    
    if not vectorstore:
        return {"status": "inactive", "message": "RAG sistemi başlatılmamış"}
    
    try:
        collection = vectorstore._collection
        count = collection.count()
        
        # Metadata örnekleri al
        results = collection.peek(limit=5)
        sample_metadata = []
        if results and 'metadatas' in results:
            sample_metadata = results['metadatas']
        
        return {
            "status": "active",
            "documents_count": count,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "vectorstore": "ChromaDB",
            "sample_metadata": sample_metadata[:3]  # İlk 3 örnek
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/search")
async def search_endpoint(request: dict):
    """Manuel arama endpoint'i - Hibrit arama destekli"""
    query = request.get("query", "")
    filters = request.get("filters", {})
    k = request.get("k", 5)
    score_threshold = request.get("score_threshold", 0.3)
    use_hybrid = request.get("use_hybrid", True)
    semantic_weight = request.get("semantic_weight", 0.7)
    keyword_weight = request.get("keyword_weight", 0.3)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query gerekli")
    
    results = await search_books_enhanced_async(
        query=query,
        filters=filters,
        k=k,
        score_threshold=score_threshold,
        use_hybrid=use_hybrid,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight
    )
    
    return {
        "query": query,
        "filters": filters,
        "search_config": {
            "use_hybrid": use_hybrid,
            "semantic_weight": semantic_weight,
            "keyword_weight": keyword_weight,
            "score_threshold": score_threshold
        },
        "results_count": len(results),
        "results": [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
                "hybrid_score": doc.metadata.get('hybrid_score', 0),
                "semantic_score": doc.metadata.get('semantic_score', 0),
                "bm25_score": doc.metadata.get('bm25_score', 0)
            }
            for doc in results
        ]
    }

@app.get("/debug")
async def debug_endpoint():
    """Debug bilgileri"""
    global vectorstore
    
    if not vectorstore:
        return {"error": "RAG sistemi aktif değil"}
    
    try:
        collection = vectorstore._collection
        
        # Toplam doküman sayısı
        total_docs = collection.count()
        
        # Metadata örnekleri
        peek_result = collection.peek(limit=10)
        
        # Sınıf ve ders dağılımını hesapla
        all_metadata = peek_result.get('metadatas', [])
        
        class_dist = {}
        subject_dist = {}
        
        for meta in all_metadata:
            if 'sinif' in meta:
                class_num = meta['sinif']
                class_dist[class_num] = class_dist.get(class_num, 0) + 1
            
            if 'ders' in meta:
                subject = meta['ders']
                subject_dist[subject] = subject_dist.get(subject, 0) + 1
        
        return {
            "total_documents": total_docs,
            "class_distribution": class_dist,
            "subject_distribution": subject_dist,
            "sample_metadata": all_metadata[:5],
            "books_directory_files": len(glob.glob("books/*.pdf"))
        }
        
    except Exception as e:
        return {"error": str(e)}



@app.post("/generate/stream")
async def stream_text(request: TextGenerationRequest, api_key: str = Depends(validate_api_key)):
    """Streaming metin üretme"""
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream"
    )
@app.post("/generate")
async def generate_rag_answer(request: TextGenerationRequest, api_key: str = Depends(validate_api_key)):
    """RAG ile cevap üretir - Score threshold ile"""
    try:
        # Input sanitization (SQL/Command injection kontrolü olmadan - AI prompt için)
        sanitized_prompt = security_validator.sanitize_ai_prompt(request.prompt, "prompt")
        logger.info(f"[GENERATE] Gelen sorgu: '{sanitized_prompt[:50]}...'")

        # 1. Arama planı oluştur
        search_plan = await get_search_plan(sanitized_prompt)
        query = search_plan.get("query", request.prompt)
        filters = search_plan.get("filters", {})
        
        logger.info(f"[GENERATE] Arama planı - Query: '{query}', Filters: {filters}")
        
        # 2. Hibrit arama stratejisi ile doküman bulma (request parametrelerini kullan)
        relevant_docs = await search_books_enhanced_async(
            query=query,
            filters=filters,
            k=request.search_k,
            score_threshold=request.score_threshold,
            use_hybrid=request.use_hybrid,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight
        )
        
        
        if not relevant_docs:
            logger.warning("[GENERATE] Threshold'u geçen doküman bulunamadı")
            
        
        # Sonuç olsun ya da olmasın, modele gönder
        
        # 3. Context oluştur
        if relevant_docs:
            context_text = "\n\n---\n\n".join([
                f"[{doc.metadata.get('source', 'Bilinmeyen')}]\n{doc.page_content}"
                for doc in relevant_docs
            ])
            logger.info(f"[GENERATE] {len(relevant_docs)} kaliteli doküman bulundu")
            
            # 4A. Dökümanlar bulunduğunda
            synthesis_prompt = f"""Sen bir ders kitabı uzmanısın. Aşağıdaki soruyu, verilen ders kitabı metinlerini kullanarak cevapla.

            SORU: {sanitized_prompt}

            DERS KİTABI İÇERİĞİ:
            {context_text}

            KURALLAR:
            1. Sadece verilen kaynaklardaki bilgileri kullan
            2. Kapsamlı ve anlaşılır bir açıklama yap  
            3. Hangi kaynaktan bilgi aldığını belirt
            4. Eğer cevap kaynaklarda yoksa, bunu söyle
            5. Bu kaynaklar hibrit arama sistemi (semantic + keyword) ile seçildi

            CEVAP:"""
        else:
            logger.warning("[GENERATE] Hiç kaliteli doküman bulunamadı, model kendi bilgisiyle cevap verecek")
            
            # 4B. Hiç doküman bulunamadığında
            synthesis_prompt = f"""Sen bir ders kitabı uzmanısın. Kullanıcı soru sordu ancak ders kitaplarında bu konuyla yeterince benzer içerik bulunamadı.

            SORU: {sanitized_prompt}

            DURUM: Ders kitaplarında bu soruyla yeterince benzer içerik bulunamadı.

            KURALLAR:
            1. Eğer bu bir selamlaşma ise normal karşılık ver
            2. Eğer genel bir soru ise kendi bilginle yardım etmeye çalış  
            3. Eğer çok spesifik bir ders konusu ise, daha spesifik terimlerle tekrar sormalarını öner
            4. Kibarca ve yardımsever ol

            CEVAP:"""

        # 5. Modeli çağır ve cevap al (resilience ile)
        fallback_response = await create_fallback_response("text_generation",
                                                          generated_text="Şu anda AI servisi kullanılamıyor. Lütfen daha sonra tekrar deneyin.")

        result = await resilient_client.run_async_with_resilience(
            FAL_MODEL_GATEWAY,
            arguments={
                "model": MODEL_NAME,
                "prompt": synthesis_prompt,
                "max_tokens": 1500,
                "temperature": 0.3,
            },
            fallback_response=fallback_response,
            operation_type="text_generation"
        )
        
        final_answer = result.get("output", "Cevap oluşturulamadı.")
        
        # Hibrit arama detaylarını topla
        search_details = {}
        if relevant_docs:
            search_details = {
                "average_hybrid_score": sum(doc.metadata.get('hybrid_score', 0) for doc in relevant_docs) / len(relevant_docs),
                "average_semantic_score": sum(doc.metadata.get('semantic_score', 0) for doc in relevant_docs) / len(relevant_docs),
                "average_bm25_score": sum(doc.metadata.get('bm25_score', 0) for doc in relevant_docs) / len(relevant_docs),
                "score_breakdown": [
                    {
                        "source": doc.metadata.get('source', 'Bilinmeyen'),
                        "hybrid_score": doc.metadata.get('hybrid_score', 0),
                        "semantic_score": doc.metadata.get('semantic_score', 0),
                        "bm25_score": doc.metadata.get('bm25_score', 0)
                    }
                    for doc in relevant_docs
                ]
            }

        return {
            "generated_text": final_answer,
            "search_plan": search_plan,
            "found_documents": len(relevant_docs),
            "sources": [doc.metadata.get('source', 'Bilinmeyen') for doc in relevant_docs] if relevant_docs else [],
            "search_method": "hybrid_search" if request.use_hybrid else "semantic_only",
            "search_config": {
                "semantic_weight": request.semantic_weight,
                "keyword_weight": request.keyword_weight,
                "score_threshold": request.score_threshold,
                "search_k": request.search_k,
                "hybrid_enabled": request.use_hybrid
            },
            "search_details": search_details
        }
        
    except Exception as e:
        logger.error(f"[GENERATE ERROR] {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quiz/generate")
async def generate_quiz_endpoint(request: QuizRequest, api_key: str = Depends(validate_api_key)):
    """Quiz soruları oluştur"""
    try:
        logger.info(f"[QUIZ API] Quiz generation request: {request.soru_sayisi} {request.soru_tipi} soru")
        
        quiz_response = await generate_quiz(request)
        
        return {
            "success": True,
            "message": "Quiz başarıyla oluşturuldu",
            "data": quiz_response.model_dump()
        }
        
    except ValidationError as e:
        logger.error(f"[QUIZ API] Pydantic validation error: {str(e)}")
        # Pydantic error'ını güzel formata çevir
        error_details = []
        for error in e.errors():
            field = " -> ".join(str(x) for x in error["loc"])
            message = error["msg"]
            error_details.append(f"{field}: {message}")
        
        raise HTTPException(
            status_code=400, 
            detail={
                "error": "Validation failed",
                "details": error_details,
                "message": "Gönderilen veriler geçersiz"
            }
        )
    except ValueError as e:
        logger.error(f"[QUIZ API] Generation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[QUIZ API ERROR] {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Quiz oluşturma hatası: {str(e)}")


@app.get("/quiz/info")
async def quiz_info():
    """Quiz sistemi hakkında bilgi"""
    return {
        "supported_question_types": ["coktan_secmeli", "acik_uclu"],
        "supported_grades": list(range(1, 13)),  # 1-12
        "max_questions": 10,  # Güncellenmiş limit
        "difficulty_levels": ["kolay", "orta", "zor"],
        "supported_languages": ["tr", "en"],
        "model": MODEL_NAME,
        "version": "2.0.0",
        "validation": {
            "grade_range": "1-12",
            "questions_range": "1-10", 
            "question_types": ["coktan_secmeli", "acik_uclu"],
            "subject_normalization": True,
            "pydantic_validation": True
        }
    }

@app.get("/models")
async def get_model_info():
    """Model bilgileri"""
    return {
        "service": "fal.ai",
        "gateway_model": FAL_MODEL_GATEWAY,
        "configured_llm": MODEL_NAME
    }

# ========================
# İNGİLİZCE ÖĞRENME API'Sİ
# ========================

@app.get("/english/levels")
async def get_english_levels():
    """Desteklenen İngilizce seviyelerini döndürür"""
    return {
        "levels": {
            "A1": "Başlangıç - Temel kelimeler ve basit cümleler",
            "A2": "Temel - Günlük konular ve basit gramer",
            "B1": "Orta - Karmaşık konular ve gelişmiş gramer",
            "B2": "Üst-Orta - Soyut konular ve akademik dil",
            "C1": "İleri - Profesyonel ve akademik İngilizce",
            "C2": "Usta - Native speaker seviyesi"
        },
        "usage": "Prompt'unuzun başına 'ingilizce seviyesi: c1' şeklinde seviyenizi belirtin",
        "workflow": "workflows/halillllibrahim58/eng-teach"
    }

@app.post("/english/generate", response_model=EnglishLearningResponse)
async def generate_english_content(request: EnglishLearningRequest, api_key: str = Depends(validate_api_key)):
    """İngilizce öğrenme içeriği üretir - seviyeye göre"""
    try:
        logger.info(f"[ENGLISH] Gelen request: '{request.prompt[:50]}...'")
        
        # Input sanitization (SQL injection kontrolü olmadan - AI prompt için)
        sanitized_prompt = security_validator.sanitize_ai_prompt(request.prompt, "english_prompt")

        # Seviyeyi algıla
        detected_level = detect_english_level(sanitized_prompt)

        # Prompt'u temizle
        clean_user_prompt = clean_prompt(sanitized_prompt)
        
        # Uygun system prompt'u seç
        system_prompt = LEVEL_SYSTEM_PROMPTS.get(detected_level, LEVEL_SYSTEM_PROMPTS["b1"])
        
        # Final prompt'u oluştur
        final_prompt = f"{system_prompt}\n\nÖğrenci sorusu: {clean_user_prompt}"
        
        logger.info(f"[ENGLISH] Seviye: {detected_level.upper()}, Temizlenmiş prompt: {clean_user_prompt[:50]}...")
        
        # FAL client ile içerik üret (resilience ile)
        fallback_response = await create_fallback_response("text_generation",
                                                          output="İngilizce öğrenme servisi şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.")

        result = await resilient_client.run_async_with_resilience(
            "workflows/halillllibrahim58/eng-teach",
            arguments={
                "prompt": final_prompt,
                "system_prompt": system_prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            },
            fallback_response=fallback_response,
            operation_type="english_learning"
        )
        
        generated_text = result.get("output", "İçerik üretilemedi.")
        
        return EnglishLearningResponse(
            generated_text=generated_text,
            detected_level=detected_level.upper(),
            system_prompt_used=f"{detected_level.upper()} seviyesi için özelleştirilmiş prompt",
            clean_prompt=clean_user_prompt
        )
        
    except Exception as e:
        logger.error(f"[ENGLISH ERROR] İçerik üretme hatası: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"İçerik üretme hatası: {str(e)}")

@app.post("/english/stream")
async def stream_english_content(request: EnglishLearningRequest, api_key: str = Depends(validate_api_key)):
    """İngilizce öğrenme içeriği stream olarak üretir"""
    
    async def generate_stream():
        try:
            logger.info(f"[ENGLISH STREAM] Request başlatıldı: '{request.prompt[:50]}...'")

            # Input sanitization (SQL injection kontrolü olmadan - AI prompt için)
            sanitized_prompt = security_validator.sanitize_ai_prompt(request.prompt, "english_stream_prompt")

            # Seviyeyi algıla
            detected_level = detect_english_level(sanitized_prompt)
            
            # Prompt'u temizle
            clean_user_prompt = clean_prompt(sanitized_prompt)
            
            # Uygun system prompt'u seç
            system_prompt = LEVEL_SYSTEM_PROMPTS.get(detected_level, LEVEL_SYSTEM_PROMPTS["b1"])
            
            # Final prompt'u oluştur
            final_prompt = f"{system_prompt}\n\nÖğrenci sorusu: {clean_user_prompt}"
            
            logger.info(f"[ENGLISH STREAM] Seviye: {detected_level.upper()}")
            
            # Önce seviye bilgisini gönder
            yield f"data: {{'type': 'level', 'level': '{detected_level.upper()}', 'clean_prompt': '{clean_user_prompt}'}}\n\n"
            
            # FAL client ile stream (resilience ile)
            stream = resilient_client.stream_async_with_resilience(
                "workflows/halillllibrahim58/eng-teach",
                arguments={
                    "prompt": final_prompt,
                    "system_prompt": system_prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                },
                operation_type="english_learning"
            )
            
            # Stream eventlerini gönder
            async for event in stream:
                if hasattr(event, 'type') and event.type == 'text':
                    yield f"data: {{'type': 'text', 'content': '{event.content}'}}\n\n"
                else:
                    # Event'i string'e çevir
                    yield f"data: {str(event)}\n\n"
                    
        except Exception as e:
            logger.error(f"[ENGLISH STREAM ERROR] {str(e)}")
            traceback.print_exc()
            yield f"data: {{'type': 'error', 'message': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/english/test-levels")
async def test_level_detection():
    """Seviye algılama fonksiyonunu test eder"""
    test_prompts = [
        "ingilizce seviyesi: a1. Merhaba nasılsın?",
        "ingilizce seviyesi: b2. I want to improve my writing skills",
        "ingilizce seviyesi: c1. Could you explain complex grammatical structures?",
        "Hello, how are you today?",  # Seviye belirtilmemiş
    ]
    
    results = []
    for prompt in test_prompts:
        level = detect_english_level(prompt)
        cleaned = clean_prompt(prompt)
        results.append({
            "original_prompt": prompt,
            "detected_level": level.upper(),
            "cleaned_prompt": cleaned
        })
    
    return {
        "test_results": results,
        "available_levels": list(LEVEL_SYSTEM_PROMPTS.keys()),
        "default_level": "B1"
    }


# ========================
# GÖRSEL ÜRETİMİ API'Sİ
# ========================

@app.post("/generate/image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest, api_key: str = Depends(validate_api_key)):
    """Görsel üretimi endpoint'i - Fal AI workflow ile"""
    try:
        logger.info(f"[IMAGE GENERATION] Gelen request: '{request.prompt[:50]}...'")
        
        # Fal AI workflow ile görsel üret (resilience ile)
        fallback_response = await create_fallback_response("image_generation",
                                                          workflow_id=request.workflow_id,
                                                          prompt=request.prompt)

        result = await resilient_client.run_async_with_resilience(
            request.workflow_id,
            arguments={
                "prompt": request.prompt,
                "max_tokens": 10000,
                "temperature": 0.7
            },
            fallback_response=fallback_response,
            operation_type="image_generation"
        )
        
        # Sonuçtan görsel URL'lerini al
        image_urls = []
        if "images" in result and result["images"]:
            # Birden fazla görsel varsa hepsini al
            for img in result["images"]:
                if isinstance(img, dict) and "url" in img:
                    image_urls.append(img["url"])
                elif isinstance(img, str):
                    image_urls.append(img)
        elif "image_url" in result:
            image_urls.append(result["image_url"])
        elif "output" in result and isinstance(result["output"], str) and result["output"].startswith("http"):
            image_urls.append(result["output"])
        
        if not image_urls:
            logger.warning("[IMAGE GENERATION] Hiç görsel URL'si bulunamadı")
            return ImageGenerationResponse(
                success=False,
                image_url=None,
                workflow_id=request.workflow_id,
                prompt=request.prompt,
                error_message="Görsel URL'si alınamadı",
                generated_at=datetime.now().isoformat()
            )
        
        # İlk görseli ana URL olarak kullan, diğerlerini metadata'da sakla
        main_image_url = image_urls[0]
        logger.info(f"[IMAGE GENERATION] {len(image_urls)} görsel başarıyla üretildi")

        # Tüm görselleri logla
        logger.info(f"[IMAGE GENERATION] Ana görsel: {main_image_url}")
        if len(image_urls) > 1:
            logger.info(f"[IMAGE GENERATION] Diğer görseller:")
            for i, img_url in enumerate(image_urls[1:], 2):
                logger.info(f"[IMAGE GENERATION]   {i}. {img_url}")
        else:
            logger.info(f"[IMAGE GENERATION] Tek görsel üretildi: {main_image_url}")
        
        return ImageGenerationResponse(
            success=True,
            image_url=main_image_url,
            workflow_id=request.workflow_id,
            prompt=request.prompt,
            error_message=None,
            generated_at=datetime.now().isoformat(),
            all_images=image_urls,  # Tüm görselleri de döndür
            total_images=len(image_urls)
        )
        
    except Exception as e:
        logger.error(f"[IMAGE GENERATION ERROR] Görsel üretme hatası: {str(e)}")
        traceback.print_exc()
        return ImageGenerationResponse(
            success=False,
            image_url=None,
            workflow_id=request.workflow_id,
            prompt=request.prompt,
            error_message=str(e),
            generated_at=datetime.now().isoformat()
        )


@app.post("/generate/image/stream")
async def stream_image_generation(request: ImageGenerationRequest, api_key: str = Depends(validate_api_key)):
    """Görsel üretimi stream endpoint'i"""
    
    async def generate_stream():
        try:
            # Input sanitization (SQL injection kontrolü olmadan - AI prompt için)
            sanitized_prompt = security_validator.sanitize_ai_prompt(request.prompt, "image_stream_prompt")
            logger.info(f"[IMAGE STREAM] Request başlatıldı: '{sanitized_prompt[:50]}...'")

            # Önce başlangıç bilgisini gönder
            yield f"data: {{'type': 'start', 'workflow_id': '{request.workflow_id}', 'prompt': '{sanitized_prompt}'}}\n\n"
            
            # Fal AI workflow ile stream (resilience ile)
            stream = resilient_client.stream_async_with_resilience(
                request.workflow_id,
                arguments={
                    "prompt": sanitized_prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                },
                operation_type="image_generation"
            )
            
            # Stream eventlerini gönder
            async for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'text':
                        yield f"data: {{'type': 'text', 'content': '{event.content}'}}\n\n"
                    elif event.type == 'image':
                        yield f"data: {{'type': 'image', 'url': '{event.url}'}}\n\n"
                    else:
                        yield f"data: {{'type': 'event', 'data': '{str(event)}'}}\n\n"
                else:
                    # Event'i string'e çevir
                    yield f"data: {{'type': 'event', 'data': '{str(event)}'}}\n\n"
                    
        except Exception as e:
            logger.error(f"[IMAGE STREAM ERROR] {str(e)}")
            traceback.print_exc()
            yield f"data: {{'type': 'error', 'message': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/generate/image/info")
async def image_generation_info():
    """Görsel üretimi sistemi hakkında bilgi"""
    return {
        "supported_workflows": [
            "workflows/halillllibrahim58/teach-img-model"
        ],
        "default_workflow": "workflows/halillllibrahim58/teach-img-model",
        "max_prompt_length": 1000,
        "supported_formats": ["jpg", "png", "webp"],
        "streaming_support": True,
        "model": "Fal AI Workflow",
        "version": "1.0.0"
    }

# ========================
# RESILIENCE & STATİSTİKLER
# ========================

@app.get("/resilience/stats")
async def get_resilience_stats():
    """Hata yönetimi ve resilience istatistikleri"""
    try:
        stats = resilient_client.get_stats()
        return {
            "service": "TutorlyAI Resilience System",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "resilience_stats": stats,
            "health_status": "healthy" if stats["circuit_breaker"]["state"] == "closed" else "degraded",
            "features": {
                "circuit_breaker": True,
                "retry_mechanism": True,
                "rate_limiting": True,
                "timeout_handling": True,
                "fallback_responses": True
            }
        }
    except Exception as e:
        logger.error(f"[RESILIENCE STATS ERROR] {str(e)}")
        return {
            "service": "TutorlyAI Resilience System",
            "version": "1.0.0",
            "error": str(e),
            "health_status": "error"
        }

@app.post("/resilience/reset")
async def reset_circuit_breaker():
    """Circuit breaker'ı sıfırla (Acil durum için)"""
    try:
        resilient_client.circuit_breaker.state = CircuitState.CLOSED
        resilient_client.circuit_breaker.failure_count = 0
        resilient_client.circuit_breaker.success_count = 0
        resilient_client.circuit_breaker.last_failure_time = None

        logger.info("[RESILIENCE] Circuit breaker manuel olarak sıfırlandı")

        return {
            "message": "Circuit breaker başarıyla sıfırlandı",
            "new_state": "closed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[RESILIENCE RESET ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Circuit breaker sıfırlama hatası: {str(e)}")

@app.get("/resilience/config")
async def get_resilience_config():
    """Mevcut resilience konfigürasyonunu döndürür"""
    try:
        config = resilient_client.config

        return {
            "timeout_seconds": config.timeout_seconds,
            "retry_config": {
                "max_attempts": config.retry_config.max_attempts,
                "base_delay": config.retry_config.base_delay,
                "max_delay": config.retry_config.max_delay,
                "exponential_base": config.retry_config.exponential_base,
                "jitter": config.retry_config.jitter
            },
            "circuit_breaker_config": {
                "failure_threshold": config.circuit_breaker_config.failure_threshold,
                "success_threshold": config.circuit_breaker_config.success_threshold,
                "timeout_seconds": config.circuit_breaker_config.timeout_seconds,
                "half_open_max_calls": config.circuit_breaker_config.half_open_max_calls
            },
            "rate_limit_config": {
                "requests_per_minute": config.rate_limit_config.requests_per_minute,
                "requests_per_hour": config.rate_limit_config.requests_per_hour
            },
            "enable_fallback": config.enable_fallback
        }
    except Exception as e:
        logger.error(f"[RESILIENCE CONFIG ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Konfigürasyon alma hatası: {str(e)}")

# ========================
# GÜVENLİK YÖNETİMİ API'Sİ
# ========================

@app.get("/security/violations")
async def get_security_violations(hours: int = 24, api_key: str = Depends(validate_api_key)):
    """Güvenlik ihlallerini listeler"""
    try:
        violations = security_validator.get_security_violations(hours)

        return {
            "total_violations": len(violations),
            "time_range_hours": hours,
            "violations": [
                {
                    "type": v.violation_type,
                    "details": v.details,
                    "client_ip": v.client_ip,
                    "timestamp": v.timestamp.isoformat(),
                    "request_path": v.request_path,
                    "severity": v.severity
                }
                for v in violations[-50:]  # Son 50 ihlalin detayını göster
            ],
            "violation_summary": {
                violation_type: len([v for v in violations if v.violation_type == violation_type])
                for violation_type in set(v.violation_type for v in violations)
            }
        }
    except Exception as e:
        logger.error(f"[SECURITY] Violation listesi alma hatası: {str(e)}")
        raise HTTPException(status_code=500, detail="Güvenlik verilerini alırken hata oluştu")

@app.post("/security/api-keys/generate")
async def generate_new_api_key(user_id: str = "new_user", admin_key: str = Depends(validate_api_key)):
    """Yeni API key oluşturur (admin gerekli)"""
    try:
        # Admin kontrolü - sabit key ile
        if admin_key != VALIDATE_API_KEY:
            raise HTTPException(status_code=403, detail="Admin privileges required")

        new_key = api_key_manager.generate_api_key(user_id)

        logger.info(f"[SECURITY] Yeni API key oluşturuldu: {user_id}")

        return {
            "message": "API key başarıyla oluşturuldu",
            "api_key": new_key,
            "user_id": user_id,
            "expires_in_hours": security_validator.config.api_key_rotation_hours,
            "created_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SECURITY] API key oluşturma hatası: {str(e)}")
        raise HTTPException(status_code=500, detail="API key oluşturulamadı")

@app.post("/security/api-keys/rotate")
async def rotate_api_key(old_key: str, admin_key: str = Depends(validate_api_key)):
    """API key'i rotate eder"""
    try:
        # Admin kontrolü
        if admin_key != VALIDATE_API_KEY:
            raise HTTPException(status_code=403, detail="Admin privileges required")

        # Key'in var olup olmadığını kontrol et
        if old_key not in api_key_manager.valid_keys:
            raise HTTPException(status_code=404, detail="API key not found")

        user_id = api_key_manager.valid_keys[old_key]["user_id"]
        new_key = api_key_manager.rotate_api_key(old_key, user_id)

        logger.info(f"[SECURITY] API key rotate edildi: {user_id}")

        return {
            "message": "API key başarıyla rotate edildi",
            "new_api_key": new_key,
            "old_key_revoked": True,
            "user_id": user_id,
            "rotated_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SECURITY] API key rotation hatası: {str(e)}")
        raise HTTPException(status_code=500, detail="API key rotate edilemedi")

@app.delete("/security/api-keys/{api_key}")
async def revoke_api_key(api_key: str, admin_key: str = Depends(validate_api_key)):
    """API key'i iptal eder"""
    try:
        # Admin kontrolü
        if admin_key != VALIDATE_API_KEY:
            raise HTTPException(status_code=403, detail="Admin privileges required")

        if api_key not in api_key_manager.valid_keys:
            raise HTTPException(status_code=404, detail="API key not found")

        api_key_manager.revoke_api_key(api_key)

        logger.info(f"[SECURITY] API key iptal edildi: {api_key[:8]}...")

        return {
            "message": "API key başarıyla iptal edildi",
            "revoked_key": f"{api_key[:8]}...",
            "revoked_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SECURITY] API key iptal hatası: {str(e)}")
        raise HTTPException(status_code=500, detail="API key iptal edilemedi")

@app.get("/security/api-keys/stats")
async def get_api_key_stats(admin_key: str = Depends(validate_api_key)):
    """API key istatistiklerini döndürür"""
    try:
        # Admin kontrolü
        if admin_key != VALIDATE_API_KEY:
            raise HTTPException(status_code=403, detail="Admin privileges required")

        stats = api_key_manager.get_key_stats()

        return {
            "service": "TutorlyAI API Key Management",
            "timestamp": datetime.now().isoformat(),
            "api_key_stats": stats,
            "static_key_info": {
                "key_preview": f"{VALIDATE_API_KEY[:8]}...",
                "note": "Static key from .env file for testing teams"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SECURITY] API key stats hatası: {str(e)}")
        raise HTTPException(status_code=500, detail="İstatistikler alınamadı")

@app.get("/security/config")
async def get_security_config(admin_key: str = Depends(validate_api_key)):
    """Güvenlik konfigürasyonunu döndürür"""
    try:
        # Admin kontrolü
        if admin_key != VALIDATE_API_KEY:
            raise HTTPException(status_code=403, detail="Admin privileges required")

        config = security_validator.config

        return {
            "security_config": {
                "max_request_size_mb": config.max_request_size / (1024 * 1024),
                "max_prompt_length": config.max_prompt_length,
                "max_filename_length": config.max_filename_length,
                "rate_limit_requests_per_minute": config.rate_limit_requests_per_minute,
                "api_key_rotation_hours": config.api_key_rotation_hours,
                "protections_enabled": {
                    "sql_injection": config.enable_sql_injection_protection,
                    "xss": config.enable_xss_protection,
                    "command_injection": config.enable_command_injection_protection,
                    "path_traversal": config.enable_path_traversal_protection,
                    "api_key_validation": config.enable_api_key_validation
                }
            },
            "allowed_file_extensions": list(security_validator.config.allowed_origins),
            "security_headers_enabled": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SECURITY] Config alma hatası: {str(e)}")
        raise HTTPException(status_code=500, detail="Güvenlik config alınamadı")

# ========================
# PERFORMANCE YÖNETİMİ API'Sİ
# ========================

@app.get("/performance/stats")
async def get_performance_stats_endpoint():
    """Performance istatistiklerini döndürür"""
    try:
        performance_stats = get_performance_stats()
        database_stats = get_database_stats()

        return {
            "service": "TutorlyAI Performance Monitoring",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "performance": performance_stats,
            "database": database_stats,
            "health_status": "healthy"
        }
    except Exception as e:
        logger.error(f"[PERFORMANCE] Stats alma hatası: {str(e)}")
        return {
            "service": "TutorlyAI Performance Monitoring",
            "version": "1.0.0",
            "error": str(e),
            "health_status": "error"
        }

@app.get("/performance/memory")
async def get_memory_stats():
    """Memory kullanım istatistikleri"""
    try:
        memory_stats = memory_monitor.get_stats()

        # Memory warning kontrolü
        status = "healthy"
        if memory_stats["current_memory_percent"] > 80:
            status = "warning"
        elif memory_stats["current_memory_percent"] > 90:
            status = "critical"

        return {
            "memory_stats": memory_stats,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[PERFORMANCE] Memory stats hatası: {str(e)}")
        raise HTTPException(status_code=500, detail="Memory stats alınamadı")
        
@app.get("/performance/concurrency")
async def get_concurrency_stats():
    """Concurrent request istatistikleri"""
    try:
        concurrency_stats = request_limiter.get_stats()

        return {
            "concurrency_stats": concurrency_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[PERFORMANCE] Concurrency stats hatası: {str(e)}")
        raise HTTPException(status_code=500, detail="Concurrency stats alınamadı")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TutorlyAI - RAG, Quiz & English Learning API Başlatılıyor")
    logger.info("=" * 60)

    # Güvenlik bilgileri
    logger.info(f"[SECURITY] Static API Key: {VALIDATE_API_KEY}")
    logger.info("[SECURITY] Using static key from .env file for testing teams")
    logger.info("[SECURITY] API Key validation enabled")
    logger.info("[SECURITY] All security protections active")
    logger.info("=" * 60)
    
    # Performance monitoring başlat
    memory_monitor.start_monitoring()
    logger.info("[✓] Memory monitoring başlatıldı")

    # Redis cache sistemini başlat
    try:
        from tools.redis_cache_adapters import initialize_redis_caches, redis_response_cache, redis_query_cache
        from tools.performance_utils import set_redis_cache
        from tools.database_pool import set_redis_query_cache

        import asyncio
        redis_success = asyncio.run(initialize_redis_caches())

        if redis_success:
            # Cache instance'larını güncelle
            set_redis_cache(redis_response_cache)
            set_redis_query_cache(redis_query_cache)
            logger.info("[✓] Redis cache sistemi başlatıldı")
        else:
            logger.warning("[!] Redis cache başlatılamadı, in-memory cache kullanılacak")

    except Exception as e:
        logger.warning(f"[!] Redis cache başlatma hatası: {str(e)}")
        logger.warning("[!] In-memory cache kullanılacak")

    # RAG sistemini başlat
    if initialize_rag_system():
        logger.info("[✓] RAG sistemi başlatıldı")

        # Kitapları yükle
        if should_load_books():
            logger.info("[INFO] Kitaplar yükleniyor...")
            import asyncio
            success = asyncio.run(load_books_async())

            if success:
                logger.info("[✓] Kitaplar başarıyla yüklendi")
            else:
                logger.warning("[!] Kitap yükleme başarısız")
        else:
            logger.info("[INFO] Kitaplar zaten yüklü")
    else:
        logger.warning("[✗] RAG sistemi başlatılamadı")
    
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info("API başlatılıyor...")
    logger.info("=" * 60)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)