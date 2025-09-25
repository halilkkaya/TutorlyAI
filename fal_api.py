from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
import os
import asyncio
from dotenv import load_dotenv
import traceback
import fal_client
import glob

from tools.get_search_plan import get_search_plan
from tools.generate_stream import generate_stream
from tools.generate_quiz import generate_quiz
from tools.classes import TextGenerationRequest, QuizRequest
from tools.initalize_rag_system import initialize_rag_system, should_load_books, load_books_async, search_books_enhanced

# Ortam değişkenlerini yükle
load_dotenv()

# FastAPI uygulaması oluştur
app = FastAPI(
    title="TutorlyAI - Enhanced RAG & Quiz API",
    description="Gelişmiş RAG sistemi ve Quiz oluşturma API'si - fal.ai Gemini 2.5 Flash ile güçlendirilmiş",
    version="2.1.0"
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

# RAG sistemi için global değişkenler
vectorstore = None
embedding_model = None
text_splitter = None

@app.get("/health")
async def health_check():
    """Sağlık kontrolü"""
    return {"status": "healthy", "rag_active": vectorstore is not None}

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
    
    results = search_books_enhanced(
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
async def stream_text(request: TextGenerationRequest):
    """Streaming metin üretme"""
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream"
    )
@app.post("/generate")
async def generate_rag_answer(request: TextGenerationRequest):
    """RAG ile cevap üretir - Score threshold ile"""
    try:
        print(f"[GENERATE] Gelen sorgu: '{request.prompt}'")
        
        # 1. Arama planı oluştur
        search_plan = await get_search_plan(request.prompt)
        query = search_plan.get("query", request.prompt)
        filters = search_plan.get("filters", {})
        
        print(f"[GENERATE] Arama planı - Query: '{query}', Filters: {filters}")
        
        # 2. Hibrit arama stratejisi ile doküman bulma (request parametrelerini kullan)
        relevant_docs = search_books_enhanced(
            query=query, 
            filters=filters, 
            k=request.search_k, 
            score_threshold=request.score_threshold,
            use_hybrid=request.use_hybrid,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight
        )
        
        
        if not relevant_docs:
            print("[GENERATE] Threshold'u geçen doküman bulunamadı")
            
        
        # Sonuç olsun ya da olmasın, modele gönder
        
        # 3. Context oluştur
        if relevant_docs:
            context_text = "\n\n---\n\n".join([
                f"[{doc.metadata.get('source', 'Bilinmeyen')}]\n{doc.page_content}"
                for doc in relevant_docs
            ])
            print(f"[GENERATE] {len(relevant_docs)} kaliteli doküman bulundu")
            
            # 4A. Dökümanlar bulunduğunda
            synthesis_prompt = f"""Sen bir ders kitabı uzmanısın. Aşağıdaki soruyu, verilen ders kitabı metinlerini kullanarak cevapla.

            SORU: {request.prompt}

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
            print("[GENERATE] Hiç kaliteli doküman bulunamadı, model kendi bilgisiyle cevap verecek")
            
            # 4B. Hiç doküman bulunamadığında
            synthesis_prompt = f"""Sen bir ders kitabı uzmanısın. Kullanıcı soru sordu ancak ders kitaplarında bu konuyla yeterince benzer içerik bulunamadı.

            SORU: {request.prompt}

            DURUM: Ders kitaplarında bu soruyla yeterince benzer içerik bulunamadı.

            KURALLAR:
            1. Eğer bu bir selamlaşma ise normal karşılık ver
            2. Eğer genel bir soru ise kendi bilginle yardım etmeye çalış  
            3. Eğer çok spesifik bir ders konusu ise, daha spesifik terimlerle tekrar sormalarını öner
            4. Kibarca ve yardımsever ol

            CEVAP:"""

        # 5. Modeli çağır ve cevap al
        result = await fal_client.run_async(
            FAL_MODEL_GATEWAY,
            arguments={
                "model": MODEL_NAME,
                "prompt": synthesis_prompt,
                "max_tokens": 1500,
                "temperature": 0.3,
            },
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
        print(f"[GENERATE ERROR] {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quiz/generate")
async def generate_quiz_endpoint(request: QuizRequest):
    """Quiz soruları oluştur"""
    try:
        print(f"[QUIZ API] Quiz generation request: {request.soru_sayisi} {request.soru_tipi} soru")
        
        quiz_response = await generate_quiz(request)
        
        return {
            "success": True,
            "message": "Quiz başarıyla oluşturuldu",
            "data": quiz_response.model_dump()
        }
        
    except ValidationError as e:
        print(f"[QUIZ API] Pydantic validation error: {str(e)}")
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
        print(f"[QUIZ API] Generation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[QUIZ API ERROR] {str(e)}")
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


if __name__ == "__main__":
    print("=" * 60)
    print("Gelişmiş RAG Sistemi Başlatılıyor")
    print("=" * 60)
    
    # RAG sistemini başlat
    if initialize_rag_system():
        print("[✓] RAG sistemi başlatıldı")
        
        # Kitapları yükle
        if should_load_books():
            print("[INFO] Kitaplar yükleniyor...")
            import asyncio
            success = asyncio.run(load_books_async())
            
            if success:
                print("[✓] Kitaplar başarıyla yüklendi")
            else:
                print("[!] Kitap yükleme başarısız")
        else:
            print("[INFO] Kitaplar zaten yüklü")
    else:
        print("[✗] RAG sistemi başlatılamadı")
    
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print("API başlatılıyor...")
    print("=" * 60)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)