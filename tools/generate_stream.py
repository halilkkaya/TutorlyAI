from tools.get_search_plan import get_search_plan
from tools.classes import TextGenerationRequest
import fal_client
import json
import traceback
from typing import AsyncGenerator
from tools.initalize_rag_system import search_books_enhanced
from tools.resilience_utils import resilient_client
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "google/gemini-2.5-flash"
FAL_MODEL_GATEWAY = "fal-ai/any-llm"


async def generate_stream(request: TextGenerationRequest) -> AsyncGenerator[str, None]:
    """RAG entegreli streaming metin üretimi"""
    try:
        # 🔍 1. ADIM: RAG Araması (hızlı)
        logger.info(f"[STREAM] RAG araması başlatılıyor: '{request.prompt}'")
        
        # Arama durumunu kullanıcıya bildir
        status_data = {
            "status": "searching",
            "message": "Ders kitaplarında arama yapılıyor...",
            "done": False
        }
        yield f"data: {json.dumps(status_data)}\n\n"
        
        # Arama planı oluştur
        search_plan = await get_search_plan(request.prompt)
        query = search_plan.get("query", request.prompt)
        filters = search_plan.get("filters", {})
        
        # Hibrit arama ile kitapları ara
        relevant_docs = search_books_enhanced(
            query=query, 
            filters=filters, 
            k=4,
            score_threshold=0.25,  # Streaming için daha düşük threshold
            use_hybrid=True,       # Hibrit aramayı etkinleştir
            semantic_weight=0.7,   # Semantic ağırlığı
            keyword_weight=0.3     # BM25 keyword ağırlığı
        )
        
        # Arama sonucunu bildir
        search_result_data = {
            "status": "search_complete",
            "message": f"{len(relevant_docs)} kaynak bulundu, cevap oluşturuluyor...",
            "found_sources": len(relevant_docs),
            "search_plan": search_plan,
            "done": False
        }
        yield f"data: {json.dumps(search_result_data)}\n\n"
        
        # 📚 2. ADIM: Context Hazırlama
        if relevant_docs:
            context_text = "\n\n---\n\n".join([
                f"[{doc.metadata.get('source', 'Kaynak')}]\n{doc.page_content}"
                for doc in relevant_docs
            ])
            
            # Enhanced prompt with context
            enhanced_prompt = f"""Sen bir ders kitabı uzmanısın. Aşağıdaki soruyu, verilen ders kitabı metinlerini kullanarak cevapla.

SORU: {request.prompt}

DERS KİTABI İÇERİĞİ:
{context_text}

KURALLAR:
1. Sadece verilen kaynaklardaki bilgileri kullan
2. Kapsamlı ve anlaşılır bir açıklama yap  
3. Hangi kaynaktan bilgi aldığını belirt
4. Cevabı direkt ver, giriş yapma

CEVAP:"""

            sources = [doc.metadata.get('source', 'Bilinmeyen') for doc in relevant_docs]
            
        else:
            # Eğer kaynak bulunamazsa genel bilgi ile devam et
            enhanced_prompt = f"""Soru: {request.prompt}

Bu soruyla ilgili ders kitaplarında spesifik bilgi bulamadım, ama genel bilgilerimi kullanarak yardımcı olmaya çalışacağım:"""
            sources = []
        
        # 🚀 3. ADIM: Streaming başlat
        logger.info(f"[STREAM] LLM streaming başlatılıyor...")
        
        # Fal.ai stream_async kullan (resilience ile)
        stream = resilient_client.stream_async_with_resilience(
            FAL_MODEL_GATEWAY,
            arguments={
                "model": MODEL_NAME,
                "prompt": enhanced_prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
            operation_type="stream"
        )

        # İlk token geldiğinde generation başladığını bildir
        first_token = True
        
        # Stream'i dinle
        async for event in stream:
            # Event yapısını logla (debug için)
            logger.info(f"[STREAM DEBUG] Event: {event}")
            
            # Farklı event tiplerini kontrol et
            if isinstance(event, dict):
                # Text chunk event'i
                if "chunk" in event and event["chunk"]:
                    chunk_text = event["chunk"]
                    
                    if first_token:
                        # İlk token geldiğinde generation_started event'i gönder
                        generation_start_data = {
                            "status": "generation_started",
                            "message": "Cevap oluşturuluyor...",
                            "sources": sources,
                            "done": False
                        }
                        yield f"data: {json.dumps(generation_start_data)}\n\n"
                        first_token = False
                    
                    # Normal text chunk
                    chunk_data = {
                        "text": chunk_text,
                        "type": "content",
                        "done": False
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Alternative formats
                elif "output" in event and event["output"]:
                    chunk_text = event["output"]
                    
                    if first_token:
                        generation_start_data = {
                            "status": "generation_started", 
                            "message": "Cevap oluşturuluyor...",
                            "sources": sources,
                            "done": False
                        }
                        yield f"data: {json.dumps(generation_start_data)}\n\n"
                        first_token = False
                    
                    chunk_data = {
                        "text": chunk_text,
                        "type": "content",
                        "done": False
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Stream completed
                elif "done" in event or "response" in event:
                    final_response = event.get("response", "")
                    
                    # Hibrit arama detaylarını hazırla
                    search_details = {}
                    if relevant_docs:
                        search_details = {
                            "average_hybrid_score": sum(doc.metadata.get('hybrid_score', 0) for doc in relevant_docs) / len(relevant_docs),
                            "average_semantic_score": sum(doc.metadata.get('semantic_score', 0) for doc in relevant_docs) / len(relevant_docs),
                            "average_bm25_score": sum(doc.metadata.get('bm25_score', 0) for doc in relevant_docs) / len(relevant_docs)
                        }
                    
                    # Generation tamamlandı
                    final_data = {
                        "status": "completed",
                        "message": "Cevap tamamlandı",
                        "full_response": final_response,
                        "sources": sources,
                        "search_plan": search_plan,
                        "found_documents": len(relevant_docs),
                        "search_method": "hybrid_search",
                        "search_details": search_details,
                        "done": True
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"
                    break
                
                # Error handling
                elif "error" in event:
                    error_data = {
                        "status": "error",
                        "error": str(event["error"]),
                        "message": "Stream sırasında hata oluştu",
                        "done": True
                    }
                    yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                    break
            
            # String event (simple text)
            elif isinstance(event, str) and event.strip():
                if first_token:
                    generation_start_data = {
                        "status": "generation_started",
                        "message": "Cevap oluşturuluyor...",
                        "sources": sources,
                        "done": False
                    }
                    yield f"data: {json.dumps(generation_start_data)}\n\n"
                    first_token = False
                
                chunk_data = {
                    "text": event,
                    "type": "content", 
                    "done": False
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

    except Exception as e:
        logger.error(f"[STREAM ERROR] {str(e)}")
        traceback.print_exc()
        error_data = {
            "status": "error",
            "error": str(e),
            "message": "Bir hata oluştu",
            "done": True
        }
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
