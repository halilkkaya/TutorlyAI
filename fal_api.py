from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from typing import Optional, AsyncGenerator, List, Dict, Any
import json
import asyncio
from dotenv import load_dotenv
import traceback
import re
# fal-client kütüphanesini içe aktar
import fal_client

# RAG sistemi için import'lar
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import chromadb
from PyPDF2 import PdfReader
import glob
from pathlib import Path

# Ortam değişkenlerini yükle
load_dotenv()

# FastAPI uygulaması oluştur
app = FastAPI(
    title="Fal.ai Any-LLM (Gemini 2.5 Flash) API with Enhanced RAG",
    description="Gelişmiş RAG sistemi ile fal.ai 'any-llm' modeli kullanarak streaming metin üretme API'si",
    version="2.0.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model - istek yapısı
class TextGenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    tools: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None

# RAG sistemi için yanıt modeli
class ToolResponse(BaseModel):
    tool_call_id: str
    result: str

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

# Gelişmiş metadata ayrıştırıcı
def parse_filename_for_metadata(filename: str) -> Optional[Dict[str, Any]]:
    """
    Dosya adından metadata çıkarır. Desteklenen formatlar:
    - 9_sinif_biyoloji.pdf
    - 10_sinif_fizik.pdf 
    - 11_sinif_matematik.pdf
    """
    print(f"[METADATA] Dosya adı ayrıştırılıyor: {filename}")
    
    # PDF uzantısını kaldır
    name_without_ext = filename.replace('.pdf', '').replace('.PDF', '')
    
    # Regex pattern: (sayı)_sinif_(ders_adı)
    patterns = [
        r"(\d+)_sinif_(.+)",           # 9_sinif_biyoloji
        r"(\d+)\.sinif\.(.+)",         # 9.sinif.biyoloji
        r"(\d+)_class_(.+)",           # 9_class_biology
        r"sinif_(\d+)_(.+)",           # sinif_9_biyoloji
    ]
    
    for pattern in patterns:
        match = re.match(pattern, name_without_ext, re.IGNORECASE)
        if match:
            try:
                class_num = int(match.group(1))
                subject = match.group(2).lower().strip()
                
                # Konu adını temizle
                subject = re.sub(r'[_\-\.\s]+', '_', subject).strip('_')
                
                metadata = {
                    "sinif": class_num,
                    "ders": subject
                }
                print(f"[METADATA] Başarıyla ayrıştırıldı: {metadata}")
                return metadata
            except (ValueError, IndexError) as e:
                print(f"[METADATA] Ayrıştırma hatası: {e}")
                continue
    
    print(f"[METADATA] Dosya adı format uyumsuz: {filename}")
    return None

# Arama planlayıcısı için gelişmiş system prompt
QUERY_PLANNER_SYSTEM_PROMPT = """Sen bir akıllı arama asistanısın. Kullanıcının sorusunu analiz ederek vektör arama için optimal parametreleri oluşturacaksın.

Çıktın SADECE aşağıdaki JSON formatında olmalıdır:
{
  "query": "anahtar kelimeler ve kavramlar",
  "filters": {
    "sinif": 9,
    "ders": "biyoloji"
  }
}

KURALLAR:
1. "query" kısmında önemli kavramları ve anahtar kelimeleri ayıkla
2. Sınıf ve ders bilgisi varsa "filters"e ekle
3. Ders adlarını küçük harfle yaz: "biyoloji", "fizik", "matematik", "kimya", "tarih", "coğrafya"
4. Sınıf numarası 9, 10, 11 veya 12 olmalıdır
5. Eğer filtre bilgisi yoksa boş bırak

ÖRNEKLER:

"10. sınıf biyoloji hücre bölünmesi nedir?" → 
{
  "query": "hücre bölünmesi mitoz mayoz",
  "filters": {"sinif": 10, "ders": "biyoloji"}
}

"Fizik dersi ivme konusu" → 
{
  "query": "ivme hareket hız",
  "filters": {"ders": "fizik"}
}

"Fotosentez nasıl olur?" → 
{
  "query": "fotosentez klorofil güneş ışığı",
  "filters": {}
}
"""

async def get_search_plan(user_prompt: str) -> Dict[str, Any]:
    """Kullanıcı sorgusundan arama planı oluşturur"""
    print(f"[PLANNER] Sorgu analizi: '{user_prompt}'")
    
    try:
        result = await fal_client.run_async(
            FAL_MODEL_GATEWAY,
            arguments={
                "model": MODEL_NAME,
                "prompt": user_prompt,
                "system_prompt": QUERY_PLANNER_SYSTEM_PROMPT,
                "max_tokens": 150,
                "temperature": 0.1,
            },
        )
        
        response_text = result.get("output", "{}").strip()
        print(f"[PLANNER] Model yanıtı: {response_text}")
        
        # JSON'u çıkar
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        
        if not json_match:
            print("[PLANNER] JSON bulunamadı, varsayılan plan kullanılıyor")
            return {"query": user_prompt, "filters": {}}
        
        clean_json = json_match.group(0)
        plan = json.loads(clean_json)
        
        # Plan doğrulama
        if "query" not in plan:
            plan["query"] = user_prompt
        if "filters" not in plan:
            plan["filters"] = {}
        
        # Boş filtreleri temizle
        plan["filters"] = {k: v for k, v in plan["filters"].items() 
                          if v is not None and str(v).strip() != ""}
        
        print(f"[PLANNER] Final plan: {plan}")
        return plan
        
    except Exception as e:
        print(f"[PLANNER] Hata: {e}")
        return {"query": user_prompt, "filters": {}}

@app.get("/")
async def root():
    """Ana endpoint"""
    global vectorstore
    
    # Veri sayısını kontrol et
    doc_count = 0
    if vectorstore:
        try:
            doc_count = vectorstore._collection.count()
        except:
            doc_count = -1
    
    return {
        "message": "Gelişmiş RAG Sistemi Aktif!",
        "status": "active",
        "version": "2.0.0",
        "documents_loaded": doc_count,
        "features": [
            "Akıllı arama planlayıcısı",
            "Metadata tabanlı filtreleme",
            "9-12. sınıf ders kitapları",
            "Gelişmiş PDF okuma",
            "Debug endpoint'leri"
        ],
        "endpoints": {
            "generate": "/generate",
            "generate_stream": "/generate/stream", 
            "load_books": "/load-books",
            "search": "/search",
            "debug": "/debug",
            "rag_status": "/rag-status"
        }
    }

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
    """Manuel arama endpoint'i"""
    query = request.get("query", "")
    filters = request.get("filters", {})
    k = request.get("k", 5)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query gerekli")
    
    results = search_books_enhanced(query, filters, k)
    
    return {
        "query": query,
        "filters": filters,
        "results_count": len(results),
        "results": [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
                "relevance_score": getattr(doc, 'relevance_score', None)
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

# Streaming için asenkron jeneratör fonksiyonu
async def generate_stream(request: TextGenerationRequest) -> AsyncGenerator[str, None]:
    """Streaming metin üretimi"""
    try:
        handler = await fal_client.asubscribe(
            FAL_MODEL_GATEWAY,
            arguments={
                "model": MODEL_NAME,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
        )

        async for event in handler:
            if "chunk" in event and event["chunk"]:
                chunk_data = {
                    "text": event["chunk"],
                    "done": False
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            elif "response" in event:
                final_data = {
                    "text": "",
                    "done": True,
                    "full_response": event["response"]
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                break

    except Exception as e:
        error_data = {"error": str(e)}
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

@app.post("/generate/stream")
async def stream_text(request: TextGenerationRequest):
    """Streaming metin üretme"""
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream"
    )

@app.post("/generate")
async def generate_rag_answer(request: TextGenerationRequest):
    """RAG ile cevap üretir"""
    try:
        print(f"[GENERATE] Gelen sorgu: '{request.prompt}'")
        
        # 1. Arama planı oluştur
        search_plan = await get_search_plan(request.prompt)
        query = search_plan.get("query", request.prompt)
        filters = search_plan.get("filters", {})
        
        print(f"[GENERATE] Arama planı - Query: '{query}', Filters: {filters}")
        
        # 2. Kitapları ara
        relevant_docs = search_books_enhanced(query, filters, k=5)
        
        if not relevant_docs:
            print("[GENERATE] Hiç doküman bulunamadı")
            return {
                "generated_text": "Üzgünüm, sorunuzla ilgili ders kitaplarında bilgi bulamadım. Sorunuzu farklı kelimelerle tekrar sorabilir misiniz?",
                "search_plan": search_plan,
                "found_documents": 0
            }
        
        # 3. Bulunan dökümanları birleştir
        context_text = "\n\n---\n\n".join([
            f"[{doc.metadata.get('source', 'Bilinmeyen')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])
        
        print(f"[GENERATE] {len(relevant_docs)} doküman bulundu, context oluşturuluyor...")
        
        # 4. Nihai cevap oluştur
        synthesis_prompt = f"""Sen bir ders kitabı uzmanısın. Aşağıdaki soruyu, verilen ders kitabı metinlerini kullanarak cevapla.

SORU: {request.prompt}

DERS KİTABI İÇERİĞİ:
{context_text}

KURALLAR:
1. Sadece verilen kaynaklardaki bilgileri kullan
2. Kapsamlı ve anlaşılır bir açıklama yap  
3. Hangi kaynaktan bilgi aldığını belirt
4. Eğer cevap kaynaklarda yoksa, bunu söyle

CEVAP:"""

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
        
        return {
            "generated_text": final_answer,
            "search_plan": search_plan,
            "found_documents": len(relevant_docs),
            "sources": [doc.metadata.get('source', 'Bilinmeyen') for doc in relevant_docs]
        }
        
    except Exception as e:
        print(f"[GENERATE ERROR] {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_model_info():
    """Model bilgileri"""
    return {
        "service": "fal.ai",
        "gateway_model": FAL_MODEL_GATEWAY,
        "configured_llm": MODEL_NAME
    }

# RAG sistem fonksiyonları
def initialize_rag_system():
    """RAG sistemini başlatır"""
    global vectorstore, embedding_model, text_splitter
    
    try:
        print("[RAG] Sistem başlatılıyor...")
        
        # Embedding modeli
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("[RAG] Embedding model yüklendi")
        
        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,      # Daha küçük chunk'lar
            chunk_overlap=100,   # Daha az overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        print("[RAG] Text splitter hazırlandı")
        
        # ChromaDB
        vectorstore = Chroma(
            collection_name="enhanced_books_collection",
            embedding_function=embedding_model,
            persist_directory="./chroma_db_v2"  # Yeni versiyon için farklı klasör
        )
        print("[RAG] ChromaDB bağlandı")
        
        return True
        
    except Exception as e:
        print(f"[RAG ERROR] Başlatma hatası: {str(e)}")
        return False

async def load_books_async():
    """Kitapları asenkron olarak yükler"""
    global vectorstore, text_splitter
    
    if not vectorstore or not text_splitter:
        print("[RAG] Sistem başlatılmamış")
        return False
    
    try:
        print("[RAG] Kitaplar yükleniyor...")
        
        # PDF dosyalarını bul
        pdf_files = glob.glob("books/*.pdf")
        
        if not pdf_files:
            print("[RAG] books/ klasöründe PDF bulunamadı")
            return False
        
        print(f"[RAG] {len(pdf_files)} PDF dosyası bulundu")
        
        all_documents = []
        successful_files = 0
        
        for pdf_path in pdf_files:
            try:
                filename = Path(pdf_path).name
                print(f"[RAG] İşleniyor: {filename}")
                
                # Metadata çıkar
                metadata = parse_filename_for_metadata(filename)
                if not metadata:
                    print(f"[RAG] UYARI: {filename} format uyumsuz, atlanıyor")
                    continue
                
                # PDF oku
                reader = PdfReader(pdf_path)
                full_text = ""
                
                print(f"[RAG] {filename}: {len(reader.pages)} sayfa okunuyor...")
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            # Metni temizle
                            text = re.sub(r'\s+', ' ', text).strip()
                            full_text += text + "\n"
                    except Exception as e:
                        print(f"[RAG] Sayfa {page_num} okuma hatası: {e}")
                        continue
                
                if not full_text.strip():
                    print(f"[RAG] UYARI: {filename} metin çıkarılamadı")
                    continue
                
                # Chunk'lara ayır
                chunks = text_splitter.split_text(full_text)
                print(f"[RAG] {filename}: {len(chunks)} chunk oluşturuldu")
                
                # Döküman objelerini oluştur
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 50:  # Çok kısa chunk'ları filtrele
                        doc_metadata = {
                            "source": Path(pdf_path).stem,
                            "filename": filename,
                            "chunk_id": i,
                            "chunk_length": len(chunk),
                            **metadata  # sinif ve ders bilgileri
                        }
                        
                        all_documents.append(Document(
                            page_content=chunk.strip(),
                            metadata=doc_metadata
                        ))
                
                successful_files += 1
                print(f"[RAG] {filename} başarıyla işlendi")
                
            except Exception as e:
                print(f"[RAG] {pdf_path} işlenirken hata: {str(e)}")
                continue
        
        if all_documents:
            print(f"[RAG] {len(all_documents)} doküman ChromaDB'ye ekleniyor...")
            
            # Batch olarak ekle (büyük dosyalar için)
            batch_size = 100
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                vectorstore.add_documents(batch)
                print(f"[RAG] Batch {i//batch_size + 1}/{(len(all_documents)-1)//batch_size + 1} eklendi")
            
            print(f"[RAG] BAŞARILI: {successful_files} dosya, {len(all_documents)} chunk yüklendi")
            return True
        else:
            print("[RAG] Hiç doküman oluşturulamadı")
            return False
            
    except Exception as e:
        print(f"[RAG] Kitap yükleme genel hatası: {str(e)}")
        traceback.print_exc()
        return False

def search_books_enhanced(query: str, filters: Optional[Dict[str, Any]] = None, k: int = 5) -> List[Document]:
    """Gelişmiş kitap arama fonksiyonu"""
    global vectorstore
    
    if not vectorstore:
        print("[SEARCH] RAG sistemi aktif değil")
        return []
    
    try:
        print(f"[SEARCH] Arama başlatılıyor: '{query}'")
        print(f"[SEARCH] Filtreler: {filters}")
        
        # Arama parametreleri
        search_kwargs = {"k": k}
        
        # ChromaDB filtreleme formatı
        if filters:
            # Birden fazla filtre için $and operatörü
            if len(filters) > 1:
                where_clause = {
                    "$and": [
                        {key: {"$eq": value}} for key, value in filters.items()
                    ]
                }
            else:
                # Tek filtre
                key, value = list(filters.items())[0]
                where_clause = {key: {"$eq": value}}
            
            search_kwargs["filter"] = where_clause
            print(f"[SEARCH] ChromaDB filtre: {where_clause}")
        
        # Arama yap
        docs = vectorstore.similarity_search(query, **search_kwargs)
        
        print(f"[SEARCH] {len(docs)} sonuç bulundu")
        
        # Sonuçları logla
        for i, doc in enumerate(docs):
            print(f"[SEARCH] Sonuç {i+1}: {doc.metadata.get('source', 'N/A')} - "
                  f"Sınıf: {doc.metadata.get('sinif', 'N/A')} - "
                  f"Ders: {doc.metadata.get('ders', 'N/A')} - "
                  f"İçerik: {doc.page_content[:100]}...")
        
        return docs
        
    except Exception as e:
        print(f"[SEARCH ERROR] Arama hatası: {str(e)}")
        traceback.print_exc()
        return []

def should_load_books():
    """Kitapların yüklenip yüklenmediğini kontrol eder"""
    global vectorstore
    
    if not vectorstore:
        return True
    
    try:
        doc_count = vectorstore._collection.count()
        pdf_count = len(glob.glob("books/*.pdf"))
        
        print(f"[RAG] ChromaDB'de {doc_count} doküman var")
        print(f"[RAG] books/ klasöründe {pdf_count} PDF var")
        
        # Eğer hiç doküman yoksa veya çok az varsa yükle
        return doc_count < (pdf_count * 5)  # Her PDF en az 5 chunk üretmeli
        
    except Exception as e:
        print(f"[RAG] Kontrol hatası: {str(e)}")
        return True

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