from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from pathlib import Path
from tools.parse_filename import parse_filename_for_metadata
from tools.hybrid_retriever import HybridRetriever
from tools.security_utils import security_validator
from tools.database_pool import query_cache
import re
import traceback
from langchain.schema import Document
from PyPDF2 import PdfReader
from typing import Dict, Any, List, Optional
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vectorstore = None
embedding_model = None
text_splitter = None
hybrid_retriever = None

def normalize_turkish_chars(text: str) -> str:
    """Türkçe karakterleri normal harflere çevirir"""
    if not text:
        return text
    
    turkish_chars = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G', 
        'ı': 'i', 'I': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U'
    }
    
    normalized = text
    for turkish_char, normal_char in turkish_chars.items():
        normalized = normalized.replace(turkish_char, normal_char)
    
    return normalized


# RAG sistem fonksiyonları
def initialize_rag_system():
    """RAG sistemini başlatır"""
    global vectorstore, embedding_model, text_splitter, hybrid_retriever
    
    try:
        logger.info("[RAG] Sistem başlatılıyor... (logger ile)")
        device = "cuda"

        # Embedding modeli
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("[RAG] Embedding model yüklendi")
        
        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,      # Daha küçük chunk'lar
            chunk_overlap=100,   # Daha az overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        logger.info("[RAG] Text splitter hazırlandı")
        
        # ChromaDB
        vectorstore = Chroma(
            collection_name="enhanced_books_collection",
            embedding_function=embedding_model,
            persist_directory="./chroma_db_v2"  # Yeni versiyon için farklı klasör
        )
        logger.info("[RAG] ChromaDB bağlandı")
        
        # Hibrit retriever'ı başlat
        hybrid_retriever = HybridRetriever(vectorstore, embedding_model)
        logger.info("[RAG] Hibrit retriever başlatıldı")
        
        return True
        
    except Exception as e:
        logger.error(f"[RAG ERROR] Başlatma hatası: {str(e)}")
        return False


def should_load_books():
    """Kitapların yüklenip yüklenmediğini kontrol eder"""
    global vectorstore
    
    if not vectorstore:
        return True
    
    try:
        doc_count = vectorstore._collection.count()
        pdf_count = len(glob.glob("books/*.pdf"))
        
        logger.info(f"[RAG] ChromaDB'de {doc_count} doküman var")
        logger.info(f"[RAG] books/ klasöründe {pdf_count} PDF var")
        
        # Eğer hiç doküman yoksa veya çok az varsa yükle
        return doc_count < (pdf_count * 5)  # Her PDF en az 5 chunk üretmeli
        
    except Exception as e:
        logger.error(f"[RAG] Kontrol hatası: {str(e)}")
        return True


async def load_books_async():
    """Kitapları asenkron olarak yükler"""
    global vectorstore, text_splitter
    
    if not vectorstore or not text_splitter:
        logger.error("[RAG] Sistem başlatılmamış")
        return False
    
    try:
        logger.info("[RAG] Kitaplar yükleniyor...")
        
        # PDF dosyalarını bul
        pdf_files = glob.glob("books/*.pdf")   
        
        if not pdf_files:
            logger.error("[RAG] books/ klasöründe PDF bulunamadı")
            return False
        
        logger.info(f"[RAG] {len(pdf_files)} PDF dosyası bulundu")
        
        all_documents = []
        successful_files = 0
        
        for pdf_path in pdf_files:
            try:
                filename = Path(pdf_path).name
                logger.info(f"[RAG] İşleniyor: {filename}")

                # Güvenlik kontrolü - dosya yolu ve içerik
                try:
                    security_validator.validate_file_path(pdf_path)
                    security_validator.validate_file_content(pdf_path)
                    logger.info(f"[SECURITY] File validated: {filename}")
                except Exception as e:
                    logger.error(f"[SECURITY] File validation failed for {filename}: {str(e)}")
                    continue

                # Metadata çıkar
                metadata = parse_filename_for_metadata(filename)
                if not metadata:
                    logger.warning(f"[RAG] UYARI: {filename} format uyumsuz, atlanıyor")
                    continue

                # PDF oku
                reader = PdfReader(pdf_path)
                full_text = ""
                
                logger.info(f"[RAG] {filename}: {len(reader.pages)} sayfa okunuyor...")
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            # Metni temizle
                            text = re.sub(r'\s+', ' ', text).strip()
                            full_text += text + "\n"
                    except Exception as e:
                        logger.error(f"[RAG] Sayfa {page_num} okuma hatası: {e}")
                        continue
                
                if not full_text.strip():
                    logger.warning(f"[RAG] UYARI: {filename} metin çıkarılamadı")
                    continue
                
                # Chunk'lara ayır
                chunks = text_splitter.split_text(full_text)
                logger.info(f"[RAG] {filename}: {len(chunks)} chunk oluşturuldu")
                
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
                logger.info(f"[RAG] {filename} başarıyla işlendi")
                
            except Exception as e:
                logger.error(f"[RAG] {pdf_path} işlenirken hata: {str(e)}")
                continue
        
        if all_documents:
            logger.info(f"[RAG] {len(all_documents)} doküman ChromaDB'ye ekleniyor...")
            
            # Batch olarak ekle (büyük dosyalar için)
            batch_size = 100
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                vectorstore.add_documents(batch)
                logger.info(f"[RAG] Batch {i//batch_size + 1}/{(len(all_documents)-1)//batch_size + 1} eklendi")
            
            logger.info(f"[RAG] BAŞARILI: {successful_files} dosya, {len(all_documents)} chunk yüklendi")
            return True
        else:
            logger.error("[RAG] Hiç doküman oluşturulamadı")
            return False
            
    except Exception as e:
        logger.error(f"[RAG] Kitap yükleme genel hatası: {str(e)}")
        traceback.print_exc()
        return False

 
async def _search_books_enhanced_impl(query: str, filters: Optional[Dict[str, Any]] = None,
                                     k: int = 5, score_threshold: float = 0.5,
                                     use_hybrid: bool = True,
                                     semantic_weight: float = 0.7,
                                     keyword_weight: float = 0.3) -> List[Document]:
    """
    Gelişmiş kitap arama fonksiyonu - Hibrit arama (Semantic + BM25) ile
    
    Args:
        query: Arama sorgusu
        filters: Metadata filtreleri (sadece arama alanını belirler)
        k: Maksimum sonuç sayısı
        score_threshold: Minimum benzerlik skoru (0.0-1.0 arası)
        use_hybrid: Hibrit arama kullanılsın mı (varsayılan: True)
        semantic_weight: Semantic search ağırlığı (varsayılan: 0.7)
        keyword_weight: BM25 keyword search ağırlığı (varsayılan: 0.3)
    """
    global vectorstore, hybrid_retriever
    
    if not vectorstore:
        logger.error("[SEARCH] RAG sistemi aktif değil")
        return []
    
    try:
        logger.info(f"[SEARCH] Arama başlatılıyor: '{query}'")
        logger.info(f"[SEARCH] Filtreler: {filters}")
        logger.info(f"[SEARCH] Score threshold: {score_threshold}")
        
        # Arama parametreleri
        search_kwargs = {"k": k * 2}  # Daha fazla sonuç al, sonra filtrele
        
        # ChromaDB filtreleme formatı (sadece arama alanını belirler)
        if filters:
            # Filtreleri normalize et (özellikle konu_slug için)
            normalized_filters = {}
            for key, value in filters.items():
                if key == 'konu_slug' and isinstance(value, str):
                    # konu_slug'ı Türkçe karakterlerden arındır
                    normalized_value = normalize_turkish_chars(value).lower()
                    normalized_filters[key] = normalized_value
                    logger.info(f"[SEARCH] konu_slug normalize edildi: '{value}' → '{normalized_value}'")
                else:
                    normalized_filters[key] = value
            
            if len(normalized_filters) > 1:
                where_clause = {
                    "$and": [
                        {key: {"$eq": value}} for key, value in normalized_filters.items()
                    ]
                }
            else:
                key, value = list(normalized_filters.items())[0]
                where_clause = {key: {"$eq": value}}
            
            search_kwargs["filter"] = where_clause
            logger.info(f"[SEARCH] ChromaDB filtre: {where_clause}")
        
        # Hibrit arama kullan
        if use_hybrid and hybrid_retriever:
            logger.info("[SEARCH] Hibrit arama modeli kullanılıyor...")
            
            # Hibrit arama için BM25 index hazırla
            if not hybrid_retriever.is_indexed:
                logger.info("[SEARCH] BM25 index oluşturuluyor...")
                hybrid_retriever.build_bm25_index()
            
            # 1. Semantic search (filtrelenmiş alanda)
            semantic_docs_with_scores = vectorstore.similarity_search_with_score(query, **search_kwargs)
            logger.info(f"[SEARCH] Semantic sonuç: {len(semantic_docs_with_scores)}")
            
            # 2. BM25 search (tüm dokümanlar üzerinde, sonra filtrele)
            bm25_results = hybrid_retriever._calculate_bm25_scores(query, k * 2)
            
            # BM25 sonuçlarını metadata ile filtrele
            if filters:
                filtered_bm25_results = []
                for doc, score in bm25_results:
                    match = True
                    for filter_key, filter_value in normalized_filters.items():
                        doc_value = doc.metadata.get(filter_key)
                        
                        # Eğer konu_slug karşılaştırması yapılıyorsa normalize et
                        if filter_key == 'konu_slug' and isinstance(doc_value, str):
                            doc_value = normalize_turkish_chars(doc_value).lower()
                        
                        if doc_value != filter_value:
                            match = False
                            break
                    if match:
                        filtered_bm25_results.append((doc, score))
                bm25_results = filtered_bm25_results
            
            logger.info(f"[SEARCH] BM25 sonuç: {len(bm25_results)}")
            
            # 3. Skorları birleştir ve hibrit skor hesapla
            final_results = []
            processed_docs = {}
            
            # Semantic sonuçları işle
            for doc, distance in semantic_docs_with_scores:
                doc_id = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', 0)}"
                semantic_score = 1.0 - distance if distance <= 1.0 else 0.0
                processed_docs[doc_id] = {
                    'doc': doc,
                    'semantic_score': semantic_score,
                    'bm25_score': 0.0
                }
            
            # BM25 sonuçlarını işle ve normalize et
            if bm25_results:
                bm25_scores = [score for _, score in bm25_results]
                max_bm25 = max(bm25_scores) if bm25_scores else 1.0
                min_bm25 = min(bm25_scores) if bm25_scores else 0.0
                
                for doc, score in bm25_results:
                    doc_id = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', 0)}"
                    normalized_bm25 = (score - min_bm25) / (max_bm25 - min_bm25) if max_bm25 > min_bm25 else 0.0
                    
                    if doc_id in processed_docs:
                        processed_docs[doc_id]['bm25_score'] = normalized_bm25
                    else:
                        processed_docs[doc_id] = {
                            'doc': doc,
                            'semantic_score': 0.0,
                            'bm25_score': normalized_bm25
                        }
            
            # Hibrit skor hesapla ve threshold'a göre filtrele
            for doc_id, scores in processed_docs.items():
                hybrid_score = (semantic_weight * scores['semantic_score'] + 
                              keyword_weight * scores['bm25_score'])
                
                if hybrid_score >= score_threshold:
                    doc = scores['doc']
                    doc.metadata['hybrid_score'] = hybrid_score
                    doc.metadata['semantic_score'] = scores['semantic_score']
                    doc.metadata['bm25_score'] = scores['bm25_score']
                    final_results.append((doc, hybrid_score))
                    
                    logger.info(f"[SEARCH] ✓ Kabul: {doc.metadata.get('source', 'N/A')} - "
                          f"Hibrit: {hybrid_score:.3f} (S:{scores['semantic_score']:.3f}, BM25:{scores['bm25_score']:.3f})")
                else:
                    logger.info(f"[SEARCH] ✗ Ret: {scores['doc'].metadata.get('source', 'N/A')} - "
                          f"Hibrit: {hybrid_score:.3f} (threshold: {score_threshold})")
            
            # Hibrit skora göre sırala
            final_results.sort(key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in final_results[:k]]
            
        else:
            # Geleneksel semantic search
            logger.info("[SEARCH] Geleneksel semantic search kullanılıyor...")
            docs_with_scores = vectorstore.similarity_search_with_score(query, **search_kwargs)
            
            final_docs = []
            for doc, score in docs_with_scores:
                similarity_score = 1.0 - score if score <= 1.0 else 0.0
                
                if similarity_score >= score_threshold:
                    doc.metadata['hybrid_score'] = similarity_score
                    doc.metadata['semantic_score'] = similarity_score
                    doc.metadata['bm25_score'] = 0.0
                    final_docs.append(doc)
                    
                    logger.info(f"[SEARCH] ✓ Kabul: {doc.metadata.get('source', 'N/A')} - "
                          f"Score: {similarity_score:.3f}")
                else:
                    logger.info(f"[SEARCH] ✗ Ret: {doc.metadata.get('source', 'N/A')} - "
                          f"Score: {similarity_score:.3f} (threshold: {score_threshold})")
            
            final_docs = final_docs[:k]
        
        logger.info(f"[SEARCH] Final: {len(final_docs)} sonuç döndürülüyor")
        
        # Eğer hiç sonuç yoksa Generate API'nin fallback mekanizması devreye girecek
        if not final_docs:
            logger.info(f"[SEARCH] Hiç doküman threshold ({score_threshold}) değerini geçemedi, boş liste döndürülüyor")
        
        return final_docs

    except Exception as e:
        logger.error(f"[SEARCH ERROR] Arama hatası: {str(e)}")
        traceback.print_exc()
        return []

async def search_books_enhanced_async(query: str, filters: Optional[Dict[str, Any]] = None,
                                    k: int = 5, score_threshold: float = 0.5,
                                    use_hybrid: bool = True,
                                    semantic_weight: float = 0.7,
                                    keyword_weight: float = 0.3) -> List[Document]:
    """
    Async cache-enabled search function
    """
    import json

    # Cache key oluştur
    cache_data = {
        "query": query,
        "filters": filters or {},
        "k": k,
        "score_threshold": score_threshold,
        "use_hybrid": use_hybrid,
        "semantic_weight": semantic_weight,
        "keyword_weight": keyword_weight
    }
    cache_key = json.dumps(cache_data, sort_keys=True)

    try:
        # Async cache kullan
        return await query_cache.get_or_execute_query(
            cache_key,
            _search_books_enhanced_impl,
            query, filters, k, score_threshold, use_hybrid, semantic_weight, keyword_weight
        )
    except Exception as e:
        logger.error(f"[SEARCH CACHE ERROR] Cache execution failed: {str(e)}")
        # Cache başarısız olursa direct implementation'ı çağır
        return await _search_books_enhanced_impl(
            query, filters, k, score_threshold, use_hybrid, semantic_weight, keyword_weight
        )

def search_books_enhanced(query: str, filters: Optional[Dict[str, Any]] = None,
                          k: int = 5, score_threshold: float = 0.5,
                          use_hybrid: bool = True,
                          semantic_weight: float = 0.7,
                          keyword_weight: float = 0.3) -> List[Document]:
    """
    Sync wrapper - cache olmadan direct implementation
    FastAPI event loop conflict'ini önlemek için cache devre dışı
    """
    import asyncio

    # Direct implementation çağır (cache olmadan)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_search_books_enhanced_impl(
                query, filters, k, score_threshold, use_hybrid, semantic_weight, keyword_weight
            ))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"[SEARCH ERROR] Search execution failed: {str(e)}")
        return []



