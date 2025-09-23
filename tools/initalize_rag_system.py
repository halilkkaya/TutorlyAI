from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from pathlib import Path
from tools.parse_filename import parse_filename_for_metadata
from tools.hybrid_retriever import HybridRetriever
import re
import traceback
from langchain.schema import Document
from PyPDF2 import PdfReader
from typing import Dict, Any, List, Optional

vectorstore = None
embedding_model = None
text_splitter = None
hybrid_retriever = None


# RAG sistem fonksiyonları
def initialize_rag_system():
    """RAG sistemini başlatır"""
    global vectorstore, embedding_model, text_splitter, hybrid_retriever
    
    try:
        print("[RAG] Sistem başlatılıyor...")
        device = "cpu"

        # Embedding modeli
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': device},
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
        
        # Hibrit retriever'ı başlat
        hybrid_retriever = HybridRetriever(vectorstore, embedding_model)
        print("[RAG] Hibrit retriever başlatıldı")
        
        return True
        
    except Exception as e:
        print(f"[RAG ERROR] Başlatma hatası: {str(e)}")
        return False


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

 
def search_books_enhanced(query: str, filters: Optional[Dict[str, Any]] = None, 
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
        print("[SEARCH] RAG sistemi aktif değil")
        return []
    
    try:
        print(f"[SEARCH] Arama başlatılıyor: '{query}'")
        print(f"[SEARCH] Filtreler: {filters}")
        print(f"[SEARCH] Score threshold: {score_threshold}")
        
        # Arama parametreleri
        search_kwargs = {"k": k * 2}  # Daha fazla sonuç al, sonra filtrele
        
        # ChromaDB filtreleme formatı (sadece arama alanını belirler)
        if filters:
            if len(filters) > 1:
                where_clause = {
                    "$and": [
                        {key: {"$eq": value}} for key, value in filters.items()
                    ]
                }
            else:
                key, value = list(filters.items())[0]
                where_clause = {key: {"$eq": value}}
            
            search_kwargs["filter"] = where_clause
            print(f"[SEARCH] ChromaDB filtre: {where_clause}")
        
        # Hibrit arama kullan
        if use_hybrid and hybrid_retriever:
            print("[SEARCH] Hibrit arama modeli kullanılıyor...")
            
            # Hibrit arama için BM25 index hazırla
            if not hybrid_retriever.is_indexed:
                print("[SEARCH] BM25 index oluşturuluyor...")
                hybrid_retriever.build_bm25_index()
            
            # 1. Semantic search (filtrelenmiş alanda)
            semantic_docs_with_scores = vectorstore.similarity_search_with_score(query, **search_kwargs)
            print(f"[SEARCH] Semantic sonuç: {len(semantic_docs_with_scores)}")
            
            # 2. BM25 search (tüm dokümanlar üzerinde, sonra filtrele)
            bm25_results = hybrid_retriever._calculate_bm25_scores(query, k * 2)
            
            # BM25 sonuçlarını metadata ile filtrele
            if filters:
                filtered_bm25_results = []
                for doc, score in bm25_results:
                    match = True
                    for filter_key, filter_value in filters.items():
                        if doc.metadata.get(filter_key) != filter_value:
                            match = False
                            break
                    if match:
                        filtered_bm25_results.append((doc, score))
                bm25_results = filtered_bm25_results
            
            print(f"[SEARCH] BM25 sonuç: {len(bm25_results)}")
            
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
                    
                    print(f"[SEARCH] ✓ Kabul: {doc.metadata.get('source', 'N/A')} - "
                          f"Hibrit: {hybrid_score:.3f} (S:{scores['semantic_score']:.3f}, BM25:{scores['bm25_score']:.3f})")
                else:
                    print(f"[SEARCH] ✗ Ret: {scores['doc'].metadata.get('source', 'N/A')} - "
                          f"Hibrit: {hybrid_score:.3f} (threshold: {score_threshold})")
            
            # Hibrit skora göre sırala
            final_results.sort(key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in final_results[:k]]
            
        else:
            # Geleneksel semantic search
            print("[SEARCH] Geleneksel semantic search kullanılıyor...")
            docs_with_scores = vectorstore.similarity_search_with_score(query, **search_kwargs)
            
            final_docs = []
            for doc, score in docs_with_scores:
                similarity_score = 1.0 - score if score <= 1.0 else 0.0
                
                if similarity_score >= score_threshold:
                    doc.metadata['hybrid_score'] = similarity_score
                    doc.metadata['semantic_score'] = similarity_score
                    doc.metadata['bm25_score'] = 0.0
                    final_docs.append(doc)
                    
                    print(f"[SEARCH] ✓ Kabul: {doc.metadata.get('source', 'N/A')} - "
                          f"Score: {similarity_score:.3f}")
                else:
                    print(f"[SEARCH] ✗ Ret: {doc.metadata.get('source', 'N/A')} - "
                          f"Score: {similarity_score:.3f} (threshold: {score_threshold})")
            
            final_docs = final_docs[:k]
        
        print(f"[SEARCH] Final: {len(final_docs)} sonuç döndürülüyor")
        
        # Eğer hiç sonuç yoksa Generate API'nin fallback mekanizması devreye girecek
        if not final_docs:
            print(f"[SEARCH] Hiç doküman threshold ({score_threshold}) değerini geçemedi, boş liste döndürülüyor")
        
        return final_docs
        
    except Exception as e:
        print(f"[SEARCH ERROR] Arama hatası: {str(e)}")
        traceback.print_exc()
        return []



