from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from pathlib import Path
from tools.parse_filename import parse_filename_for_metadata
import re
import traceback
from langchain.schema import Document
from PyPDF2 import PdfReader
from typing import Dict, Any, List, Optional

vectorstore = None
embedding_model = None
text_splitter = None


# RAG sistem fonksiyonları
def initialize_rag_system():
    """RAG sistemini başlatır"""
    global vectorstore, embedding_model, text_splitter
    
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
                          k: int = 5, score_threshold: float = 0.5) -> List[Document]:
    """
    Gelişmiş kitap arama fonksiyonu - Benzerlik skoruna göre filtreleme ile
    
    Args:
        query: Arama sorgusu
        filters: Metadata filtreleri
        k: Maksimum sonuç sayısı
        score_threshold: Minimum benzerlik skoru (0.0-1.0 arası)
    """
    global vectorstore
    
    if not vectorstore:
        print("[SEARCH] RAG sistemi aktif değil")
        return []
    
    try:
        print(f"[SEARCH] Arama başlatılıyor: '{query}'")
        print(f"[SEARCH] Filtreler: {filters}")
        print(f"[SEARCH] Score threshold: {score_threshold}")
        
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
        
        # Skorlu arama yap
        docs_with_scores = vectorstore.similarity_search_with_score(query, **search_kwargs)
        
        # Score threshold'a göre filtrele
        filtered_docs = []
        for doc, score in docs_with_scores:
            # ChromaDB distance kullanır (düşük = benzer), score'a çevirelim
            similarity_score = 1.0 - score if score <= 1.0 else 0.0
            
            if similarity_score >= score_threshold:
                filtered_docs.append(doc)
                print(f"[SEARCH] ✓ Kabul: {doc.metadata.get('source', 'N/A')} - "
                      f"Score: {similarity_score:.3f} - "
                      f"Sınıf: {doc.metadata.get('sinif', 'N/A')} - "
                      f"Ders: {doc.metadata.get('ders', 'N/A')}")
            else:
                print(f"[SEARCH] ✗ Ret: {doc.metadata.get('source', 'N/A')} - "
                      f"Score: {similarity_score:.3f} (threshold: {score_threshold})")
        
        print(f"[SEARCH] {len(docs_with_scores)} sonuçtan {len(filtered_docs)} tanesi threshold'u geçti")
        
        # Sonuçları içerik preview ile logla
        for i, doc in enumerate(filtered_docs):
            print(f"[SEARCH] Final Sonuç {i+1}: {doc.metadata.get('source', 'N/A')} - "
                  f"İçerik: {doc.page_content[:100]}...")
        
        return filtered_docs
        
    except Exception as e:
        print(f"[SEARCH ERROR] Arama hatası: {str(e)}")
        traceback.print_exc()
        return []

