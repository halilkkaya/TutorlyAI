"""
Hibrit RAG Arama Sistemi
Semantic Search + BM25 Keyword Search + Metadata Filtering kombinasyonu
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import traceback

class HybridRetriever:
    """
    Hibrit arama sistemi - Semantic + Keyword + Metadata filtering
    """
    
    def __init__(self, vectorstore, embedding_model):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.bm25_index = None
        self.documents = []
        self.turkish_stop_words = self._get_turkish_stop_words()
        self.is_indexed = False
        
    def _get_turkish_stop_words(self) -> set:
        """Türkçe stop words listesi"""
        turkish_stops = {
            'bir', 'bu', 'da', 'de', 'den', 'dır', 'dir', 'için', 'ile', 'ise', 'ki', 
            'mi', 'ne', 'olan', 'olarak', 've', 'ya', 'ile', 'gibi', 'kadar', 'sonra',
            'önce', 'arasında', 'üzerine', 'altında', 'içinde', 'dışında', 'karşı',
            'göre', 'doğru', 'her', 'hiç', 'çok', 'az', 'daha', 'en', 'şu', 'o',
            'ben', 'sen', 'biz', 'siz', 'onlar', 'bunlar', 'şunlar', 'hangi',
            'nasıl', 'neden', 'niçin', 'nerede', 'ne zaman', 'kim', 'kimin',
            'ama', 'ancak', 'fakat', 'lakin', 'ise', 'eğer', 'şayet', 'madem',
            'çünkü', 'zira', 'hatta', 'hâlbuki', 'oysa', 'yalnız', 'sadece'
        }
        return turkish_stops
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Metni BM25 için ön işleme"""
        if not text:
            return []
        
        # Küçük harf yapma ve noktalama işaretlerini temizleme
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Kelimelere ayırma
        words = text.split()
        
        # Stop words ve çok kısa kelimeleri filtreleme
        filtered_words = [
            word for word in words 
            if len(word) > 2 and word not in self.turkish_stop_words
        ]
        
        return filtered_words
    
    def build_bm25_index(self):
        """BM25 indexini oluştur"""
        try:
            print("[HYBRID] BM25 index oluşturuluyor...")
            
            # Tüm dökümanları al
            collection = self.vectorstore._collection
            all_docs = collection.get()
            
            if not all_docs or 'documents' not in all_docs:
                print("[HYBRID] Hiç döküman bulunamadı")
                return False
            
            # Dökümanları hazırla
            self.documents = []
            processed_texts = []
            
            for i, (doc_text, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                # Document objesi oluştur
                doc = Document(
                    page_content=doc_text,
                    metadata=metadata or {}
                )
                self.documents.append(doc)
                
                # BM25 için metni işle
                processed_text = self._preprocess_text(doc_text)
                processed_texts.append(processed_text)
            
            # BM25 indexini oluştur
            if processed_texts:
                self.bm25_index = BM25Okapi(processed_texts)
                self.is_indexed = True
                print(f"[HYBRID] BM25 index oluşturuldu: {len(self.documents)} döküman")
                return True
            else:
                print("[HYBRID] İşlenecek metin bulunamadı")
                return False
                
        except Exception as e:
            print(f"[HYBRID] BM25 index oluşturma hatası: {str(e)}")
            traceback.print_exc()
            return False
    
    def _calculate_bm25_scores(self, query: str, k: int = 50) -> List[Tuple[Document, float]]:
        """BM25 skorlarını hesapla"""
        if not self.is_indexed or not self.bm25_index:
            return []
        
        try:
            # Query'yi işle
            processed_query = self._preprocess_text(query)
            if not processed_query:
                return []
            
            # BM25 skorlarını hesapla
            scores = self.bm25_index.get_scores(processed_query)
            
            # Skor-döküman çiftlerini oluştur
            doc_scores = [
                (self.documents[i], float(score))
                for i, score in enumerate(scores)
                if score > 0  # Sadece pozitif skorları al
            ]
            
            # Skora göre sırala ve en iyi k tanesini al
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores[:k]
            
        except Exception as e:
            print(f"[HYBRID] BM25 skor hesaplama hatası: {str(e)}")
            return []
    
    def _apply_metadata_filters(self, docs: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """Metadata filtrelerini uygula"""
        if not filters:
            return docs
        
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def hybrid_search(self, 
                     query: str, 
                     filters: Optional[Dict[str, Any]] = None,
                     k: int = 5,
                     semantic_weight: float = 0.7,
                     keyword_weight: float = 0.3,
                     score_threshold: float = 0.3) -> List[Document]:
        """
        Hibrit arama: Semantic + BM25 + Metadata filtering
        
        Args:
            query: Arama sorgusu
            filters: Metadata filtreleri
            k: Döndürülecek sonuç sayısı
            semantic_weight: Semantic search ağırlığı (0-1)
            keyword_weight: BM25 ağırlığı (0-1)
            score_threshold: Minimum skor threshold'u
        """
        try:
            print(f"[HYBRID] Hibrit arama başlatılıyor: '{query}'")
            print(f"[HYBRID] Ağırlıklar - Semantic: {semantic_weight}, BM25: {keyword_weight}")
            
            # Index yoksa oluştur
            if not self.is_indexed:
                print("[HYBRID] BM25 index bulunamadı, oluşturuluyor...")
                if not self.build_bm25_index():
                    print("[HYBRID] BM25 index oluşturulamadı, sadece semantic search kullanılıyor")
                    return self._fallback_semantic_search(query, filters, k, score_threshold)
            
            # 1. Semantic search
            semantic_results = self._get_semantic_results(query, filters, k * 2)
            print(f"[HYBRID] Semantic search: {len(semantic_results)} sonuç")
            
            # 2. BM25 keyword search
            bm25_results = self._calculate_bm25_scores(query, k * 2)
            print(f"[HYBRID] BM25 search: {len(bm25_results)} sonuç")
            
            # 3. Metadata filtreleme (BM25 sonuçları için)
            if filters:
                bm25_docs = [doc for doc, score in bm25_results]
                filtered_bm25_docs = self._apply_metadata_filters(bm25_docs, filters)
                bm25_results = [(doc, score) for doc, score in bm25_results if doc in filtered_bm25_docs]
                print(f"[HYBRID] BM25 filtered: {len(bm25_results)} sonuç")
            
            # 4. Skorları normalize et ve birleştir
            final_results = self._combine_and_rank_results(
                semantic_results, bm25_results, 
                semantic_weight, keyword_weight,
                k, score_threshold
            )
            
            print(f"[HYBRID] Final results: {len(final_results)} sonuç")
            
            # Detaylı loglar
            for i, doc in enumerate(final_results[:5]):
                print(f"[HYBRID] Sonuç {i+1}: {doc.metadata.get('source', 'N/A')} - "
                      f"Sınıf: {doc.metadata.get('sinif', 'N/A')} - "
                      f"Ders: {doc.metadata.get('ders', 'N/A')} - "
                      f"İçerik: {doc.page_content[:100]}...")
            
            return final_results
            
        except Exception as e:
            print(f"[HYBRID] Hibrit arama hatası: {str(e)}")
            traceback.print_exc()
            # Hata durumunda semantic search'e geri dön
            return self._fallback_semantic_search(query, filters, k, score_threshold)
    
    def _get_semantic_results(self, query: str, filters: Optional[Dict[str, Any]], k: int) -> List[Tuple[Document, float]]:
        """Semantic search sonuçlarını al"""
        try:
            search_kwargs = {"k": k}
            
            # ChromaDB filtreleme formatı
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
            
            # Skorlu arama
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, **search_kwargs)
            
            # ChromaDB distance'ı similarity'e çevir
            semantic_results = []
            for doc, distance in docs_with_scores:
                similarity = 1.0 - distance if distance <= 1.0 else 0.0
                semantic_results.append((doc, similarity))
            
            return semantic_results
            
        except Exception as e:
            print(f"[HYBRID] Semantic search hatası: {str(e)}")
            return []
    
    def _combine_and_rank_results(self, 
                                 semantic_results: List[Tuple[Document, float]],
                                 bm25_results: List[Tuple[Document, float]],
                                 semantic_weight: float,
                                 keyword_weight: float,
                                 k: int,
                                 score_threshold: float) -> List[Document]:
        """Semantic ve BM25 sonuçlarını birleştir ve sırala"""
        
        # Skorları normalize et
        semantic_scores = self._normalize_scores([score for _, score in semantic_results])
        bm25_scores = self._normalize_scores([score for _, score in bm25_results])
        
        # Döküman-skor haritası oluştur
        combined_scores = {}
        
        # Semantic sonuçları ekle
        for i, (doc, _) in enumerate(semantic_results):
            doc_id = self._get_doc_id(doc)
            combined_scores[doc_id] = {
                'doc': doc,
                'semantic': semantic_scores[i],
                'bm25': 0.0
            }
        
        # BM25 sonuçları ekle/güncelle
        for i, (doc, _) in enumerate(bm25_results):
            doc_id = self._get_doc_id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id]['bm25'] = bm25_scores[i]
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'semantic': 0.0,
                    'bm25': bm25_scores[i]
                }
        
        # Hibrit skoru hesapla
        final_results = []
        for doc_id, scores in combined_scores.items():
            hybrid_score = (
                semantic_weight * scores['semantic'] + 
                keyword_weight * scores['bm25']
            )
            
            if hybrid_score >= score_threshold:
                doc = scores['doc']
                doc.metadata['hybrid_score'] = hybrid_score
                doc.metadata['semantic_score'] = scores['semantic']
                doc.metadata['bm25_score'] = scores['bm25']
                final_results.append((doc, hybrid_score))
        
        # Hibrit skora göre sırala
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Eğer hiç sonuç yoksa log ekle
        if not final_results:
            print(f"[HYBRID] Hiç doküman threshold ({score_threshold}) değerini geçemedi")
        
        return [doc for doc, _ in final_results[:k]]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Skorları 0-1 arasında normalize et"""
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def _get_doc_id(self, doc: Document) -> str:
        """Döküman için unique ID oluştur"""
        return f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', 0)}"
    
    def _fallback_semantic_search(self, query: str, filters: Optional[Dict[str, Any]], k: int, score_threshold: float) -> List[Document]:
        """BM25 başarısız olursa semantic search'e geri dön - threshold düşürme YOK"""
        print("[HYBRID] Fallback: Sadece semantic search kullanılıyor")
        semantic_results = self._get_semantic_results(query, filters, k)
        
        filtered_docs = []
        for doc, score in semantic_results:
            if score >= score_threshold:
                doc.metadata['hybrid_score'] = score
                doc.metadata['semantic_score'] = score
                doc.metadata['bm25_score'] = 0.0
                filtered_docs.append(doc)
        
        # Threshold'u geçen doküman yoksa boş liste döndür
        # Generate API kendi fallback mekanizmasını kullanacak
        if not filtered_docs:
            print("[HYBRID] Threshold'u geçen doküman bulunamadı, boş liste döndürülüyor")
        
        return filtered_docs[:k]
