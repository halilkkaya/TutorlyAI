"""
Hibrit RAG Arama Sistemi
Semantic Search + BM25 Keyword Search + Metadata Filtering kombinasyonu
Smart Cache sistemi ile BM25 Index Persistence desteği
Enhanced with Redis cache integration
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import traceback
import pickle
import json
import hashlib
import os
import glob
from pathlib import Path
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class HybridRetriever:
    """
    Hibrit arama sistemi - Semantic + Keyword + Metadata filtering
    Smart Cache sistemi ile BM25 Index Persistence desteği
    Enhanced with Redis cache integration
    """

    def __init__(self, vectorstore, embedding_model, use_redis_cache=True):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.bm25_index = None
        self.documents = []
        self.turkish_stop_words = self._get_turkish_stop_words()
        self.is_indexed = False

        # Cache system selection
        self.use_redis_cache = use_redis_cache
        self.redis_cache = None

        # File-based cache (fallback)
        self.storage_dir = Path("storage")
        self.storage_dir.mkdir(exist_ok=True)
        self.bm25_index_path = self.storage_dir / "bm25_index.pkl"
        self.index_meta_path = self.storage_dir / "index_meta.json"
        self.documents_cache_path = self.storage_dir / "documents_cache.pkl"

        # Initialize cache system
        self._initialize_cache_system()
        
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
    
    # ===== SMART CACHE SYSTEM =====
    
    def _initialize_cache_system(self):
        """Cache sistemini başlat - Redis ve file-based cache desteği"""
        try:
            logger.info("[CACHE] Smart cache sistemi başlatılıyor...")

            # Redis cache'i dene
            if self.use_redis_cache:
                try:
                    from .redis_cache_adapters import redis_bm25_cache
                    self.redis_cache = redis_bm25_cache
                    logger.info("[CACHE] Redis cache integration enabled")
                except ImportError as e:
                    logger.warning(f"[CACHE] Redis cache not available: {str(e)}")
                    self.use_redis_cache = False

            # Cache validation and loading
            cache_loaded = False

            # Redis cache'den yüklemeye çalış
            if self.use_redis_cache and self.redis_cache:
                if self._is_redis_cache_valid():
                    logger.info("[CACHE] Geçerli Redis cache bulundu, yükleniyor...")
                    if self._load_redis_cache():
                        logger.info(f"[CACHE] ✓ Redis cache başarıyla yüklendi: {len(self.documents)} döküman")
                        cache_loaded = True

            # Redis başarısız olursa file-based cache dene
            if not cache_loaded:
                if self._is_cache_valid():
                    logger.info("[CACHE] Geçerli file cache bulundu, yükleniyor...")
                    if self._load_cache():
                        logger.info(f"[CACHE] ✓ File cache başarıyla yüklendi: {len(self.documents)} döküman")
                        cache_loaded = True

            # Hiç cache bulunamadıysa yeni oluştur
            if not cache_loaded:
                logger.info("[CACHE] Cache bulunamadı, yeni index oluşturuluyor...")
                self._schedule_index_rebuild()

        except Exception as e:
            logger.error(f"[CACHE ERROR] Cache sistemi başlatma hatası: {str(e)}")
            traceback.print_exc()
    
    def _is_cache_valid(self) -> bool:
        """Cache'in geçerli olup olmadığını kontrol et"""
        try:
            # Cache dosyaları var mı?
            if not (self.bm25_index_path.exists() and 
                   self.index_meta_path.exists() and 
                   self.documents_cache_path.exists()):
                logger.warning("[CACHE] Cache dosyaları eksik")
                return False
            
            # Metadata'yı oku
            with open(self.index_meta_path, 'r', encoding='utf-8') as f:
                cache_meta = json.load(f)
            
            # Mevcut sistem durumunu hesapla
            current_signature = self._calculate_content_signature()
            cached_signature = cache_meta.get('content_signature', '')
            
            if current_signature != cached_signature:
                logger.warning(f"[CACHE] İçerik değişmiş - Cached: {cached_signature[:8]}..., Current: {current_signature[:8]}...")
                return False
            
            logger.info(f"[CACHE] ✓ Cache geçerli - Signature: {current_signature[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"[CACHE] Cache validasyon hatası: {str(e)}")
            return False
    
    def _calculate_content_signature(self) -> str:
        """Sistem içeriğinin signature'ını hesapla"""
        try:
            # PDF dosya bilgileri
            pdf_files = glob.glob("books/*.pdf")
            pdf_info = []
            for pdf_file in sorted(pdf_files):
                stat = os.stat(pdf_file)
                pdf_info.append({
                    'name': os.path.basename(pdf_file),
                    'size': stat.st_size,
                    'mtime': stat.st_mtime
                })
            
            # ChromaDB doküman sayısı
            chroma_count = 0
            try:
                chroma_count = self.vectorstore._collection.count()
            except:
                pass
            
            # Signature objesi
            signature_data = {
                'pdf_files': pdf_info,
                'chroma_doc_count': chroma_count,
                'version': '1.0'
            }
            
            # MD5 hash hesapla
            signature_str = json.dumps(signature_data, sort_keys=True)
            return hashlib.md5(signature_str.encode('utf-8')).hexdigest()
            
        except Exception as e:
            logger.error(f"[CACHE] Signature hesaplama hatası: {str(e)}")
            return str(datetime.now().timestamp())  # Fallback
    
    def _load_cache(self) -> bool:
        """Cache'den BM25 index ve dökümanları yükle"""
        try:
            logger.info("[CACHE] Cache dosyaları yükleniyor...")
            
            # BM25 index yükle
            with open(self.bm25_index_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            
            # Dökümanları yükle
            with open(self.documents_cache_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            # Metadata kontrol
            with open(self.index_meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            self.is_indexed = True
            
            logger.info(f"[CACHE] ✓ Cache yüklendi: {len(self.documents)} döküman, "
                  f"Created: {meta.get('created_at', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"[CACHE] Cache yükleme hatası: {str(e)}")
            traceback.print_exc()
            
            # Hatalı cache dosyalarını temizle
            self._clean_cache_files()
            return False
    
    def _save_cache(self) -> bool:
        """BM25 index ve dökümanları cache'e kaydet"""
        try:
            logger.info("[CACHE] Cache kaydediliyor...")
            
            # BM25 index kaydet
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump(self.bm25_index, f)
            
            # Dökümanları kaydet
            with open(self.documents_cache_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Metadata kaydet
            meta = {
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'content_signature': self._calculate_content_signature(),
                'document_count': len(self.documents),
                'bm25_terms_count': len(self.bm25_index.doc_freqs) if self.bm25_index else 0,
                'version': '1.0'
            }
            
            with open(self.index_meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[CACHE] ✓ Cache kaydedildi: {len(self.documents)} döküman")
            return True
            
        except Exception as e:
            logger.error(f"[CACHE] Cache kaydetme hatası: {str(e)}")
            traceback.print_exc()
            return False
    
    def _clean_cache_files(self):
        """Hatalı cache dosyalarını temizle"""
        try:
            for file_path in [self.bm25_index_path, self.documents_cache_path, self.index_meta_path]:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"[CACHE] Temizlendi: {file_path.name}")
        except Exception as e:
            logger.error(f"[CACHE] Temizleme hatası: {str(e)}")

    # ===== REDIS CACHE METHODS =====

    def _is_redis_cache_valid(self) -> bool:
        """Redis cache'in geçerli olup olmadığını kontrol et"""
        if not self.redis_cache:
            return False

        try:
            import asyncio
            current_signature = self._calculate_content_signature()

            # Async function'ı sync olarak çalıştır
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.redis_cache.is_cache_valid(current_signature)
                )
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[REDIS_CACHE] Validation error: {str(e)}")
            return False

    def _load_redis_cache(self) -> bool:
        """Redis cache'den BM25 index ve dökümanları yükle"""
        if not self.redis_cache:
            return False

        try:
            import asyncio
            logger.info("[REDIS_CACHE] Redis'den cache yükleniyor...")

            # Async function'ı sync olarak çalıştır
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.redis_cache.load_bm25_index())
            finally:
                loop.close()

            if result is not None:
                self.bm25_index, self.documents, metadata = result
                self.is_indexed = True

                logger.info(f"[REDIS_CACHE] ✓ Redis cache yüklendi: {len(self.documents)} döküman, "
                          f"Created: {metadata.get('created_at', 'N/A')}")
                return True
            else:
                logger.info("[REDIS_CACHE] Redis cache bulunamadı")
                return False

        except Exception as e:
            logger.error(f"[REDIS_CACHE] Load error: {str(e)}")
            return False

    def _save_redis_cache(self) -> bool:
        """BM25 index ve dökümanları Redis cache'e kaydet"""
        if not self.redis_cache:
            return False

        try:
            import asyncio
            logger.info("[REDIS_CACHE] Redis'e cache kaydediliyor...")

            # Metadata oluştur
            metadata = {
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'content_signature': self._calculate_content_signature(),
                'document_count': len(self.documents),
                'bm25_terms_count': len(self.bm25_index.doc_freqs) if self.bm25_index else 0,
                'version': '1.0'
            }

            # Async function'ı sync olarak çalıştır
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.redis_cache.save_bm25_index(self.bm25_index, self.documents, metadata)
                )
            finally:
                loop.close()

            if result:
                logger.info(f"[REDIS_CACHE] ✓ Redis cache kaydedildi: {len(self.documents)} döküman")
            else:
                logger.warning("[REDIS_CACHE] ⚠ Redis cache kaydetme başarısız")

            return result

        except Exception as e:
            logger.error(f"[REDIS_CACHE] Save error: {str(e)}")
            return False
    
    def _schedule_index_rebuild(self):
        """Index yeniden oluşturmayı planla (şimdilik sync)"""
        logger.info("[CACHE] Index yeniden oluşturuluyor...")
        self.build_bm25_index()
    
    def force_rebuild_cache(self):
        """Cache'i zorla yeniden oluştur"""
        logger.info("[CACHE] Cache zorla yeniden oluşturuluyor...")
        self._clean_cache_files()
        self.is_indexed = False
        self.bm25_index = None
        self.documents = []
        self.build_bm25_index()
    
    # ===== END SMART CACHE SYSTEM =====
    
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
        """BM25 indexini oluştur ve cache'e kaydet"""
        try:
            logger.info("[HYBRID] BM25 index oluşturuluyor...")
            
            # Tüm dökümanları al
            collection = self.vectorstore._collection
            all_docs = collection.get()
            
            if not all_docs or 'documents' not in all_docs:
                logger.warning("[HYBRID] Hiç döküman bulunamadı")
                return False
            
            # Dökümanları hazırla
            self.documents = []
            processed_texts = []
            
            logger.info(f"[HYBRID] {len(all_docs['documents'])} döküman işleniyor...")
            
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
                
                # Progress log (her 100 döküman)
                if (i + 1) % 100 == 0:
                    logger.info(f"[HYBRID] İşlendi: {i + 1}/{len(all_docs['documents'])}")
            
            # BM25 indexini oluştur
            if processed_texts:
                logger.info("[HYBRID] BM25Okapi index hesaplanıyor...")
                self.bm25_index = BM25Okapi(processed_texts)
                self.is_indexed = True
                
                logger.info(f"[HYBRID] ✓ BM25 index oluşturuldu: {len(self.documents)} döküman")

                # Cache'e kaydet - önce Redis'e sonra file'a
                cache_saved = False

                # Redis cache'e kaydet
                if self.use_redis_cache and self.redis_cache:
                    if self._save_redis_cache():
                        logger.info("[HYBRID] ✓ Index Redis cache'e kaydedildi")
                        cache_saved = True
                    else:
                        logger.warning("[HYBRID] ⚠ Redis cache kaydetme başarısız")

                # File cache'e kaydet (fallback)
                if self._save_cache():
                    logger.info("[HYBRID] ✓ Index file cache'e kaydedildi")
                    cache_saved = True
                else:
                    logger.warning("[HYBRID] ⚠ File cache kaydetme başarısız")

                if not cache_saved:
                    logger.warning("[HYBRID] ⚠ Hiçbir cache'e kaydetme başarısız")

                return True
            else:
                logger.warning("[HYBRID] İşlenecek metin bulunamadı")
                return False
                
        except Exception as e:
            logger.error(f"[HYBRID] BM25 index oluşturma hatası: {str(e)}")
            traceback.print_exc()
            
            # Hata durumunda cache'i temizle
            self._clean_cache_files()
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
            logger.error(f"[HYBRID] BM25 skor hesaplama hatası: {str(e)}")
            return []
    
    def _apply_metadata_filters(self, docs: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """Metadata filtrelerini uygula - Türkçe karakter normalize ile"""
        if not filters:
            return docs
        
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                
                doc_value = doc.metadata[key]
                filter_value = value
                
                # konu_slug için Türkçe karakter normalizasyonu
                if key == 'konu_slug' and isinstance(doc_value, str) and isinstance(filter_value, str):
                    doc_value = normalize_turkish_chars(doc_value).lower()
                    filter_value = normalize_turkish_chars(filter_value).lower()
                
                if doc_value != filter_value:
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
            logger.info(f"[HYBRID] Hibrit arama başlatılıyor: '{query}'")
            logger.info(f"[HYBRID] Ağırlıklar - Semantic: {semantic_weight}, BM25: {keyword_weight}")
            
            # Index yoksa cache'den yükle veya oluştur
            if not self.is_indexed:
                logger.info("[HYBRID] BM25 index yok, cache kontrol ediliyor...")
                
                # Cache'den yüklemeye çalış
                if self._is_cache_valid() and self._load_cache():
                    logger.info("[HYBRID] ✓ Cache'den yüklendi")
                else:
                    logger.warning("[HYBRID] Cache yüklenemedi, yeni index oluşturuluyor...")
                    if not self.build_bm25_index():
                        logger.warning("[HYBRID] BM25 index oluşturulamadı, sadece semantic search kullanılıyor")
                        return self._fallback_semantic_search(query, filters, k, score_threshold)
            
            # 1. Semantic search
            semantic_results = self._get_semantic_results(query, filters, k * 2)
            logger.info(f"[HYBRID] Semantic search: {len(semantic_results)} sonuç")
            
            # 2. BM25 keyword search
            bm25_results = self._calculate_bm25_scores(query, k * 2)
            logger.info(f"[HYBRID] BM25 search: {len(bm25_results)} sonuç")
            
            # 3. Metadata filtreleme (BM25 sonuçları için)
            if filters:
                bm25_docs = [doc for doc, score in bm25_results]
                filtered_bm25_docs = self._apply_metadata_filters(bm25_docs, filters)
                bm25_results = [(doc, score) for doc, score in bm25_results if doc in filtered_bm25_docs]
                logger.info(f"[HYBRID] BM25 filtered: {len(bm25_results)} sonuç")
            
            # 4. Skorları normalize et ve birleştir
            final_results = self._combine_and_rank_results(
                semantic_results, bm25_results, 
                semantic_weight, keyword_weight,
                k, score_threshold
            )
            
            logger.info(f"[HYBRID] Final results: {len(final_results)} sonuç")
            
            # Detaylı loglar
            for i, doc in enumerate(final_results[:5]):
                logger.info(f"[HYBRID] Sonuç {i+1}: {doc.metadata.get('source', 'N/A')} - "
                      f"Sınıf: {doc.metadata.get('sinif', 'N/A')} - "
                      f"Ders: {doc.metadata.get('ders', 'N/A')} - "
                      f"İçerik: {doc.page_content[:100]}...")
            
            return final_results
            
        except Exception as e:
            logger.error(f"[HYBRID] Hibrit arama hatası: {str(e)}")
            traceback.print_exc()
            # Hata durumunda semantic search'e geri dön
            return self._fallback_semantic_search(query, filters, k, score_threshold)
    
    def _get_semantic_results(self, query: str, filters: Optional[Dict[str, Any]], k: int) -> List[Tuple[Document, float]]:
        """Semantic search sonuçlarını al"""
        try:
            search_kwargs = {"k": k}
            
            # ChromaDB filtreleme formatı
            if filters:
                # Filtreleri normalize et (özellikle konu_slug için)
                normalized_filters = {}
                for key, value in filters.items():
                    if key == 'konu_slug' and isinstance(value, str):
                        normalized_filters[key] = normalize_turkish_chars(value).lower()
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
            
            # Skorlu arama
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, **search_kwargs)
            
            # ChromaDB distance'ı similarity'e çevir
            semantic_results = []
            for doc, distance in docs_with_scores:
                similarity = 1.0 - distance if distance <= 1.0 else 0.0
                semantic_results.append((doc, similarity))
            
            return semantic_results
            
        except Exception as e:
            logger.error(f"[HYBRID] Semantic search hatası: {str(e)}")
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
            logger.warning(f"[HYBRID] Hiç doküman threshold ({score_threshold}) değerini geçemedi")
        
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
        logger.info("[HYBRID] Fallback: Sadece semantic search kullanılıyor")
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
            logger.warning("[HYBRID] Threshold'u geçen doküman bulunamadı, boş liste döndürülüyor")
        
        return filtered_docs[:k]
