"""
Similarity-based Cache System for TutorlyAI
Implements fuzzy cache matching using query similarity
"""

import os
import json
import hashlib
import asyncio
import logging
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .redis_client import redis_client

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class SimilarityConfig:
    similarity_threshold: float = 0.80
    max_similar_queries: int = 100  # Her database'de max kaç query saklanacak
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    enable_similarity_cache: bool = True

class SimilarityBasedCache:
    """
    Similarity-based cache system
    Query'ler arasında semantic similarity hesaplayarak cache hit rate'i artırır
    """

    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or self._load_config_from_env()
        self.embedding_model = None
        self._model_loaded = False
        self._global_model_injected = False

    def _load_config_from_env(self) -> SimilarityConfig:
        """Environment variables'dan konfigürasyon yükle"""
        return SimilarityConfig(
            similarity_threshold=float(os.getenv("REDIS_SIMILARITY_THRESHOLD", "0.80")),
            max_similar_queries=int(os.getenv("REDIS_MAX_SIMILAR_QUERIES", "100")),
            enable_similarity_cache=os.getenv("REDIS_ENABLE_SIMILARITY_CACHE", "true").lower() == "true"
        )

    def set_global_embedding_model(self, embedding_model):
        """Global embedding model'i inject et (RAG system'den)"""
        if embedding_model and not self._global_model_injected:
            self.embedding_model = embedding_model
            self._model_loaded = True
            self._global_model_injected = True
            logger.info("[SIMILARITY] ✅ Global embedding model injected (no reload needed)")

    def _load_embedding_model(self):
        """Embedding modelini lazy loading ile yükle"""
        if not self._model_loaded:
            try:
                logger.warning("[SIMILARITY] Loading new embedding model instance")
                self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
                self._model_loaded = True
                logger.info(f"[SIMILARITY] New embedding model loaded: {self.config.embedding_model_name}")
            except Exception as e:
                logger.error(f"[SIMILARITY] Failed to load embedding model: {str(e)}")
                self.config.enable_similarity_cache = False

    def _normalize_query(self, query: str) -> str:
        """Query'yi normalize et"""
        if not query:
            return ""

        # Türkçe karakter normalizasyonu
        from .hybrid_retriever import normalize_turkish_chars
        normalized = normalize_turkish_chars(query.lower().strip())

        # Çoklu boşlukları tek boşluğa çevir
        import re
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized

    def _normalize_filters(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Filtreleri normalize et"""
        if not filters:
            return {}

        normalized = {}
        for key, value in filters.items():
            if isinstance(value, str):
                from .hybrid_retriever import normalize_turkish_chars
                normalized[key] = normalize_turkish_chars(value.lower().strip())
            else:
                normalized[key] = value

        # Sıralı dict oluştur
        return dict(sorted(normalized.items()))

    def _create_query_signature(self, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """Query için unique signature oluştur"""
        normalized_query = self._normalize_query(query)
        normalized_filters = self._normalize_filters(filters)

        signature_data = {
            "query": normalized_query,
            "filters": normalized_filters
        }

        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode('utf-8')).hexdigest()

    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Query için embedding hesapla"""
        if not self.config.enable_similarity_cache:
            return None

        self._load_embedding_model()

        if not self._model_loaded or not self.embedding_model:
            return None

        try:
            normalized_query = self._normalize_query(query)

            # HuggingFaceEmbeddings vs SentenceTransformer API uyumluluğu
            if hasattr(self.embedding_model, 'encode'):
                # SentenceTransformer API
                embedding = self.embedding_model.encode([normalized_query])
                return embedding[0]
            elif hasattr(self.embedding_model, 'embed_query'):
                # HuggingFaceEmbeddings API
                embedding = self.embedding_model.embed_query(normalized_query)
                return np.array(embedding)
            else:
                logger.error(f"[SIMILARITY] Unknown embedding model type: {type(self.embedding_model)}")
                return None

        except Exception as e:
            logger.error(f"[SIMILARITY] Embedding calculation error: {str(e)}")
            return None

    async def _store_query_metadata(self, cache_type: str, query: str, filters: Optional[Dict[str, Any]],
                                   cache_key: str, embedding: Optional[np.ndarray]):
        """Query metadata'sını Redis'e sakla"""
        try:
            metadata_key = f"tutorlyai:similarity:{cache_type}:metadata"

            query_metadata = {
                "query": self._normalize_query(query),
                "filters": self._normalize_filters(filters),
                "cache_key": cache_key,
                "embedding": embedding.tolist() if embedding is not None else None,
                "timestamp": asyncio.get_event_loop().time()
            }

            # Mevcut metadata'ları al
            existing_metadata = await redis_client.get_cache_async("session", metadata_key)
            if existing_metadata is None:
                existing_metadata = []
            elif not isinstance(existing_metadata, list):
                existing_metadata = []

            # Yeni metadata'yı ekle
            existing_metadata.append(query_metadata)

            # Max limit kontrolü
            if len(existing_metadata) > self.config.max_similar_queries:
                # En eskilerini sil (timestamp'e göre)
                existing_metadata.sort(key=lambda x: x.get('timestamp', 0))
                existing_metadata = existing_metadata[-self.config.max_similar_queries:]

            # Geri kaydet
            await redis_client.set_cache_async("session", metadata_key, existing_metadata,
                                              ttl=self.config.max_similar_queries * 60)  # 1 saat per query

            logger.debug(f"[SIMILARITY] Query metadata stored for {cache_type}")

        except Exception as e:
            logger.error(f"[SIMILARITY] Failed to store query metadata: {str(e)}")

    async def _find_similar_cached_query(self, cache_type: str, query: str,
                                        filters: Optional[Dict[str, Any]]) -> Optional[str]:
        """Benzer bir cached query bul"""
        if not self.config.enable_similarity_cache:
            return None

        try:
            # Query embedding'i hesapla
            query_embedding = self._get_query_embedding(query)
            if query_embedding is None:
                return None

            # Metadata'ları al
            metadata_key = f"tutorlyai:similarity:{cache_type}:metadata"
            existing_metadata = await redis_client.get_cache_async("session", metadata_key)

            if not existing_metadata or not isinstance(existing_metadata, list):
                return None

            normalized_filters = self._normalize_filters(filters)
            best_similarity = 0.0
            best_cache_key = None
            all_similarities = []  # Tüm similarity skorlarını topla

            for metadata in existing_metadata:
                try:
                    # Filter uyumluluğunu kontrol et
                    if metadata.get("filters") != normalized_filters:
                        continue

                    # Embedding similarity hesapla
                    cached_embedding = metadata.get("embedding")
                    if cached_embedding is None:
                        continue

                    cached_embedding_array = np.array(cached_embedding).reshape(1, -1)
                    query_embedding_array = query_embedding.reshape(1, -1)

                    similarity = cosine_similarity(query_embedding_array, cached_embedding_array)[0][0]

                    # Similarity skorunu kaydet (debugging için)
                    cached_query = metadata.get("query", "unknown")
                    all_similarities.append({
                        "cached_query": cached_query[:50] + "..." if len(cached_query) > 50 else cached_query,
                        "similarity": similarity,
                        "above_threshold": similarity >= self.config.similarity_threshold
                    })

                    if similarity >= self.config.similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_cache_key = metadata.get("cache_key")

                except Exception as e:
                    logger.warning(f"[SIMILARITY] Error processing metadata: {str(e)}")
                    continue

            # Detaylı similarity logları
            if all_similarities:
                # En yüksek similarity'yi bul
                max_similarity = max(all_similarities, key=lambda x: x["similarity"])

                logger.info(f"[SIMILARITY] Query: '{self._normalize_query(query)[:50]}...'")
                logger.info(f"[SIMILARITY] Threshold: {self.config.similarity_threshold:.2f}")
                logger.info(f"[SIMILARITY] Max similarity found: {max_similarity['similarity']:.4f} -> '{max_similarity['cached_query']}'")

                # Top 3 similarity skorlarını göster
                sorted_similarities = sorted(all_similarities, key=lambda x: x["similarity"], reverse=True)[:3]
                for i, sim_data in enumerate(sorted_similarities):
                    status = "✓ HIT" if sim_data["above_threshold"] else "✗ MISS"
                    logger.info(f"[SIMILARITY] #{i+1}: {sim_data['similarity']:.4f} {status} -> '{sim_data['cached_query']}'")

            if best_cache_key:
                logger.info(f"[SIMILARITY] ✅ Cache hit with similarity: {best_similarity:.4f} (threshold: {self.config.similarity_threshold:.2f})")
                return best_cache_key
            else:
                max_sim = max_similarity["similarity"] if all_similarities else 0.0
                logger.info(f"[SIMILARITY] ❌ No cache hit - max similarity: {max_sim:.4f} (threshold: {self.config.similarity_threshold:.2f})")

            return None

        except Exception as e:
            logger.error(f"[SIMILARITY] Similar query search error: {str(e)}")
            return None

    async def get_cached_result_with_similarity(self, cache_type: str, cache_key: str,
                                               query: str, filters: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Similarity-based cache lookup
        1. Exact match dene
        2. Similar match dene
        """
        try:
            # 1. Exact match dene
            exact_result = await redis_client.get_cache_async(cache_type, cache_key)
            if exact_result is not None:
                logger.debug(f"[SIMILARITY] Exact cache hit for {cache_type}")
                return exact_result

            # 2. Similar match dene (sadece similarity enabled ise)
            if self.config.enable_similarity_cache:
                similar_cache_key = await self._find_similar_cached_query(cache_type, query, filters)
                if similar_cache_key:
                    similar_result = await redis_client.get_cache_async(cache_type, similar_cache_key)
                    if similar_result is not None:
                        logger.info(f"[SIMILARITY] Similar cache hit for {cache_type}")
                        return similar_result

            logger.debug(f"[SIMILARITY] Cache miss for {cache_type}")
            return None

        except Exception as e:
            logger.error(f"[SIMILARITY] Cache lookup error: {str(e)}")
            return None

    async def set_cached_result_with_similarity(self, cache_type: str, cache_key: str,
                                               value: Any, query: str,
                                               filters: Optional[Dict[str, Any]] = None,
                                               ttl: Optional[int] = None) -> bool:
        """
        Similarity metadata ile cache'e kaydet
        """
        try:
            # Normal cache'e kaydet
            result = await redis_client.set_cache_async(cache_type, cache_key, value, ttl)

            if result and self.config.enable_similarity_cache:
                # Query embedding hesapla ve metadata kaydet
                embedding = self._get_query_embedding(query)
                if embedding is not None:
                    await self._store_query_metadata(cache_type, query, filters, cache_key, embedding)

            return result

        except Exception as e:
            logger.error(f"[SIMILARITY] Cache set error: {str(e)}")
            return False

    async def clear_similarity_metadata(self, cache_type: Optional[str] = None):
        """Similarity metadata'larını temizle"""
        try:
            if cache_type:
                metadata_key = f"tutorlyai:similarity:{cache_type}:metadata"
                await redis_client.delete_cache_async("session", metadata_key)
                logger.info(f"[SIMILARITY] Metadata cleared for {cache_type}")
            else:
                # Tüm similarity metadata'larını temizle
                for ct in ["query", "performance", "bm25", "session"]:
                    metadata_key = f"tutorlyai:similarity:{ct}:metadata"
                    await redis_client.delete_cache_async("session", metadata_key)
                logger.info("[SIMILARITY] All similarity metadata cleared")

        except Exception as e:
            logger.error(f"[SIMILARITY] Metadata clear error: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Similarity cache istatistikleri"""
        return {
            "similarity_threshold": self.config.similarity_threshold,
            "max_similar_queries": self.config.max_similar_queries,
            "embedding_model": self.config.embedding_model_name,
            "model_loaded": self._model_loaded,
            "similarity_enabled": self.config.enable_similarity_cache
        }

# Global instance
similarity_cache = SimilarityBasedCache()