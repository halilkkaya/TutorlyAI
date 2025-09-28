"""
Redis Cache Adapters for TutorlyAI
Adapts existing cache systems to use Redis as backend
"""

import asyncio
import json
import pickle
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass

from .redis_client import redis_client

logger = logging.getLogger(__name__)

# ============================
# REDIS MEMORY CACHE ADAPTER
# ============================

class RedisMemoryCache:
    """
    Redis-backed version of MemoryCache
    Compatible API with existing MemoryCache but uses Redis
    """

    def __init__(self, cache_type: str = "performance"):
        self.cache_type = cache_type
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "current_size": 0,
            "memory_usage_mb": 0
        }

    async def get(self, key: str) -> Optional[Any]:
        """Cache'den değer al - Redis'den"""
        try:
            cache_key = redis_client.generate_cache_key(f"memory_cache", key)
            result = await redis_client.get_cache_async(self.cache_type, cache_key)

            if result is not None:
                self.stats["hits"] += 1
                logger.debug(f"[REDIS_CACHE] Hit: {key}")
                return result
            else:
                self.stats["misses"] += 1
                logger.debug(f"[REDIS_CACHE] Miss: {key}")
                return None

        except Exception as e:
            logger.error(f"[REDIS_CACHE] Get error: {str(e)}")
            self.stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Cache'e değer ekle - Redis'e"""
        try:
            cache_key = redis_client.generate_cache_key(f"memory_cache", key)
            ttl = ttl_seconds or redis_client.get_ttl_for_cache_type(self.cache_type)

            result = await redis_client.set_cache_async(self.cache_type, cache_key, value, ttl)

            if result:
                logger.debug(f"[REDIS_CACHE] Set: {key} (TTL: {ttl}s)")

            return result

        except Exception as e:
            logger.error(f"[REDIS_CACHE] Set error: {str(e)}")
            return False

    async def remove(self, key: str) -> bool:
        """Cache'den değer sil - Redis'den"""
        try:
            cache_key = redis_client.generate_cache_key(f"memory_cache", key)
            result = await redis_client.delete_cache_async(self.cache_type, cache_key)

            if result:
                self.stats["evictions"] += 1
                logger.debug(f"[REDIS_CACHE] Removed: {key}")

            return result

        except Exception as e:
            logger.error(f"[REDIS_CACHE] Remove error: {str(e)}")
            return False

    async def clear(self):
        """Cache'i tamamen temizle"""
        try:
            client = redis_client.get_async_client(self.cache_type)
            # Pattern match ile bu adapter'a ait keyler
            pattern = redis_client.generate_cache_key("memory_cache", "*")

            # Scan kullanarak keyler bul ve sil
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                    self.stats["evictions"] += len(keys)
                if cursor == 0:
                    break

            logger.info(f"[REDIS_CACHE] Cache cleared for {self.cache_type}")

        except Exception as e:
            logger.error(f"[REDIS_CACHE] Clear error: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Cache istatistiklerini döndür"""
        hit_rate = 0
        total_requests = self.stats["hits"] + self.stats["misses"]
        if total_requests > 0:
            hit_rate = self.stats["hits"] / total_requests * 100

        return {
            **self.stats,
            "hit_rate_percent": hit_rate,
            "backend": "redis",
            "cache_type": self.cache_type
        }

# ============================
# REDIS QUERY CACHE ADAPTER
# ============================

class RedisQueryCache:
    """
    Redis-backed version of QueryCache
    Compatible API with existing database_pool.QueryCache
    """

    def __init__(self):
        self.cache_type = "query"
        self.pending_queries: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplicated_queries": 0,
            "cached_queries": 0
        }

    async def get_or_execute_query(self, cache_key: str, query_func, *args, **kwargs):
        """Cache'den al veya query'yi çalıştır - Redis ile"""
        async with self._lock:
            redis_key = redis_client.generate_cache_key("query_cache", cache_key)

            # Cache hit kontrolü
            cached_result = await redis_client.get_cache_async(self.cache_type, redis_key)
            if cached_result is not None:
                self.stats["cache_hits"] += 1
                logger.info(f"[REDIS_QUERY] 🚀 CACHE HIT - Query cache'den alındı")
                return cached_result

            # Query deduplication
            if cache_key in self.pending_queries:
                self.stats["deduplicated_queries"] += 1
                logger.info(f"[REDIS_QUERY] 🔄 CACHE DEDUP - Aynı query bekletiliyor")
                future = self.pending_queries[cache_key]
                self._lock.release()
                try:
                    return await future
                finally:
                    await self._lock.acquire()

        # Cache miss - query'yi çalıştır
        self.stats["cache_misses"] += 1
        logger.info(f"[REDIS_QUERY] 📡 CACHE MISS - Arama yapılıyor ve cache'lenecek")

        # Query'yi pending olarak işaretle
        future = asyncio.Future()
        self.pending_queries[cache_key] = future

        try:
            # Query'yi çalıştır
            result = await query_func(*args, **kwargs)

            # Redis'e cache'le
            success = await redis_client.set_cache_async(self.cache_type, redis_key, result)

            if success:
                self.stats["cached_queries"] += 1
                logger.info(f"[REDIS_QUERY] ✅ Query result cached successfully")

            # Pending'den kaldır ve sonucu set et
            async with self._lock:
                if cache_key in self.pending_queries:
                    pending_future = self.pending_queries[cache_key]
                    del self.pending_queries[cache_key]
                    if not pending_future.done():
                        pending_future.set_result(result)

            return result

        except Exception as e:
            # Error durumunda pending'den kaldır
            async with self._lock:
                if cache_key in self.pending_queries:
                    pending_future = self.pending_queries[cache_key]
                    del self.pending_queries[cache_key]
                    if not pending_future.done():
                        pending_future.set_exception(e)
            raise

    async def clear_cache(self):
        """Query cache'i temizle"""
        try:
            client = redis_client.get_async_client(self.cache_type)
            pattern = redis_client.generate_cache_key("query_cache", "*")

            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break

            logger.info("[REDIS_QUERY] Query cache cleared")

        except Exception as e:
            logger.error(f"[REDIS_QUERY] Clear cache error: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Cache istatistikleri"""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = 0
        if total_requests > 0:
            hit_rate = self.stats["cache_hits"] / total_requests * 100

        return {
            **self.stats,
            "hit_rate_percent": hit_rate,
            "pending_queries": len(self.pending_queries),
            "backend": "redis",
            "cache_type": self.cache_type
        }

# ============================
# REDIS BM25 CACHE ADAPTER
# ============================

class RedisBM25Cache:
    """
    Redis-backed BM25 index cache
    Replaces file-based pickle cache with Redis
    """

    def __init__(self):
        self.cache_type = "bm25"

    async def save_bm25_index(self, bm25_index, documents: List, metadata: Dict[str, Any]) -> bool:
        """BM25 index'i Redis'e kaydet"""
        try:
            # BM25 index'i kaydet
            index_key = redis_client.generate_cache_key("bm25_index", "main")
            index_success = await redis_client.set_cache_async(
                self.cache_type, index_key, bm25_index
            )

            # Documents'ı kaydet
            docs_key = redis_client.generate_cache_key("bm25_documents", "main")
            docs_success = await redis_client.set_cache_async(
                self.cache_type, docs_key, documents
            )

            # Metadata'yı kaydet
            meta_key = redis_client.generate_cache_key("bm25_metadata", "main")
            meta_success = await redis_client.set_cache_async(
                self.cache_type, meta_key, metadata
            )

            success = index_success and docs_success and meta_success

            if success:
                logger.info(f"[REDIS_BM25] ✓ BM25 cache saved to Redis: {len(documents)} docs")
            else:
                logger.error("[REDIS_BM25] ✗ Failed to save BM25 cache to Redis")

            return success

        except Exception as e:
            logger.error(f"[REDIS_BM25] Save error: {str(e)}")
            return False

    async def load_bm25_index(self) -> Optional[tuple]:
        """BM25 index'i Redis'den yükle"""
        try:
            # BM25 index'i yükle
            index_key = redis_client.generate_cache_key("bm25_index", "main")
            bm25_index = await redis_client.get_cache_async(self.cache_type, index_key)

            # Documents'ı yükle
            docs_key = redis_client.generate_cache_key("bm25_documents", "main")
            documents = await redis_client.get_cache_async(self.cache_type, docs_key)

            # Metadata'yı yükle
            meta_key = redis_client.generate_cache_key("bm25_metadata", "main")
            metadata = await redis_client.get_cache_async(self.cache_type, meta_key)

            if bm25_index is not None and documents is not None and metadata is not None:
                logger.info(f"[REDIS_BM25] ✓ BM25 cache loaded from Redis: {len(documents)} docs")
                return bm25_index, documents, metadata
            else:
                logger.info("[REDIS_BM25] BM25 cache not found in Redis")
                return None

        except Exception as e:
            logger.error(f"[REDIS_BM25] Load error: {str(e)}")
            return None

    async def is_cache_valid(self, current_signature: str) -> bool:
        """Cache'in geçerli olup olmadığını kontrol et"""
        try:
            meta_key = redis_client.generate_cache_key("bm25_metadata", "main")
            metadata = await redis_client.get_cache_async(self.cache_type, meta_key)

            if metadata is None:
                return False

            cached_signature = metadata.get('content_signature', '')
            is_valid = current_signature == cached_signature

            logger.info(f"[REDIS_BM25] Cache validation: {is_valid} "
                       f"(Current: {current_signature[:8]}..., Cached: {cached_signature[:8]}...)")

            return is_valid

        except Exception as e:
            logger.error(f"[REDIS_BM25] Cache validation error: {str(e)}")
            return False

    async def clear_cache(self):
        """BM25 cache'i temizle"""
        try:
            client = redis_client.get_async_client(self.cache_type)
            pattern = redis_client.generate_cache_key("bm25_*", "*")

            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break

            logger.info("[REDIS_BM25] BM25 cache cleared")

        except Exception as e:
            logger.error(f"[REDIS_BM25] Clear cache error: {str(e)}")

# ============================
# GLOBAL INSTANCES
# ============================

# Redis-backed cache instances
redis_response_cache = RedisMemoryCache("performance")
redis_query_cache = RedisQueryCache()
redis_bm25_cache = RedisBM25Cache()

async def initialize_redis_caches():
    """Redis cache'lerini başlat"""
    try:
        logger.info("[REDIS_CACHE] Initializing Redis cache adapters...")

        # Redis client'ı başlat
        from .redis_client import initialize_redis_async
        success = await initialize_redis_async()

        if success:
            logger.info("[REDIS_CACHE] ✓ Redis cache adapters initialized successfully")
            return True
        else:
            logger.error("[REDIS_CACHE] ✗ Failed to initialize Redis cache adapters")
            return False

    except Exception as e:
        logger.error(f"[REDIS_CACHE] Initialization error: {str(e)}")
        return False