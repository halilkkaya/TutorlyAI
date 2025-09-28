"""
Database Connection Pooling and Optimization
ChromaDB connection pooling, query caching, optimization strategies
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager
import weakref
import threading
import hashlib
import json

logger = logging.getLogger(__name__)

# =========================
# CHROMADB CONNECTION POOLING
# =========================

@dataclass
class DatabaseConfig:
    max_connections: int = 10
    min_connections: int = 2
    connection_timeout: int = 30
    query_timeout: int = 60
    retry_attempts: int = 3
    pool_recycle_time: int = 3600  # 1 hour

class ChromaDBConnectionPool:
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.connections: List[Any] = []
        self.busy_connections: List[Any] = []
        self.connection_created_times: Dict[Any, datetime] = {}
        self.stats = {
            "total_connections_created": 0,
            "active_connections": 0,
            "busy_connections": 0,
            "total_queries": 0,
            "failed_queries": 0,
            "recycled_connections": 0
        }
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self, vectorstore_factory):
        """Connection pool'u başlat"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                # Minimum connection sayısını oluştur
                for _ in range(self.config.min_connections):
                    connection = await self._create_connection(vectorstore_factory)
                    if connection:
                        self.connections.append(connection)

                self._initialized = True
                logger.info(f"[DATABASE] Connection pool initialized with {len(self.connections)} connections")

            except Exception as e:
                logger.error(f"[DATABASE] Failed to initialize connection pool: {str(e)}")
                raise

    async def _create_connection(self, vectorstore_factory) -> Optional[Any]:
        """Yeni connection oluştur"""
        try:
            connection = vectorstore_factory()
            self.connection_created_times[connection] = datetime.now()
            self.stats["total_connections_created"] += 1
            self.stats["active_connections"] += 1
            logger.debug("[DATABASE] New connection created")
            return connection
        except Exception as e:
            logger.error(f"[DATABASE] Failed to create connection: {str(e)}")
            return None

    @asynccontextmanager
    async def get_connection(self, vectorstore_factory):
        """Connection pool'dan connection al"""
        connection = None
        try:
            async with self._lock:
                # Mevcut bağlantılardan temiz bir tane al
                available_connections = [
                    conn for conn in self.connections
                    if conn not in self.busy_connections and self._is_connection_healthy(conn)
                ]

                if available_connections:
                    connection = available_connections[0]
                    self.busy_connections.append(connection)
                    self.stats["busy_connections"] += 1
                elif len(self.connections) < self.config.max_connections:
                    # Yeni connection oluştur
                    connection = await self._create_connection(vectorstore_factory)
                    if connection:
                        self.connections.append(connection)
                        self.busy_connections.append(connection)
                        self.stats["busy_connections"] += 1
                else:
                    # Pool dolu, bekle
                    logger.warning("[DATABASE] Connection pool is full, waiting...")
                    raise Exception("Connection pool exhausted")

            if connection:
                yield connection
            else:
                raise Exception("Could not acquire database connection")

        except Exception as e:
            logger.error(f"[DATABASE] Error with connection: {str(e)}")
            raise
        finally:
            # Connection'ı geri ver
            if connection:
                async with self._lock:
                    try:
                        self.busy_connections.remove(connection)
                        self.stats["busy_connections"] -= 1
                    except ValueError:
                        pass

    def _is_connection_healthy(self, connection) -> bool:
        """Connection'ın sağlıklı olup olmadığını kontrol et"""
        try:
            # Connection yaş kontrolü
            created_time = self.connection_created_times.get(connection)
            if created_time:
                age = (datetime.now() - created_time).total_seconds()
                if age > self.config.pool_recycle_time:
                    return False

            # ChromaDB için basit health check
            if hasattr(connection, '_collection') and connection._collection:
                # Collection hala erişilebilir mi?
                return True

            return True
        except Exception:
            return False

    async def cleanup_old_connections(self):
        """Eski connection'ları temizle"""
        async with self._lock:
            connections_to_remove = []
            current_time = datetime.now()

            for connection in self.connections:
                if connection in self.busy_connections:
                    continue

                created_time = self.connection_created_times.get(connection)
                if created_time:
                    age = (current_time - created_time).total_seconds()
                    if age > self.config.pool_recycle_time:
                        connections_to_remove.append(connection)

            # Eski connection'ları kaldır
            for connection in connections_to_remove:
                try:
                    self.connections.remove(connection)
                    if connection in self.connection_created_times:
                        del self.connection_created_times[connection]
                    self.stats["active_connections"] -= 1
                    self.stats["recycled_connections"] += 1
                    logger.debug("[DATABASE] Old connection recycled")
                except Exception as e:
                    logger.warning(f"[DATABASE] Error removing old connection: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Connection pool istatistikleri"""
        return {
            **self.stats,
            "available_connections": len(self.connections) - len(self.busy_connections),
            "total_connections": len(self.connections),
            "pool_utilization_percent": len(self.busy_connections) / max(len(self.connections), 1) * 100,
            "config": {
                "max_connections": self.config.max_connections,
                "min_connections": self.config.min_connections,
                "connection_timeout": self.config.connection_timeout
            }
        }

# =========================
# QUERY CACHING
# =========================

@dataclass
class QueryCacheConfig:
    max_cache_size: int = 500
    default_ttl_seconds: int = 300  # 5 dakika
    enable_query_deduplication: bool = True
    cache_hit_threshold: int = 2  # Aynı query 2 kez çağrılırsa cache'le

class QueryCache:
    def __init__(self, config: QueryCacheConfig = None):
        self.config = config or QueryCacheConfig()
        self.query_cache: Dict[str, Any] = {}
        self.query_timestamps: Dict[str, datetime] = {}
        self.query_hit_counts: Dict[str, int] = {}
        self.pending_queries: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplicated_queries": 0,
            "cached_queries": 0
        }

    def _generate_cache_key(self, query: str, filters: Dict, k: int, **kwargs) -> str:
        """Query için cache key oluştur"""
        cache_data = {
            "query": query,
            "filters": filters,
            "k": k,
            **kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    async def get_or_execute_query(self, cache_key: str, query_func, *args, **kwargs):
        """Cache'den al veya query'yi çalıştır"""
        async with self._lock:
            # Cache hit kontrolü
            if cache_key in self.query_cache:
                # TTL kontrolü
                if cache_key in self.query_timestamps:
                    age = (datetime.now() - self.query_timestamps[cache_key]).total_seconds()
                    if age <= self.config.default_ttl_seconds:
                        self.stats["cache_hits"] += 1
                        self.query_hit_counts[cache_key] = self.query_hit_counts.get(cache_key, 0) + 1
                        logger.info(f"[SEARCH] 🚀 CACHE HIT - Query cache'den alındı (age: {age:.1f}s)")
                        return self.query_cache[cache_key]
                    else:
                        # Expired, remove from cache
                        del self.query_cache[cache_key]
                        del self.query_timestamps[cache_key]
                        logger.info(f"[SEARCH] ⏰ CACHE EXPIRED - Cache'den silindi (age: {age:.1f}s)")

            # Query deduplication - aynı query eş zamanlı çalışıyorsa bekle
            if self.config.enable_query_deduplication and cache_key in self.pending_queries:
                self.stats["deduplicated_queries"] += 1
                logger.info(f"[SEARCH] 🔄 CACHE DEDUP - Aynı query bekletiliyor: {cache_key[:8]}...")
                future = self.pending_queries[cache_key]
                # Lock'u serbest bırak ve sonucu bekle
                self._lock.release()
                try:
                    return await future
                finally:
                    await self._lock.acquire()

        # Cache miss - query'yi çalıştır
        self.stats["cache_misses"] += 1
        logger.info(f"[SEARCH] 📡 CACHE MISS - Arama yapılıyor ve cache'lenecek")

        # Query'yi pending olarak işaretle
        if self.config.enable_query_deduplication:
            future = asyncio.Future()
            self.pending_queries[cache_key] = future

        try:
            # Query'yi çalıştır
            result = await query_func(*args, **kwargs)

            # Cache'e ekle
            async with self._lock:
                # Cache boyut kontrolü
                if len(self.query_cache) >= self.config.max_cache_size:
                    await self._evict_oldest_cache_entry()

                self.query_cache[cache_key] = result
                self.query_timestamps[cache_key] = datetime.now()
                self.query_hit_counts[cache_key] = 1
                self.stats["cached_queries"] += 1

                # Pending'den kaldır ve sonucu set et
                if cache_key in self.pending_queries:
                    future = self.pending_queries[cache_key]
                    del self.pending_queries[cache_key]
                    if not future.done():
                        future.set_result(result)

            return result

        except Exception as e:
            # Error durumunda pending'den kaldır
            async with self._lock:
                if cache_key in self.pending_queries:
                    future = self.pending_queries[cache_key]
                    del self.pending_queries[cache_key]
                    if not future.done():
                        future.set_exception(e)
            raise

    async def _evict_oldest_cache_entry(self):
        """En eski cache entry'yi çıkar"""
        if self.query_timestamps:
            oldest_key = min(self.query_timestamps.keys(),
                           key=lambda k: self.query_timestamps[k])

            if oldest_key in self.query_cache:
                del self.query_cache[oldest_key]
            if oldest_key in self.query_timestamps:
                del self.query_timestamps[oldest_key]
            if oldest_key in self.query_hit_counts:
                del self.query_hit_counts[oldest_key]

    async def clear_cache(self):
        """Cache'i temizle"""
        async with self._lock:
            self.query_cache.clear()
            self.query_timestamps.clear()
            self.query_hit_counts.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Cache istatistikleri"""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = 0
        if total_requests > 0:
            hit_rate = self.stats["cache_hits"] / total_requests * 100

        return {
            **self.stats,
            "hit_rate_percent": hit_rate,
            "current_cache_size": len(self.query_cache),
            "pending_queries": len(self.pending_queries),
            "config": {
                "max_cache_size": self.config.max_cache_size,
                "default_ttl_seconds": self.config.default_ttl_seconds,
                "enable_query_deduplication": self.config.enable_query_deduplication
            }
        }

# =========================
# GLOBAL INSTANCES
# =========================

database_config = DatabaseConfig()
query_cache_config = QueryCacheConfig()

db_pool = ChromaDBConnectionPool(database_config)
query_cache = QueryCache(query_cache_config)

def get_database_stats() -> Dict[str, Any]:
    """Database performance istatistikleri"""
    return {
        "connection_pool": db_pool.get_stats(),
        "query_cache": query_cache.get_stats(),
        "timestamp": datetime.now().isoformat()
    }