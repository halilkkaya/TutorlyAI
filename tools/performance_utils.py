"""
Performance and Scalability Utils
Concurrent request limiting, connection pooling, caching, memory monitoring
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from fastapi import HTTPException, Request
import weakref
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# =========================
# CONCURRENT REQUEST LIMITING
# =========================

@dataclass
class ConcurrencyConfig:
    max_concurrent_requests: int = 50
    max_requests_per_user: int = 10
    request_timeout_seconds: int = 300  # 5 minutes
    queue_timeout_seconds: int = 30

class ConcurrentRequestLimiter:
    def __init__(self, config: ConcurrencyConfig = None):
        self.config = config or ConcurrencyConfig()
        self.global_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self.user_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.active_requests: Dict[str, Set[str]] = defaultdict(set)
        self.request_start_times: Dict[str, datetime] = {}
        self.stats = {
            "total_requests": 0,
            "concurrent_requests": 0,
            "rejected_requests": 0,
            "timeout_requests": 0,
            "max_concurrent_reached": 0
        }
        self._lock = asyncio.Lock()

    async def get_user_semaphore(self, user_id: str) -> asyncio.Semaphore:
        """User başına semaphore al/oluştur"""
        if user_id not in self.user_semaphores:
            self.user_semaphores[user_id] = asyncio.Semaphore(self.config.max_requests_per_user)
        return self.user_semaphores[user_id]

    @asynccontextmanager
    async def limit_request(self, request_id: str, user_id: str = "anonymous"):
        """Request limiting context manager"""
        async with self._lock:
            self.stats["total_requests"] += 1
            current_concurrent = self.stats["concurrent_requests"]
            if current_concurrent > self.stats["max_concurrent_reached"]:
                self.stats["max_concurrent_reached"] = current_concurrent

        # Global concurrency check
        if self.global_semaphore.locked():
            self.stats["rejected_requests"] += 1
            logger.warning(f"[PERFORMANCE] Global request limit reached, rejecting request {request_id}")
            raise HTTPException(status_code=429, detail="Server is busy, please try again later")

        # User-specific concurrency check
        user_semaphore = await self.get_user_semaphore(user_id)
        if user_semaphore.locked():
            self.stats["rejected_requests"] += 1
            logger.warning(f"[PERFORMANCE] User {user_id} request limit reached, rejecting request {request_id}")
            raise HTTPException(status_code=429, detail="Too many requests from user, please try again later")

        # Acquire both semaphores
        try:
            async with self.global_semaphore:
                async with user_semaphore:
                    async with self._lock:
                        self.stats["concurrent_requests"] += 1
                        self.active_requests[user_id].add(request_id)
                        self.request_start_times[request_id] = datetime.now()

                    logger.info(f"[PERFORMANCE] Request {request_id} started for user {user_id}")
                    try:
                        yield
                    finally:
                        # Cleanup
                        async with self._lock:
                            self.stats["concurrent_requests"] -= 1
                            self.active_requests[user_id].discard(request_id)
                            if request_id in self.request_start_times:
                                duration = datetime.now() - self.request_start_times[request_id]
                                if duration.total_seconds() > self.config.request_timeout_seconds:
                                    self.stats["timeout_requests"] += 1
                                del self.request_start_times[request_id]

                        logger.info(f"[PERFORMANCE] Request {request_id} completed for user {user_id}")

        except asyncio.TimeoutError:
            self.stats["timeout_requests"] += 1
            logger.error(f"[PERFORMANCE] Request {request_id} timed out for user {user_id}")
            raise HTTPException(status_code=408, detail="Request timeout")

    def get_stats(self) -> Dict[str, Any]:
        """Performance istatistiklerini döndür"""
        return {
            "concurrent_requests": self.stats["concurrent_requests"],
            "total_requests": self.stats["total_requests"],
            "rejected_requests": self.stats["rejected_requests"],
            "timeout_requests": self.stats["timeout_requests"],
            "max_concurrent_reached": self.stats["max_concurrent_reached"],
            "active_users": len([u for u, reqs in self.active_requests.items() if reqs]),
            "config": {
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "max_requests_per_user": self.config.max_requests_per_user,
                "request_timeout_seconds": self.config.request_timeout_seconds
            }
        }

# =========================
# MEMORY MONITORING
# =========================

@dataclass
class MemoryConfig:
    max_memory_percent: float = 80.0  # % cinsinden
    max_memory_mb: int = 2048  # MB cinsinden
    monitoring_interval: int = 60  # saniye
    cleanup_threshold: float = 85.0  # % cinsinden cleanup tetikleme

class MemoryMonitor:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.stats = {
            "current_memory_mb": 0,
            "current_memory_percent": 0,
            "peak_memory_mb": 0,
            "cleanup_count": 0,
            "warning_count": 0
        }
        self.monitoring_active = False
        self._monitoring_task = None

    def start_monitoring(self):
        """Memory monitoring'i başlat"""
        if not self.monitoring_active:
            self.monitoring_active = True
            logger.info("[PERFORMANCE] Memory monitoring started")

    def stop_monitoring(self):
        """Memory monitoring'i durdur"""
        if self.monitoring_active:
            self.monitoring_active = False
            logger.info("[PERFORMANCE] Memory monitoring stopped")

    def get_memory_info(self) -> Dict[str, Any]:
        """Mevcut memory bilgilerini al"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        current_mb = memory_info.rss / 1024 / 1024

        # Stats güncelle
        self.stats["current_memory_mb"] = current_mb
        self.stats["current_memory_percent"] = memory_percent

        if current_mb > self.stats["peak_memory_mb"]:
            self.stats["peak_memory_mb"] = current_mb

        return {
            "memory_mb": current_mb,
            "memory_percent": memory_percent,
            "memory_rss": memory_info.rss,
            "memory_vms": memory_info.vms,
            "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024
        }

    def check_memory_limits(self) -> bool:
        """Memory limitlerini kontrol et"""
        memory_info = self.get_memory_info()
        current_mb = memory_info["memory_mb"]
        current_percent = memory_info["memory_percent"]

        # Limit kontrolü
        if (current_percent > self.config.max_memory_percent or
            current_mb > self.config.max_memory_mb):

            self.stats["warning_count"] += 1
            logger.warning(f"[PERFORMANCE] Memory limit exceeded: {current_mb:.1f}MB ({current_percent:.1f}%)")

            # Cleanup threshold kontrolü
            if current_percent > self.config.cleanup_threshold:
                self.stats["cleanup_count"] += 1
                logger.error(f"[PERFORMANCE] Memory cleanup required: {current_mb:.1f}MB ({current_percent:.1f}%)")
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Memory istatistiklerini döndür"""
        current_info = self.get_memory_info()
        return {
            **self.stats,
            **current_info,
            "config": {
                "max_memory_percent": self.config.max_memory_percent,
                "max_memory_mb": self.config.max_memory_mb,
                "cleanup_threshold": self.config.cleanup_threshold
            }
        }

# =========================
# CACHE MANAGEMENT
# =========================

@dataclass
class CacheConfig:
    max_cache_size: int = 1000  # Maksimum cache item sayısı
    default_ttl_seconds: int = 300  # 5 dakika
    cleanup_interval: int = 60  # 1 dakika
    max_memory_mb: int = 100  # Cache için maksimum memory

@dataclass
class CacheItem:
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    ttl_seconds: int
    access_count: int = 0
    size_bytes: int = 0

class MemoryCache:
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.cache: Dict[str, CacheItem] = {}
        self.access_order = deque()  # LRU için
        self._lock = asyncio.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "current_size": 0,
            "memory_usage_mb": 0
        }

    async def get(self, key: str) -> Optional[Any]:
        """Cache'den değer al"""
        async with self._lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            item = self.cache[key]

            # TTL kontrolü
            if self._is_expired(item):
                await self._remove_item(key)
                self.stats["misses"] += 1
                return None

            # Access güncelle
            item.last_accessed = datetime.now()
            item.access_count += 1
            self._update_access_order(key)

            self.stats["hits"] += 1
            return item.value

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Cache'e değer ekle"""
        async with self._lock:
            ttl = ttl_seconds or self.config.default_ttl_seconds
            size_bytes = self._estimate_size(value)

            # Memory limit kontrolü
            if (self.stats["memory_usage_mb"] + size_bytes / 1024 / 1024) > self.config.max_memory_mb:
                await self._cleanup_cache()

            # Size limit kontrolü
            if len(self.cache) >= self.config.max_cache_size:
                await self._evict_lru()

            now = datetime.now()
            item = CacheItem(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl_seconds=ttl,
                size_bytes=size_bytes
            )

            # Mevcut item varsa güncelle
            if key in self.cache:
                old_item = self.cache[key]
                self.stats["memory_usage_mb"] -= old_item.size_bytes / 1024 / 1024

            self.cache[key] = item
            self._update_access_order(key)

            self.stats["current_size"] = len(self.cache)
            self.stats["memory_usage_mb"] += size_bytes / 1024 / 1024

            return True

    async def remove(self, key: str) -> bool:
        """Cache'den değer sil"""
        async with self._lock:
            return await self._remove_item(key)

    async def clear(self):
        """Cache'i temizle"""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats["current_size"] = 0
            self.stats["memory_usage_mb"] = 0

    def _is_expired(self, item: CacheItem) -> bool:
        """Item'ın süresi dolmuş mu?"""
        return (datetime.now() - item.created_at).total_seconds() > item.ttl_seconds

    async def _remove_item(self, key: str) -> bool:
        """Item'ı cache'den kaldır"""
        if key in self.cache:
            item = self.cache[key]
            del self.cache[key]

            # Access order'dan kaldır
            try:
                self.access_order.remove(key)
            except ValueError:
                pass

            self.stats["current_size"] = len(self.cache)
            self.stats["memory_usage_mb"] -= item.size_bytes / 1024 / 1024
            return True
        return False

    async def _evict_lru(self):
        """LRU eviction"""
        if self.access_order:
            lru_key = self.access_order.popleft()
            await self._remove_item(lru_key)
            self.stats["evictions"] += 1

    async def _cleanup_cache(self):
        """Expired items'ı temizle"""
        expired_keys = []
        for key, item in self.cache.items():
            if self._is_expired(item):
                expired_keys.append(key)

        for key in expired_keys:
            await self._remove_item(key)

    def _update_access_order(self, key: str):
        """LRU access order'ı güncelle"""
        try:
            self.access_order.remove(key)
        except ValueError:
            pass
        self.access_order.append(key)

    def _estimate_size(self, value: Any) -> int:
        """Değerin yaklaşık boyutunu hesapla"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
            elif isinstance(value, list):
                return sum(self._estimate_size(item) for item in value)
            else:
                return len(str(value))
        except:
            return 100  # Default estimate

    def get_stats(self) -> Dict[str, Any]:
        """Cache istatistiklerini döndür"""
        hit_rate = 0
        total_requests = self.stats["hits"] + self.stats["misses"]
        if total_requests > 0:
            hit_rate = self.stats["hits"] / total_requests * 100

        return {
            **self.stats,
            "hit_rate_percent": hit_rate,
            "config": {
                "max_cache_size": self.config.max_cache_size,
                "default_ttl_seconds": self.config.default_ttl_seconds,
                "max_memory_mb": self.config.max_memory_mb
            }
        }

# =========================
# GLOBAL INSTANCES
# =========================

# Performance managers
concurrency_config = ConcurrencyConfig()
memory_config = MemoryConfig()
cache_config = CacheConfig()

request_limiter = ConcurrentRequestLimiter(concurrency_config)
memory_monitor = MemoryMonitor(memory_config)
response_cache = MemoryCache(cache_config)

def get_performance_stats() -> Dict[str, Any]:
    """Tüm performance istatistiklerini döndür"""
    return {
        "concurrency": request_limiter.get_stats(),
        "memory": memory_monitor.get_stats(),
        "cache": response_cache.get_stats(),
        "timestamp": datetime.now().isoformat()
    }