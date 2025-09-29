"""
TutorlyAI Redis Cache Client
Optimized Redis connection and cache management for TutorlyAI RAG system
Using redis-py with asyncio support (no aioredis dependency)
"""

import os
import json
import pickle
import hashlib
import asyncio
import logging
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis
import redis.asyncio as aioredis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6380
    password: Optional[str] = None

    # Database mapping
    query_cache_db: int = 0
    performance_cache_db: int = 1
    bm25_cache_db: int = 2
    session_cache_db: int = 3

    # TTL settings (seconds)
    query_cache_ttl: int = 300  # 5 minutes
    performance_cache_ttl: int = 60  # 1 minute
    bm25_cache_ttl: int = 300  # 5 minutes
    session_cache_ttl: int = 1800  # 30 minutes

    # Connection settings
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30

class TutorlyAIRedisClient:
    """
    TutorlyAI Redis Cache Client
    Uses redis-py with asyncio support for better Python 3.13 compatibility
    """

    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or self._load_config_from_env()

        # Connection pools for different databases
        self.query_cache_pool = None
        self.performance_cache_pool = None
        self.bm25_cache_pool = None
        self.session_cache_pool = None

        # Async connection pools
        self.async_query_cache = None
        self.async_performance_cache = None
        self.async_bm25_cache = None
        self.async_session_cache = None

        self._initialized = False
        self._health_check_task = None

    def _load_config_from_env(self) -> RedisConfig:
        """Environment variables'dan konfigürasyon yükle"""
        return RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6380")),
            password=os.getenv("REDIS_PASSWORD") or None,

            query_cache_db=int(os.getenv("REDIS_QUERY_CACHE_DB", "0")),
            performance_cache_db=int(os.getenv("REDIS_PERFORMANCE_CACHE_DB", "1")),
            bm25_cache_db=int(os.getenv("REDIS_BM25_CACHE_DB", "2")),
            session_cache_db=int(os.getenv("REDIS_SESSION_CACHE_DB", "3")),

            query_cache_ttl=int(os.getenv("REDIS_QUERY_CACHE_TTL", "300")),
            performance_cache_ttl=int(os.getenv("REDIS_PERFORMANCE_CACHE_TTL", "60")),
            bm25_cache_ttl=int(os.getenv("REDIS_BM25_CACHE_TTL", "300")),
        )

    def initialize_sync_pools(self):
        """Synchronous Redis connection pools'ları başlat"""
        try:
            base_config = {
                'host': self.config.host,
                'port': self.config.port,
                'password': self.config.password,
                'socket_timeout': self.config.socket_timeout,
                'socket_connect_timeout': self.config.socket_connect_timeout,
                'retry_on_timeout': self.config.retry_on_timeout,
                'max_connections': self.config.max_connections,
                'decode_responses': False  # Binary data için
            }

            # Query cache pool (DB 0)
            self.query_cache_pool = redis.ConnectionPool(
                db=self.config.query_cache_db,
                **base_config
            )

            # Performance cache pool (DB 1)
            self.performance_cache_pool = redis.ConnectionPool(
                db=self.config.performance_cache_db,
                **base_config
            )

            # BM25 cache pool (DB 2)
            self.bm25_cache_pool = redis.ConnectionPool(
                db=self.config.bm25_cache_db,
                **base_config
            )

            # Session cache pool (DB 3)
            self.session_cache_pool = redis.ConnectionPool(
                db=self.config.session_cache_db,
                **base_config
            )

            logger.info("[REDIS] Sync connection pools initialized")
            return True

        except Exception as e:
            logger.error(f"[REDIS] Failed to initialize sync pools: {str(e)}")
            return False

    async def initialize_async_pools(self):
        """Asynchronous Redis connection pools'ları başlat"""
        try:
            base_config = {
                'host': self.config.host,
                'port': self.config.port,
                'password': self.config.password,
                'socket_timeout': self.config.socket_timeout,
                'socket_connect_timeout': self.config.socket_connect_timeout,
                'retry_on_timeout': self.config.retry_on_timeout,
                'max_connections': self.config.max_connections,
                'decode_responses': False
            }

            # Query cache (DB 0)
            self.async_query_cache = aioredis.Redis(
                db=self.config.query_cache_db,
                **base_config
            )

            # Performance cache (DB 1)
            self.async_performance_cache = aioredis.Redis(
                db=self.config.performance_cache_db,
                **base_config
            )

            # BM25 cache (DB 2)
            self.async_bm25_cache = aioredis.Redis(
                db=self.config.bm25_cache_db,
                **base_config
            )

            # Session cache (DB 3)
            self.async_session_cache = aioredis.Redis(
                db=self.config.session_cache_db,
                **base_config
            )

            logger.info("[REDIS] Async connection pools initialized")
            return True

        except Exception as e:
            logger.error(f"[REDIS] Failed to initialize async pools: {str(e)}")
            return False

    def get_sync_client(self, cache_type: str) -> redis.Redis:
        """Synchronous Redis client al"""
        pool_map = {
            'query': self.query_cache_pool,
            'performance': self.performance_cache_pool,
            'bm25': self.bm25_cache_pool,
            'session': self.session_cache_pool
        }

        pool = pool_map.get(cache_type)
        if not pool:
            raise ValueError(f"Invalid cache type: {cache_type}")

        return redis.Redis(connection_pool=pool)

    def get_async_client(self, cache_type: str) -> aioredis.Redis:
        """Asynchronous Redis client al"""
        client_map = {
            'query': self.async_query_cache,
            'performance': self.async_performance_cache,
            'bm25': self.async_bm25_cache,
            'session': self.async_session_cache
        }

        client = client_map.get(cache_type)
        if not client:
            raise ValueError(f"Invalid cache type: {cache_type}")

        return client

    def get_ttl_for_cache_type(self, cache_type: str) -> int:
        """Cache type için TTL değerini al"""
        ttl_map = {
            'query': self.config.query_cache_ttl,
            'performance': self.config.performance_cache_ttl,
            'bm25': self.config.bm25_cache_ttl,
            'session': self.config.session_cache_ttl
        }

        return ttl_map.get(cache_type, 300)  # Default 5 minutes

    def generate_cache_key(self, prefix: str, data: Union[str, Dict, List]) -> str:
        """Cache key oluştur"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        hash_obj = hashlib.md5(data_str.encode('utf-8'))
        return f"tutorlyai:{prefix}:{hash_obj.hexdigest()}"

    # ========== SYNC CACHE OPERATIONS ==========

    def set_cache(self, cache_type: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Sync cache set"""
        try:
            client = self.get_sync_client(cache_type)
            ttl = ttl or self.get_ttl_for_cache_type(cache_type)

            # Serialize value
            try:
                # Önce JSON deneme (basic types için)
                if isinstance(value, (dict, list, str, int, float, bool)) and not hasattr(value, '__dict__'):
                    serialized_value = json.dumps(value).encode('utf-8')
                    logger.debug(f"[REDIS] Using JSON serialization for {type(value).__name__}")
                else:
                    # Complex objects (Document, etc.) için pickle kullan
                    serialized_value = pickle.dumps(value)
                    logger.debug(f"[REDIS] Using pickle serialization for {type(value).__name__}")
            except (TypeError, ValueError) as e:
                # JSON serialize edilemezse pickle kullan
                serialized_value = pickle.dumps(value)
                logger.debug(f"[REDIS] JSON failed, using pickle for {type(value).__name__}: {str(e)}")

            result = client.setex(key, ttl, serialized_value)
            logger.debug(f"[REDIS] Cache set: {cache_type}:{key} (TTL: {ttl}s)")
            return result

        except Exception as e:
            logger.error(f"[REDIS] Cache set error: {str(e)}")
            return False

    def get_cache(self, cache_type: str, key: str) -> Optional[Any]:
        """Sync cache get"""
        try:
            client = self.get_sync_client(cache_type)
            cached_data = client.get(key)

            if cached_data is None:
                return None

            # Deserialize value
            try:
                # JSON olarak deneme
                return json.loads(cached_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Pickle olarak deneme
                return pickle.loads(cached_data)

        except Exception as e:
            logger.error(f"[REDIS] Cache get error: {str(e)}")
            return None

    def delete_cache(self, cache_type: str, key: str) -> bool:
        """Sync cache delete"""
        try:
            client = self.get_sync_client(cache_type)
            result = client.delete(key)
            logger.debug(f"[REDIS] Cache deleted: {cache_type}:{key}")
            return bool(result)

        except Exception as e:
            logger.error(f"[REDIS] Cache delete error: {str(e)}")
            return False

    # ========== ASYNC CACHE OPERATIONS ==========

    async def set_cache_async(self, cache_type: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Async cache set"""
        try:
            client = self.get_async_client(cache_type)
            ttl = ttl or self.get_ttl_for_cache_type(cache_type)

            # Serialize value
            try:
                # Önce JSON deneme (basic types için)
                if isinstance(value, (dict, list, str, int, float, bool)) and not hasattr(value, '__dict__'):
                    serialized_value = json.dumps(value).encode('utf-8')
                    logger.debug(f"[REDIS] Using JSON serialization for {type(value).__name__}")
                else:
                    # Complex objects (Document, etc.) için pickle kullan
                    serialized_value = pickle.dumps(value)
                    logger.debug(f"[REDIS] Using pickle serialization for {type(value).__name__}")
            except (TypeError, ValueError) as e:
                # JSON serialize edilemezse pickle kullan
                serialized_value = pickle.dumps(value)
                logger.debug(f"[REDIS] JSON failed, using pickle for {type(value).__name__}: {str(e)}")

            result = await client.setex(key, ttl, serialized_value)
            logger.debug(f"[REDIS] Async cache set: {cache_type}:{key} (TTL: {ttl}s)")
            return result

        except Exception as e:
            logger.error(f"[REDIS] Async cache set error: {str(e)}")
            return False

    async def get_cache_async(self, cache_type: str, key: str) -> Optional[Any]:
        """Async cache get"""
        try:
            client = self.get_async_client(cache_type)
            cached_data = await client.get(key)

            if cached_data is None:
                return None

            # Deserialize value
            try:
                # JSON olarak deneme
                return json.loads(cached_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Pickle olarak deneme
                return pickle.loads(cached_data)

        except Exception as e:
            logger.error(f"[REDIS] Async cache get error: {str(e)}")
            return None

    async def delete_cache_async(self, cache_type: str, key: str) -> bool:
        """Async cache delete"""
        try:
            client = self.get_async_client(cache_type)
            result = await client.delete(key)
            logger.debug(f"[REDIS] Async cache deleted: {cache_type}:{key}")
            return bool(result)

        except Exception as e:
            logger.error(f"[REDIS] Async cache delete error: {str(e)}")
            return False

    async def get_ttl_async(self, cache_type: str, key: str) -> Optional[int]:
        """Async TTL get"""
        try:
            client = self.get_async_client(cache_type)
            ttl = await client.ttl(key)

            # Redis TTL returns:
            # -2: key doesn't exist
            # -1: key exists but no expiry set
            # >0: remaining seconds

            if ttl == -2:
                return None  # Key doesn't exist
            elif ttl == -1:
                return 0     # No expiry (permanent)
            else:
                return ttl   # Remaining seconds

        except Exception as e:
            logger.error(f"[REDIS] TTL get error: {str(e)}")
            return None

    # ========== HEALTH CHECK ==========

    async def health_check(self) -> Dict[str, Any]:
        """Redis sağlık kontrolü"""
        health_status = {
            'redis_available': False,
            'databases': {},
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Her database için ping test
            for cache_type in ['query', 'performance', 'bm25', 'session']:
                try:
                    client = self.get_async_client(cache_type)
                    result = await client.ping()

                    # Info al
                    info = await client.info()

                    # Database-specific info
                    db_num = getattr(client, 'db', 0)
                    db_info = info.get(f'db{db_num}', {})

                    health_status['databases'][cache_type] = {
                        'available': result,
                        'db_size': db_info.get('keys', 0) if isinstance(db_info, dict) else 0,
                        'memory_usage': info.get('used_memory_human', 'N/A')
                    }
                except Exception as e:
                    health_status['databases'][cache_type] = {
                        'available': False,
                        'error': str(e)
                    }

            # Genel durum
            health_status['redis_available'] = any(
                db_status.get('available', False)
                for db_status in health_status['databases'].values()
            )

        except Exception as e:
            health_status['error'] = str(e)

        return health_status

    async def start_health_monitoring(self):
        """Health monitoring'i başlat"""
        async def health_monitor():
            while True:
                try:
                    await asyncio.sleep(self.config.health_check_interval)
                    health = await self.health_check()
                    if not health['redis_available']:
                        logger.warning("[REDIS] Health check failed")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[REDIS] Health monitor error: {str(e)}")

        self._health_check_task = asyncio.create_task(health_monitor())
        logger.info("[REDIS] Health monitoring started")

    async def stop_health_monitoring(self):
        """Health monitoring'i durdur"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("[REDIS] Health monitoring stopped")

    async def cleanup(self):
        """Cleanup resources"""
        await self.stop_health_monitoring()

        # Close async connections
        for client in [self.async_query_cache, self.async_performance_cache,
                      self.async_bm25_cache, self.async_session_cache]:
            if client:
                try:
                    await client.aclose()
                except:
                    pass

# ========== GLOBAL INSTANCE ==========

redis_client = TutorlyAIRedisClient()

def initialize_redis():
    """Redis client'ı başlat"""
    try:
        # Sync pools
        success = redis_client.initialize_sync_pools()
        if success:
            logger.info("[REDIS] TutorlyAI Redis client initialized successfully")
            return True
        else:
            logger.error("[REDIS] Failed to initialize Redis client")
            return False
    except Exception as e:
        logger.error(f"[REDIS] Initialization error: {str(e)}")
        return False

async def initialize_redis_async():
    """Redis async client'ı başlat"""
    try:
        success = await redis_client.initialize_async_pools()
        if success:
            await redis_client.start_health_monitoring()
            logger.info("[REDIS] TutorlyAI Redis async client initialized successfully")
            return True
        else:
            logger.error("[REDIS] Failed to initialize Redis async client")
            return False
    except Exception as e:
        logger.error(f"[REDIS] Async initialization error: {str(e)}")
        return False