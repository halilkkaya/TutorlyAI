"""
Resilience ve Hata Yönetimi Utilities
Circuit breaker, retry, timeout, rate limiting ve fallback stratejileri
"""

import asyncio
import time
import logging
from typing import Any, Dict, Optional, Union, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import fal_client

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal çalışma
    OPEN = "open"          # Hata durumu, istekler reddediliyor
    HALF_OPEN = "half_open"  # Test modu

class ErrorCategory(Enum):
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    UNKNOWN = "unknown"

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3

@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 1000

@dataclass
class FalClientConfig:
    timeout_seconds: float = 30.0
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    rate_limit_config: RateLimitConfig = field(default_factory=RateLimitConfig)
    enable_fallback: bool = True

# Endpoint'lere özel timeout konfigürasyonları
ENDPOINT_TIMEOUT_CONFIG = {
    "search_plan": 20.0,           # RAG arama planı - hızlı olmalı
    "text_generation": 30.0,      # Normal text üretimi
    "quiz_generation": 45.0,      # Quiz üretimi - biraz daha uzun
    "english_learning": 35.0,     # İngilizce öğrenme
    "image_generation": 90.0,     # Görsel üretimi - en uzun
    "stream": 120.0,              # Stream işlemleri - çok uzun
    "default": 30.0               # Varsayılan
}

class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests_minute = []
        self.requests_hour = []

    def can_proceed(self) -> bool:
        now = time.time()

        # Son 1 dakikayı temizle
        self.requests_minute = [req_time for req_time in self.requests_minute
                              if now - req_time < 60]

        # Son 1 saati temizle
        self.requests_hour = [req_time for req_time in self.requests_hour
                            if now - req_time < 3600]

        # Limitleri kontrol et
        if len(self.requests_minute) >= self.config.requests_per_minute:
            logger.warning(f"[RATE_LIMIT] Dakika limiti aşıldı: {len(self.requests_minute)}/{self.config.requests_per_minute}")
            return False

        if len(self.requests_hour) >= self.config.requests_per_hour:
            logger.warning(f"[RATE_LIMIT] Saat limiti aşıldı: {len(self.requests_hour)}/{self.config.requests_per_hour}")
            return False

        return True

    def record_request(self):
        now = time.time()
        self.requests_minute.append(now)
        self.requests_hour.append(now)

class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0

    def can_proceed(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.config.timeout_seconds:
                logger.info("[CIRCUIT_BREAKER] OPEN -> HALF_OPEN geçişi")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                logger.info("[CIRCUIT_BREAKER] HALF_OPEN -> CLOSED geçişi")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                logger.warning(f"[CIRCUIT_BREAKER] CLOSED -> OPEN geçişi (hata sayısı: {self.failure_count})")
                self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning("[CIRCUIT_BREAKER] HALF_OPEN -> OPEN geçişi")
            self.state = CircuitState.OPEN
            self.success_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1

def categorize_error(error: Exception) -> ErrorCategory:
    """Hata tipini kategorize eder"""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    if "timeout" in error_str or "timeout" in error_type:
        return ErrorCategory.TIMEOUT
    elif "network" in error_str or "connection" in error_str:
        return ErrorCategory.NETWORK
    elif "rate" in error_str and "limit" in error_str:
        return ErrorCategory.RATE_LIMIT
    elif "auth" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
        return ErrorCategory.AUTHENTICATION
    elif "500" in error_str or "502" in error_str or "503" in error_str:
        return ErrorCategory.SERVER_ERROR
    elif "400" in error_str or "404" in error_str:
        return ErrorCategory.CLIENT_ERROR
    else:
        return ErrorCategory.UNKNOWN

class ResilientFalClient:
    def __init__(self, config: FalClientConfig = None):
        self.config = config or FalClientConfig()
        self.rate_limiter = RateLimiter(self.config.rate_limit_config)
        self.circuit_breaker = CircuitBreaker(self.config.circuit_breaker_config)

    async def run_async_with_resilience(
        self,
        endpoint: str,
        arguments: Dict[str, Any],
        fallback_response: Optional[Dict[str, Any]] = None,
        timeout_override: Optional[float] = None,
        operation_type: str = "default"
    ) -> Dict[str, Any]:
        """Resilient fal_client.run_async çağrısı"""

        # Dinamik timeout belirleme
        if timeout_override:
            timeout_seconds = timeout_override
        else:
            timeout_seconds = ENDPOINT_TIMEOUT_CONFIG.get(operation_type, self.config.timeout_seconds)

        logger.info(f"[RESILIENCE] Operation: {operation_type}, Timeout: {timeout_seconds}s")

        # Circuit breaker kontrolü
        if not self.circuit_breaker.can_proceed():
            logger.error("[RESILIENCE] Circuit breaker OPEN - istek engellendi")
            if self.config.enable_fallback and fallback_response:
                logger.info("[RESILIENCE] Fallback response döndürülüyor")
                return fallback_response
            raise Exception("Circuit breaker is OPEN - requests are blocked")

        # Rate limiting kontrolü
        if not self.rate_limiter.can_proceed():
            logger.error("[RESILIENCE] Rate limit aşıldı - istek engellendi")
            if self.config.enable_fallback and fallback_response:
                logger.info("[RESILIENCE] Fallback response döndürülüyor")
                return fallback_response
            raise Exception("Rate limit exceeded")

        # Rate limiting kaydı
        self.rate_limiter.record_request()

        # Retry mekanizması ile çağrı yap
        last_error = None
        for attempt in range(self.config.retry_config.max_attempts):
            try:
                logger.info(f"[RESILIENCE] fal_client.run_async çağrısı (deneme {attempt + 1}/{self.config.retry_config.max_attempts})")

                # Timeout ile çağrı (dinamik timeout)
                result = await asyncio.wait_for(
                    fal_client.run_async(endpoint, arguments=arguments),
                    timeout=timeout_seconds
                )

                # Başarılı sonuç
                self.circuit_breaker.record_success()
                logger.info(f"[RESILIENCE] Başarılı sonuç alındı (deneme {attempt + 1})")
                return result

            except asyncio.TimeoutError as e:
                last_error = e
                error_category = ErrorCategory.TIMEOUT
                logger.warning(f"[RESILIENCE] Timeout hatası (deneme {attempt + 1}): {str(e)}")

            except Exception as e:
                last_error = e
                error_category = categorize_error(e)
                logger.warning(f"[RESILIENCE] Hata (deneme {attempt + 1}, kategori: {error_category.value}): {str(e)}")

            # Bazı hata tiplerinde retry yapmayalım
            if error_category in [ErrorCategory.AUTHENTICATION, ErrorCategory.CLIENT_ERROR]:
                logger.error(f"[RESILIENCE] Retry yapılmayacak hata tipi: {error_category.value}")
                break

            # Son deneme değilse bekle
            if attempt < self.config.retry_config.max_attempts - 1:
                delay = self._calculate_delay(attempt)
                logger.info(f"[RESILIENCE] {delay:.2f} saniye bekleniyor...")
                await asyncio.sleep(delay)

        # Tüm denemeler başarısız
        self.circuit_breaker.record_failure()
        logger.error(f"[RESILIENCE] Tüm denemeler başarısız: {str(last_error)}")

        # Fallback varsa kullan
        if self.config.enable_fallback and fallback_response:
            logger.info("[RESILIENCE] Fallback response döndürülüyor")
            return fallback_response

        # Son hatayı fırlat
        raise last_error

    def stream_async_with_resilience(
        self,
        endpoint: str,
        arguments: Dict[str, Any],
        operation_type: str = "stream"
    ):
        """Resilient fal_client.stream_async çağrısı"""

        # Stream için timeout bilgisi
        timeout_seconds = ENDPOINT_TIMEOUT_CONFIG.get(operation_type, self.config.timeout_seconds)
        logger.info(f"[RESILIENCE] Stream Operation: {operation_type}, Timeout: {timeout_seconds}s")

        # Circuit breaker kontrolü
        if not self.circuit_breaker.can_proceed():
            logger.error("[RESILIENCE] Circuit breaker OPEN - stream engellendi")
            raise Exception("Circuit breaker is OPEN - streams are blocked")

        # Rate limiting kontrolü
        if not self.rate_limiter.can_proceed():
            logger.error("[RESILIENCE] Rate limit aşıldı - stream engellendi")
            raise Exception("Rate limit exceeded")

        # Rate limiting kaydı
        self.rate_limiter.record_request()

        try:
            logger.info("[RESILIENCE] fal_client.stream_async çağrısı başlatılıyor")
            stream = fal_client.stream_async(endpoint, arguments=arguments)
            self.circuit_breaker.record_success()
            return stream

        except Exception as e:
            self.circuit_breaker.record_failure()
            error_category = categorize_error(e)
            logger.error(f"[RESILIENCE] Stream hatası (kategori: {error_category.value}): {str(e)}")
            raise

    def _calculate_delay(self, attempt: int) -> float:
        """Exponential backoff ile bekleme süresini hesaplar"""
        delay = self.config.retry_config.base_delay * \
                (self.config.retry_config.exponential_base ** attempt)

        delay = min(delay, self.config.retry_config.max_delay)

        if self.config.retry_config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri döndürür"""
        return {
            "circuit_breaker": {
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
                "success_count": self.circuit_breaker.success_count,
                "last_failure_time": self.circuit_breaker.last_failure_time
            },
            "rate_limiter": {
                "requests_last_minute": len(self.rate_limiter.requests_minute),
                "requests_last_hour": len(self.rate_limiter.requests_hour),
                "minute_limit": self.config.rate_limit_config.requests_per_minute,
                "hour_limit": self.config.rate_limit_config.requests_per_hour
            },
            "config": {
                "default_timeout_seconds": self.config.timeout_seconds,
                "max_retry_attempts": self.config.retry_config.max_attempts,
                "circuit_breaker_threshold": self.config.circuit_breaker_config.failure_threshold
            },
            "endpoint_timeouts": ENDPOINT_TIMEOUT_CONFIG
        }

# Global instance
resilient_client = ResilientFalClient()

# Fallback response templates
FALLBACK_RESPONSES = {
    "text_generation": {
        "output": "Üzgünüm, şu anda AI servisi kullanılamıyor. Lütfen daha sonra tekrar deneyin.",
        "generated_text": "Servis geçici olarak kullanılamıyor. Lütfen daha sonra tekrar deneyin.",
        "is_fallback": True
    },
    "quiz_generation": {
        "success": False,
        "message": "Quiz servisi geçici olarak kullanılamıyor",
        "is_fallback": True,
        "fallback_quiz": {
            "sorular": [
                {
                    "soru": "Genel Bilgi Sorusu (Fallback)",
                    "secenekler": ["A) Geçici hata", "B) Servis kullanılamıyor", "C) Lütfen tekrar deneyin", "D) Tümü"],
                    "dogru_cevap": "D",
                    "aciklama": "AI servisi geçici olarak kullanılamıyor. Lütfen daha sonra tekrar deneyin."
                }
            ]
        }
    },
    "image_generation": {
        "success": False,
        "image_url": None,
        "error_message": "Görsel üretim servisi geçici olarak kullanılamıyor",
        "is_fallback": True
    }
}

async def create_fallback_response(response_type: str, **kwargs) -> Dict[str, Any]:
    """Belirli tip için fallback response oluşturur"""
    base_response = FALLBACK_RESPONSES.get(response_type, {
        "output": "Servis geçici olarak kullanılamıyor",
        "is_fallback": True
    })

    # Ek parametreleri ekle
    base_response.update(kwargs)
    base_response["generated_at"] = datetime.now().isoformat()

    return base_response