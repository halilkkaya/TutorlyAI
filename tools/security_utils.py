"""
Güvenlik Utilities
Input sanitization, validation, file security, API key management
"""

import re
import os
import hashlib
import hmac
import secrets
import time
import logging
from typing import Any, Dict, List, Optional, Union, Set
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from fastapi import HTTPException, Request
import base64
import json

logger = logging.getLogger(__name__)

# Güvenli file extensions
ALLOWED_FILE_EXTENSIONS = {
    '.pdf', '.txt', '.md', '.json'
}

# Maksimum file boyutları (bytes)
MAX_FILE_SIZES = {
    '.pdf': 50 * 1024 * 1024,  # 50MB
    '.txt': 5 * 1024 * 1024,   # 5MB
    '.md': 2 * 1024 * 1024,    # 2MB
    '.json': 1 * 1024 * 1024,  # 1MB
}

# Yasaklı path patterns
FORBIDDEN_PATH_PATTERNS = [
    r'\.\./',        # Directory traversal
    r'\.\.\\',       # Directory traversal (Windows)
    r'/etc/',        # System files
    r'/root/',       # Root directory
    r'/proc/',       # Process info
    r'/sys/',        # System info
    r'C:/',          # Windows system
    r'C:\\',         # Windows system
    r'/tmp/',        # Temp files (unless explicitly allowed)
    r'~/',           # Home directory
]

# SQL Injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
    r"(--|#|/\*|\*/)",
    r"(\bor\b\s+\d+\s*=\s*\d+)",
    r"(\band\b\s+\d+\s*=\s*\d+)",
    r"('|\")(\s)*(or|and|union|select)(\s)+(\w+\s*=|\w+\s*\(|from\s+\w+|into\s+\w+|\d+)",  # More specific SQL context
    r"\b(script|javascript|vbscript|onload|onerror|onclick)\b",
    r"(\<\s*script)",
    r"(\bxp_cmdshell\b)"
]

# XSS patterns
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"vbscript:",
    r"onload\s*=",
    r"onerror\s*=",
    r"onclick\s*=",
    r"onmouseover\s*=",
    r"<iframe[^>]*>",
    r"<object[^>]*>",
    r"<embed[^>]*>",
]

# Command injection patterns
COMMAND_INJECTION_PATTERNS = [
    r"[;&|`$(){}[\]<>]",
    r"\b(rm|del|format|fdisk|mkfs)\b",
    r"\b(cat|type|more|less)\b\s+/",
    r"\b(wget|curl|nc|netcat)\b",
    r"\b(python|perl|php|ruby|bash|sh|cmd|powershell)\b\s+",
]

@dataclass
class SecurityConfig:
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_sql_injection_protection: bool = True
    enable_xss_protection: bool = True
    enable_command_injection_protection: bool = True
    enable_path_traversal_protection: bool = True
    max_prompt_length: int = 10000
    max_filename_length: int = 255
    rate_limit_requests_per_minute: int = 100

    # API Key security
    api_key_min_length: int = 32
    api_key_rotation_hours: int = 24 * 7  # 1 week
    enable_api_key_validation: bool = True

@dataclass
class SecurityViolation:
    violation_type: str
    details: str
    client_ip: str
    timestamp: datetime
    request_path: str
    severity: str  # low, medium, high, critical

class SecurityValidator:
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.violations: List[SecurityViolation] = []

        # Compile regex patterns for performance
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(p, re.IGNORECASE) for p in XSS_PATTERNS]
        self.cmd_patterns = [re.compile(p, re.IGNORECASE) for p in COMMAND_INJECTION_PATTERNS]
        self.path_patterns = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATH_PATTERNS]

    def sanitize_input(self, input_text: str, field_name: str = "input") -> str:
        """Input'u güvenli hale getirir"""
        if not isinstance(input_text, str):
            raise HTTPException(status_code=400, detail=f"Invalid input type for {field_name}")

        # Boş veya None kontrolü
        if not input_text or not input_text.strip():
            return ""

        # Uzunluk kontrolü
        if len(input_text) > self.config.max_prompt_length:
            raise HTTPException(status_code=400,
                              detail=f"{field_name} too long (max {self.config.max_prompt_length} chars)")

        # Tehlikeli karakterleri temizle
        sanitized = input_text.strip()

        # SQL Injection kontrolü
        if self.config.enable_sql_injection_protection:
            self._check_sql_injection(sanitized, field_name)

        # XSS kontrolü
        if self.config.enable_xss_protection:
            self._check_xss(sanitized, field_name)

        # Command injection kontrolü
        if self.config.enable_command_injection_protection:
            self._check_command_injection(sanitized, field_name)

        # HTML entity encoding for safety
        sanitized = self._html_escape(sanitized)

        logger.info(f"[SECURITY] Input sanitized: {field_name}")
        return sanitized

    def sanitize_ai_prompt(self, input_text: str, field_name: str = "ai_prompt") -> str:
        """AI prompt'larını güvenli hale getirir (minimal güvenlik kontrolleri)"""
        if not isinstance(input_text, str):
            raise HTTPException(status_code=400, detail=f"Invalid input type for {field_name}")

        # Boş veya None kontrolü
        if not input_text or not input_text.strip():
            return ""

        # Uzunluk kontrolü
        if len(input_text) > self.config.max_prompt_length:
            raise HTTPException(status_code=400,
                              detail=f"{field_name} too long (max {self.config.max_prompt_length} chars)")

        # Tehlikeli karakterleri temizle
        sanitized = input_text.strip()

        # AI PROMPT'LARI İÇİN MINIMAL GÜVENLİK:
        # - SQL Injection kontrolü YOK (veritabanı işlemi yok)
        # - Command injection kontrolü YOK (sistem komut çalıştırma yok)
        # - XSS kontrolü sadece çok tehlikeli script tag'leri için

        # Sadece açık script tag'lerini kontrol et
        if "<script" in sanitized.lower() or "javascript:" in sanitized.lower():
            violation = SecurityViolation(
                violation_type="script_injection",
                details=f"Script tag detected in {field_name}",
                client_ip="unknown",
                timestamp=datetime.now(),
                request_path="unknown",
                severity="medium"
            )
            self.violations.append(violation)
            logger.warning(f"[SECURITY] Script tag detected in {field_name}")
            raise HTTPException(status_code=400, detail="Invalid input detected")

        # HTML entity encoding YOK - AI model'e gönderilecek text

        logger.info(f"[SECURITY] AI prompt sanitized (minimal): {field_name}")
        return sanitized

    def _check_sql_injection(self, text: str, field_name: str):
        """SQL injection saldırılarını kontrol eder"""
        for pattern in self.sql_patterns:
            if pattern.search(text):
                violation = SecurityViolation(
                    violation_type="sql_injection",
                    details=f"SQL injection pattern detected in {field_name}: {pattern.pattern}",
                    client_ip="unknown",
                    timestamp=datetime.now(),
                    request_path="unknown",
                    severity="high"
                )
                self.violations.append(violation)
                logger.warning(f"[SECURITY] SQL injection detected in {field_name}: {text[:100]}")
                raise HTTPException(status_code=400, detail="Invalid input detected")

    def _check_xss(self, text: str, field_name: str):
        """XSS saldırılarını kontrol eder"""
        for pattern in self.xss_patterns:
            if pattern.search(text):
                violation = SecurityViolation(
                    violation_type="xss_attempt",
                    details=f"XSS pattern detected in {field_name}: {pattern.pattern}",
                    client_ip="unknown",
                    timestamp=datetime.now(),
                    request_path="unknown",
                    severity="high"
                )
                self.violations.append(violation)
                logger.warning(f"[SECURITY] XSS detected in {field_name}: {text[:100]}")
                raise HTTPException(status_code=400, detail="Invalid input detected")

    def _check_command_injection(self, text: str, field_name: str):
        """Command injection saldırılarını kontrol eder"""
        for pattern in self.cmd_patterns:
            if pattern.search(text):
                violation = SecurityViolation(
                    violation_type="command_injection",
                    details=f"Command injection pattern detected in {field_name}: {pattern.pattern}",
                    client_ip="unknown",
                    timestamp=datetime.now(),
                    request_path="unknown",
                    severity="high"
                )
                self.violations.append(violation)
                logger.warning(f"[SECURITY] Command injection detected in {field_name}: {text[:100]}")
                raise HTTPException(status_code=400, detail="Invalid input detected")

    def _html_escape(self, text: str) -> str:
        """HTML karakterlerini escape eder"""
        html_escape_table = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
        }
        return "".join(html_escape_table.get(c, c) for c in text)

    def validate_file_path(self, file_path: str) -> str:
        """File path'ini güvenlik açısından validate eder"""
        if not file_path:
            raise HTTPException(status_code=400, detail="File path cannot be empty")

        # Path traversal kontrolü
        if self.config.enable_path_traversal_protection:
            for pattern in self.path_patterns:
                if pattern.search(file_path):
                    violation = SecurityViolation(
                        violation_type="path_traversal",
                        details=f"Path traversal attempt: {file_path}",
                        client_ip="unknown",
                        timestamp=datetime.now(),
                        request_path="unknown",
                        severity="high"
                    )
                    self.violations.append(violation)
                    logger.warning(f"[SECURITY] Path traversal detected: {file_path}")
                    raise HTTPException(status_code=400, detail="Invalid file path")

        # Normalize path
        normalized_path = os.path.normpath(file_path)

        # File extension kontrolü
        file_ext = Path(normalized_path).suffix.lower()
        if file_ext not in ALLOWED_FILE_EXTENSIONS:
            raise HTTPException(status_code=400,
                              detail=f"File extension {file_ext} not allowed")

        return normalized_path

    def validate_file_content(self, file_path: str, content: bytes = None) -> bool:
        """File içeriğini güvenlik açısından validate eder"""
        path_obj = Path(file_path)

        # File extension kontrolü
        file_ext = path_obj.suffix.lower()
        if file_ext not in ALLOWED_FILE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed")

        # File boyut kontrolü
        if content:
            file_size = len(content)
        elif path_obj.exists():
            file_size = path_obj.stat().st_size
        else:
            return True  # File yoksa size check yapamayız

        max_size = MAX_FILE_SIZES.get(file_ext, 1024 * 1024)  # Default 1MB
        if file_size > max_size:
            raise HTTPException(status_code=400,
                              detail=f"File too large ({file_size} > {max_size} bytes)")

        # PDF için ek kontroller
        if file_ext == '.pdf' and content:
            if not content.startswith(b'%PDF-'):
                raise HTTPException(status_code=400, detail="Invalid PDF file format")

        # Text dosyaları için encoding kontrolü
        if file_ext in ['.txt', '.md'] and content:
            try:
                content.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Invalid text file encoding")

        logger.info(f"[SECURITY] File validated: {file_path}")
        return True

    def get_security_violations(self, hours: int = 24) -> List[SecurityViolation]:
        """Son X saatteki güvenlik ihlallerini döndürür"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [v for v in self.violations if v.timestamp > cutoff_time]

class APIKeyManager:
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.valid_keys: Dict[str, Dict] = {}
        self.key_usage: Dict[str, List[datetime]] = {}

    def generate_api_key(self, user_id: str = "default") -> str:
        """Güvenli API key üretir"""
        # 32 byte random key (256 bit)
        key_bytes = secrets.token_bytes(32)
        api_key = base64.urlsafe_b64encode(key_bytes).decode('ascii')

        # Key metadata
        self.valid_keys[api_key] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_used": None,
            "usage_count": 0,
            "is_active": True
        }

        self.key_usage[api_key] = []

        logger.info(f"[SECURITY] API key generated for user: {user_id}")
        return api_key

    def validate_api_key(self, api_key: str) -> bool:
        """API key'i validate eder"""
        if not self.config.enable_api_key_validation:
            return True  # Validation kapalıysa her şeyi kabul et

        if not api_key:
            return False

        if api_key not in self.valid_keys:
            logger.warning(f"[SECURITY] Invalid API key used: {api_key[:8]}...")
            return False

        key_info = self.valid_keys[api_key]

        # Active kontrolü
        if not key_info.get("is_active", False):
            logger.warning(f"[SECURITY] Inactive API key used: {api_key[:8]}...")
            return False

        # Expiration kontrolü
        created_at = key_info["created_at"]
        expiration_time = created_at + timedelta(hours=self.config.api_key_rotation_hours)

        if datetime.now() > expiration_time:
            logger.warning(f"[SECURITY] Expired API key used: {api_key[:8]}...")
            self.revoke_api_key(api_key)
            return False

        # Usage tracking
        self._track_key_usage(api_key)

        return True

    def _track_key_usage(self, api_key: str):
        """API key kullanımını takip eder"""
        now = datetime.now()

        # Usage count güncelle
        if api_key in self.valid_keys:
            self.valid_keys[api_key]["last_used"] = now
            self.valid_keys[api_key]["usage_count"] += 1

        # Usage history
        if api_key not in self.key_usage:
            self.key_usage[api_key] = []

        self.key_usage[api_key].append(now)

        # Son 1 saatlik kayıtları tut
        one_hour_ago = now - timedelta(hours=1)
        self.key_usage[api_key] = [
            usage_time for usage_time in self.key_usage[api_key]
            if usage_time > one_hour_ago
        ]

    def revoke_api_key(self, api_key: str):
        """API key'i iptal eder"""
        if api_key in self.valid_keys:
            self.valid_keys[api_key]["is_active"] = False
            logger.info(f"[SECURITY] API key revoked: {api_key[:8]}...")

    def rotate_api_key(self, old_key: str, user_id: str = None) -> str:
        """API key'i rotate eder"""
        if old_key in self.valid_keys:
            user_id = user_id or self.valid_keys[old_key]["user_id"]
            self.revoke_api_key(old_key)

        new_key = self.generate_api_key(user_id)
        logger.info(f"[SECURITY] API key rotated for user: {user_id}")
        return new_key

    def get_key_stats(self) -> Dict[str, Any]:
        """API key istatistiklerini döndürür"""
        total_keys = len(self.valid_keys)
        active_keys = sum(1 for k in self.valid_keys.values() if k.get("is_active", False))

        return {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "inactive_keys": total_keys - active_keys,
            "rotation_hours": self.config.api_key_rotation_hours,
            "validation_enabled": self.config.enable_api_key_validation
        }

# Global instances
security_config = SecurityConfig()
security_validator = SecurityValidator(security_config)
api_key_manager = APIKeyManager(security_config)

def create_security_headers() -> Dict[str, str]:
    """Güvenlik header'larını oluşturur"""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }

def hash_sensitive_data(data: str, salt: str = None) -> str:
    """Hassas veriyi hash'ler"""
    if salt is None:
        salt = secrets.token_hex(16)

    # PBKDF2 with SHA256
    key = hashlib.pbkdf2_hmac('sha256', data.encode('utf-8'), salt.encode('utf-8'), 100000)
    return f"{salt}:{base64.b64encode(key).decode('ascii')}"

def verify_sensitive_data(data: str, hashed: str) -> bool:
    """Hash'lenmiş veriyi verify eder"""
    try:
        salt, key = hashed.split(':')
        expected_key = hashlib.pbkdf2_hmac('sha256', data.encode('utf-8'), salt.encode('utf-8'), 100000)
        expected_hash = base64.b64encode(expected_key).decode('ascii')
        return hmac.compare_digest(key, expected_hash)
    except Exception:
        return False

def get_client_ip(request: Request) -> str:
    """Client IP adresini güvenli şekilde alır"""
    # Proxy headers kontrolü
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    return request.client.host if request.client else "unknown"