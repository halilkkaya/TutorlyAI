"""
Ders Adı Normalizasyon Sistemi
Model'in gönderdiği ders adlarını kanonik forma çevirir
"""

import re
from typing import Optional
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
        'ı': 'i', 'I': 'I', 'İ': 'I', 'i̇': 'i',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U'
    }
    
    normalized = text
    for turkish_char, normal_char in turkish_chars.items():
        normalized = normalized.replace(turkish_char, normal_char)
    
    return normalized

# Kapsamlı ders adı eşleştirme sözlüğü
COMPREHENSIVE_SUBJECT_MAPPING = {
    # Din Kültürü ve Ahlak Bilgisi
    "din": "din",
    "din_kulturu": "din", 
    "din_kültürü": "din",
    "din kulturu": "din",
    "din kültürü": "din",
    "din_kulturu_ve_ahlak_bilgisi": "din",
    "din kültürü ve ahlak bilgisi": "din",
    "dkab": "din",
    "ahlak": "din",
    
    # İnkılap Tarihi  
    "inkilap": "inkilap",
    "inkılap": "inkilap",
    "inkilap_tarihi": "inkilap",
    "inkılap tarihi": "inkilap",
    "inkilap tarihi": "inkilap",
    "ınkılap tarihi": "inkilap",  # Türkçe karakter normalizasyonu sonrası
    "inkilap tarihi": "inkilap",  # İ -> I dönüşümü sonrası (lowercase)
    "atatürk ilkeleri": "inkilap",
    "atatürk_ilkeleri": "inkilap",
    "ataturk ilkeleri": "inkilap",
    "atatürk_ilkeleri_ve_inkilap_tarihi": "inkilap",
    "ataturk ılkelerı": "inkilap",  # Türkçe normalize sonrası
    "atatürk ılkelerı": "inkilap", # Türkçe normalize sonrası
    "cumhuriyet tarihi": "inkilap",
    
    # Coğrafya
    "cografya": "cografya",
    "coğrafya": "cografya",
    "cografya": "cografya",
    "geografya": "cografya",
    "jeografi": "cografya",
    
    # Türk Dili ve Edebiyatı
    "turkdili": "turkdili",
    "turk_dili": "turkdili",
    "türk dili": "turkdili",
    "turk dili": "turkdili",
    "turk_dili_ve_edebiyati": "turkdili",
    "türk dili ve edebiyatı": "turkdili",
    "turk dili ve edebiyati": "turkdili",
    "türk dili ve edebiyati": "turkdili",
    "edebiyat": "turkdili",
    "türk edebiyatı": "turkdili",
    "turk edebiyati": "turkdili",
    "dil ve anlatım": "turkdili",
    "dil_ve_anlatim": "turkdili",
    
    # Türkçe (ayrı ders olarak)
    "turkce": "turkce", 
    "türkçe": "turkce",
    "türkçe dersi": "turkce",
    "turkce dersi": "turkce",
    
    # Matematik
    "matematik": "matematik",
    "mat": "matematik",
    "matematık": "matematik",
    "matematik dersi": "matematik",
    "math": "matematik",
    "matematik": "matematik",
    "math": "matematik",
    
    # Fen Bilimleri
    "biyoloji dersi": "biyoloji",
    "biology": "biyoloji",
    "biyoloji": "biyoloji",

    "fizik": "fizik",    
    "fizik_dersi": "fizik",
    "fızık": "fizik",
    "fizik dersi": "fizik",
    "fızık": "fizik",

    "kimya": "kimya",
    "kimya dersi": "kimya",
    "kımya": "kimya",
    "kimya dersi": "kimya",
    "kımya": "kimya",

    # Tarih
    "tarih": "tarih",
    "history": "tarih",
    "genel tarih": "tarih",
    "dünya tarihi": "tarih",
    "dunya tarihi": "tarih",

 
    "tarih dersi": "tarih",
    "tarih dersi": "tarih",
   
    # Felsefe (varsa)
    "felsefe": "felsefe",
    "philosophy": "felsefe",
}

def normalize_subject_name(subject_name: str) -> Optional[str]:
    """
    Model'in gönderdiği ders adını kanonik forma çevirir
    
    Args:
        subject_name: Model'in gönderdiği ders adı
    
    Returns:
        Kanonik ders adı veya None (tanınmazsa)
    """
    if not subject_name or not isinstance(subject_name, str):
        return None
    
    
    
    # Temel normalizasyon
    normalized = subject_name.strip().lower()
    
    # Türkçe karakterleri düzelt
    normalized = normalize_turkish_chars(normalized)
    
    # Noktalama işaretlerini ve fazla boşlukları temizle
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Boşlukları alt çizgi ile değiştir (hem orijinal hem de alt çizgili versiyonu dene)
    normalized_underscore = normalized.replace(' ', '_')
    
    # Direct match dene
    if normalized in COMPREHENSIVE_SUBJECT_MAPPING:
        result = COMPREHENSIVE_SUBJECT_MAPPING[normalized]
        logger.info(f"[SUBJECT_NORM] '{subject_name}' -> '{result}' (direct match)")
        return result
    
    # Alt çizgili versiyonu dene
    if normalized_underscore in COMPREHENSIVE_SUBJECT_MAPPING:
        result = COMPREHENSIVE_SUBJECT_MAPPING[normalized_underscore]
        logger.info(f"[SUBJECT_NORM] '{subject_name}' -> '{result}' (underscore match)")
        return result
    
    # Partial match dene - uzun pattern'lar önce (daha spesifik eşleşmeler)
    patterns_by_length = sorted(COMPREHENSIVE_SUBJECT_MAPPING.items(), key=lambda x: len(x[0]), reverse=True)
    
    for pattern, canonical in patterns_by_length:
        # Pattern'in normalized içinde geçip geçmediğini kontrol et
        if pattern in normalized:
            logger.info(f"[SUBJECT_NORM] '{subject_name}' -> '{canonical}' (partial match: '{pattern}')")
            return canonical
    
    # Tersine partial match - normalized'ın pattern içinde geçmesi
    for pattern, canonical in patterns_by_length:
        if normalized in pattern and len(normalized) > 3:  # Çok kısa kelimelerle eşleşmesin
            logger.info(f"[SUBJECT_NORM] '{subject_name}' -> '{canonical}' (reverse partial match: '{pattern}')")
            return canonical
    
    # Eğer hiçbir şey bulunamazsa log ve None döndür
    logger.warning(f"[SUBJECT_NORM] ⚠️  '{subject_name}' -> TANINAMADI")
    return None

def get_valid_subjects() -> list:
    """Geçerli kanonik ders adları listesi"""
    return ["din", "inkilap", "cografya", "turkdili", "turkce", "matematik", "biyoloji", "fizik", "kimya", "tarih"]

def validate_subject(subject_name: str) -> bool:
    """Ders adının geçerli olup olmadığını kontrol et"""
    return subject_name in get_valid_subjects()
