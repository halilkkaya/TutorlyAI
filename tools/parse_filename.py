import re
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# RAG sistemi için global değişkenler
vectorstore = None
embedding_model = None
text_splitter = None

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

_PAT_FULL = re.compile(
    r"^(?P<grade>\d{1,2})_sinif_(?P<subject>[a-z0-9_]+)_unite_(?P<unit>\d{1,2})_(?P<slug>[a-z0-9_]+)\.pdf$",
    re.IGNORECASE
)

SUBJECT_ALIASES = {
    "din": "din",
    "din_kültürü": "din",
    "din_kulturu": "din",
    "inkilap": "inkilap",
    "inkılap": "inkilap",
    "cografya": "cografya",
    "coğrafya": "cografya",
    "turk_dili_ve_edebiyati": "turkdili",
    "turk_dili_ve_edebiyatı": "turkdili",
    "turkce": "turkce",
    "türkçe": "turkce",
    "biyoloji": "biyoloji",
    "fizik": "fizik",
    "kimya": "kimya",
    "matematik": "matematik",
    "tarih": "tarih"
}

def canonical_subject(raw: str) -> str:
    s = raw.lower().strip()
    s = re.sub(r'[_\s\-]+', '_', s)
    return SUBJECT_ALIASES.get(s, s)


def parse_filename_for_metadata(filename: str):
    """
    Beklenen format:
    <grade>_sinif_<subject>_unite_<unit>_<slug>.pdf
    Örn: 9_sinif_biyoloji_unite_01_yasam.pdf
         9_sinif_din_unite_2_islamda_inanc_esaslari.pdf
    """
    name = filename
    if name.lower().endswith(".pdf"):
        name = name[:-4]

    m = _PAT_FULL.match(filename)
    if not m:
        # Uymayan dosyaları sessizce geçmek yerine logla:
        logger.warning(f"[PARSE_FILENAME] Uyumsuz dosya adı: {filename}")
        return None

    grade = int(m.group("grade"))
    subject = canonical_subject(m.group("subject"))
    unit = int(m.group("unit"))
    slug = m.group("slug").lower().strip("_")
    
    # konu_slug'ı Türkçe karakterlerden arındır
    normalized_slug = normalize_turkish_chars(slug)

    meta = {
        "sinif": grade,
        "ders": subject,
        "unite": unit,
        "konu_slug": normalized_slug
    }
    logger.info(f"[PARSE_FILENAME] Metadata ayrıştırıldı: {filename}, {meta}")
    return meta
