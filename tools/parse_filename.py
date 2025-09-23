import re

# RAG sistemi için global değişkenler
vectorstore = None
embedding_model = None
text_splitter = None

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
        print(f"[METADATA] UYUMSUZ AD: {filename}")
        return None

    grade = int(m.group("grade"))
    subject = canonical_subject(m.group("subject"))
    unit = int(m.group("unit"))
    slug = m.group("slug").lower().strip("_")

    meta = {
        "sinif": grade,
        "ders": subject,
        "unite": unit,
        "konu_slug": slug
    }
    print(f"[METADATA] Ayrıştırıldı: {filename} -> {meta}")
    return meta
