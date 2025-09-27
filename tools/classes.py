from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal



class TextGenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    tools: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    
    # Hibrit arama parametreleri
    use_hybrid: Optional[bool] = True
    semantic_weight: Optional[float] = 0.7
    keyword_weight: Optional[float] = 0.3
    score_threshold: Optional[float] = 0.25
    search_k: Optional[int] = 5


# RAG sistemi için yanıt modeli
class ToolResponse(BaseModel):
    tool_call_id: str
    result: str


# Quiz sistemi için modeller
class QuizRequest(BaseModel):
    sinif: int = Field(
        ..., 
        ge=1, 
        le=12, 
        description="Sınıf numarası (1-12 arası)"
    )
    ders: str = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="Ders adı (otomatik normalize edilir)"
    )
    konu: str = Field(
        ..., 
        min_length=1, 
        max_length=200,
        description="Konu adı"
    )
    soru_sayisi: int = Field(
        ..., 
        ge=1, 
        le=10, 
        description="Soru sayısı (1-10 arası)"
    )
    soru_tipi: Literal["coktan_secmeli", "acik_uclu"] = Field(
        ...,
        description="Soru tipi"
    )
    zorluk: Literal["kolay", "orta", "zor"] = Field(
        default="orta",
        description="Zorluk seviyesi"
    )
    dil: Literal["tr", "en"] = Field(
        default="tr",
        description="Dil (şimdilik sadece Türkçe)"
    )
    
    @validator('ders', 'konu')
    def validate_text_fields(cls, v):
        """Metin alanlarını temizle ve validate et"""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError('Bu alan boş olamaz')
        return v
    
    @validator('sinif')
    def validate_grade(cls, v):
        """Sınıf kontrolü - özel mesaj için"""
        if v not in range(1, 13):
            raise ValueError(f'Geçersiz sınıf: {v}. 1-12 arası olmalı.')
        return v


class MultipleChoiceQuestion(BaseModel):
    soru: str = Field(..., min_length=5, max_length=500, description="Soru metni")
    a: str = Field(..., min_length=1, max_length=200, description="A şıkkı")
    b: str = Field(..., min_length=1, max_length=200, description="B şıkkı") 
    c: str = Field(..., min_length=1, max_length=200, description="C şıkkı")
    d: str = Field(..., min_length=1, max_length=200, description="D şıkkı")
    cevap: Literal["a", "b", "c", "d"] = Field(..., description="Doğru cevap şıkkı")
    aciklama: str = Field(..., min_length=5, max_length=1000, description="Cevap açıklaması")


class OpenEndedQuestion(BaseModel):
    soru: str = Field(..., min_length=5, max_length=500, description="Açık uçlu soru")
    cevap: str = Field(..., min_length=10, max_length=2000, description="Sorunun cevabı")
    aciklama: str = Field(..., min_length=5, max_length=1000, description="Cevap açıklaması")


class QuizResponse(BaseModel):
    quiz_info: Dict[str, Any]  # Metadata about the quiz
    sorular: List[Dict[str, Any]]  # List of questions
    total_soru: int
    soru_tipi: str
    created_at: str


# İngilizce öğrenme sistemi için modeller
class EnglishLearningRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Kullanıcının İngilizce sorusu/isteği")
    max_tokens: Optional[int] = Field(default=1000, ge=50, le=4000, description="Maksimum token sayısı")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Yaratıcılık seviyesi")

    @validator('prompt')
    def validate_prompt(cls, v):
        """Prompt'u temizle ve validate et"""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError('Prompt boş olamaz')
        return v


class EnglishLearningResponse(BaseModel):
    generated_text: str = Field(..., description="Üretilen İngilizce öğrenme içeriği")
    detected_level: str = Field(..., description="Algılanan İngilizce seviyesi (A1, A2, B1, B2, C1, C2)")
    system_prompt_used: str = Field(..., description="Kullanılan system prompt açıklaması")
    clean_prompt: str = Field(..., description="Seviye bilgisi temizlenmiş prompt")


# Görsel üretimi için modeller
class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Görsel üretimi için prompt")
    workflow_id: str = Field(default="workflows/halillllibrahim58/teach-img-model", description="Fal AI workflow ID")
    max_tokens: Optional[int] = Field(default=1000, ge=50, le=4000, description="Maksimum token sayısı")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Yaratıcılık seviyesi")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Prompt'u temizle ve validate et"""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError('Prompt boş olamaz')
        return v


class ImageGenerationResponse(BaseModel):
    success: bool = Field(..., description="İşlem başarı durumu")
    image_url: Optional[str] = Field(None, description="Ana görselin URL'si (ilk görsel)")
    workflow_id: str = Field(..., description="Kullanılan workflow ID")
    prompt: str = Field(..., description="Kullanılan prompt")
    error_message: Optional[str] = Field(None, description="Hata mesajı (varsa)")
    generated_at: str = Field(..., description="Üretim zamanı")
    all_images: Optional[List[str]] = Field(None, description="Tüm üretilen görsellerin URL'leri")
    total_images: Optional[int] = Field(None, description="Toplam üretilen görsel sayısı")