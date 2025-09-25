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