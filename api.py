from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import Optional
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# FastAPI uygulaması oluştur
app = FastAPI(
    title="Gemini API Model",
    description="Gemini AI ile metin üretme API'si",
    version="1.0.0"
)

# CORS ayarları - başkalarının bağlanabilmesi için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da belirli domain'leri belirtin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model - istek yapısı
class TextGenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

class TextGenerationResponse(BaseModel):
    generated_text: str
    model: str

# Gemini API anahtarını ortam değişkeninden al
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmamış!")

# Gemini modelini yapılandır
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')


@app.get("/")
async def root():
    """Ana endpoint - API'nin çalıştığını kontrol etmek için"""
    return {
        "message": "Gemini API Model çalışıyor!",
        "status": "active",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü endpoint'i"""
    return {"status": "healthy"}

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """
    Gemini AI ile metin üretme endpoint'i

    Args:
        request: TextGenerationRequest - prompt, max_tokens, temperature

    Returns:
        TextGenerationResponse - üretilen metin ve model bilgisi
    """
    try:
        # Generation config
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Metin üret
        response = model.generate_content(
            request.prompt,
            generation_config=generation_config
        )

        # Yanıtı işle
        generated_text = response.text if response.text else "Metin üretilemedi."

        return TextGenerationResponse(
            generated_text=generated_text,
            model="gemini-2.5-flash"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Metin üretme hatası: {str(e)}"
        )

@app.get("/models")
async def get_available_models():
    """Mevcut modelleri listele"""
    try:
        models = [model.name for model in genai.list_models()]
        return {
            "available_models": models,
            "current_model": "gemini-2.5-flash"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Modeller alınamadı: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
