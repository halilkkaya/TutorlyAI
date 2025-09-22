from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from typing import Optional, AsyncGenerator
import json
from dotenv import load_dotenv

# fal-client kütüphanesini içe aktar
import fal_client

# Ortam değişkenlerini yükle
load_dotenv()

# FastAPI uygulaması oluştur
app = FastAPI(
    title="Fal.ai Any-LLM (Gemini 2.5 Flash) API",
    description="fal.ai 'any-llm' modeli ile Gemini 2.5 Flash kullanarak streaming metin üretme API'si",
    version="1.0.1"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model - istek yapısı
class TextGenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    # Ek parametreler eklenebilir (top_p, frequency_penalty vb.)

# Fal.ai API anahtarını kontrol et
FAL_KEY = os.getenv("FAL_KEY")

if not FAL_KEY:
    raise ValueError("FAL_KEY ortam değişkeni ayarlanmamış! Lütfen .env dosyasını kontrol edin veya ortam değişkenini ayarlayın.")

# Model adını belirle
MODEL_NAME = "google/gemini-2.5-flash"
FAL_MODEL_GATEWAY = "fal-ai/any-llm"

@app.get("/")
async def root():
    """Ana endpoint"""
    return {
        "message": "Fal.ai Any-LLM (Gemini 2.5 Flash) API çalışıyor!",
        "status": "active",
        "docs_url": "/docs",
        "streaming_endpoint": "/generate/stream"
    }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü endpoint'i"""
    return {"status": "healthy"}

# Streaming için asenkron jeneratör fonksiyonu
async def generate_stream(request: TextGenerationRequest) -> AsyncGenerator[str, None]:
    """
    Fal.ai'dan gelen stream olaylarını SSE (Server-Sent Events) formatına dönüştürür.
    """
    try:
        # fal.ai streaming aboneliği başlat
        # aclient.stream_connect() veya asubscribe() kullanılabilir.
        # Burada asubscribe() ile daha modern bir yaklaşım kullanıyoruz.

        handler = await fal_client.asubscribe(
            FAL_MODEL_GATEWAY,
            arguments={
                "model_name": MODEL_NAME,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
        )

        # Gelen olayları (events) dinle
        async for event in handler:
            # 'chunk' anahtarı, o anda oluşan metin parçasını içerir
            if "chunk" in event and event["chunk"]:
                chunk_data = {
                    "text": event["chunk"],
                    "done": False
                }
                # SSE formatı: "data: <json_string>\n\n"
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # 'response' anahtarı, işlemin tamamlandığını ve son halini belirtir
            elif "response" in event:
                 final_data = {
                    "text": "",
                    "done": True,
                    "full_response": event["response"]
                }
                 yield f"data: {json.dumps(final_data)}\n\n"
                 break

    except Exception as e:
        # Hata durumunda SSE hata olayı gönder
        error_data = {"error": str(e)}
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

@app.post("/generate/stream")
async def stream_text(request: TextGenerationRequest):
    """
    Gemini 2.5 Flash ile streaming (akış) halinde metin üretme endpoint'i.
    Yanıt, Server-Sent Events (SSE) olarak döner.
    """
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream"
    )

# Geriye dönük uyumluluk için normal (blocking) bir endpoint de ekleyelim
@app.post("/generate")
async def generate_text_sync(request: TextGenerationRequest):
    """
    Gemini 2.5 Flash ile tek seferde (non-streaming) metin üretme endpoint'i.
    """
    try:
        print(f"[LOG] /generate endpoint'ine istek geldi - prompt: {request.prompt[:50]}...")
        
        # Düzeltilmiş fal.ai çağrısı (asenkron)
        result = await fal_client.run_async(
            "fal-ai/any-llm",  # Model gateway/endpoint adı
            arguments={
                "model": MODEL_NAME,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
        )

        print(f"[LOG] fal.ai'dan yanıt alındı: {result}")
        
        # Yanıt formatı değişmiş olabilir, 'text' yerine 'output' olabilir.
        # Gelen 'result' objesini loglayarak doğru anahtarı teyit edin.
        generated_text = result.get("output", "Metin üretilemedi.")
        
        print(f"[LOG] Üretilen metin: {generated_text[:100]}...")

        return {
            "generated_text": generated_text,
            "model": MODEL_NAME,
            "gateway": "fal-ai/any-llm"
        }

    except Exception as e:
        print(f"[HATA] /generate endpoint'inde hata oluştu: {str(e)}")
        print(f"[HATA] Hata tipi: {type(e).__name__}")
        import traceback
        print(f"[HATA] Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Metin üretme hatası: {str(e)}"
        )

@app.get("/models")
async def get_model_info():
    """Kullanılan model hakkında bilgi verir."""
    return {
        "service": "fal.ai",
        "gateway_model": FAL_MODEL_GATEWAY,
        "configured_llm": MODEL_NAME
    }

if __name__ == "__main__":
    import uvicorn
    print(f"API başlatılıyor. Model: {MODEL_NAME} via {FAL_MODEL_GATEWAY}")
    uvicorn.run(app, host="0.0.0.0", port=8000)