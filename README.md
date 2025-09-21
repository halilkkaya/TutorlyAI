# Gemini API Model - FastAPI Uygulaması

Bu uygulama Google'ın Gemini AI modelini kullanarak metin üretme API'si sağlar.

## Özellikler

- ✅ FastAPI ile hızlı ve modern API
- ✅ Gemini AI entegrasyonu
- ✅ CORS desteği (başkaları bağlanabilir)
- ✅ Özelleştirilebilir parametreler (max_tokens, temperature)
- ✅ Hata yönetimi
- ✅ Otomatik dokümantasyon (/docs)

## Kurulum


### 1. Gerekli Paketleri Yükleyin

```bash
pip install fastapi uvicorn google-generativeai
```

Veya requirements.txt dosyasını kullanın:

```bash
pip install -r requirements.txt
```

### 2. Gemini API Anahtarını Ayarlayın

Ortam değişkeni olarak GEMINI_API_KEY'i ayarlayın:

#### Windows (CMD):
```bash
set GEMINI_API_KEY=your_api_key_here
```

#### Windows (PowerShell):
```bash
$env:GEMINI_API_KEY="your_api_key_here"
```

#### Linux/Mac:
```bash
export GEMINI_API_KEY=your_api_key_here
```

**Gemini API anahtarını almak için:**
1. [Google AI Studio](https://makersuite.google.com/app/apikey)'ya gidin
2. Yeni bir API anahtarı oluşturun
3. Anahtarı ortam değişkeni olarak ayarlayın

## Çalıştırma

### Otomatik (Tavsiye Edilen)

```bash
python api.py
```

### Manuel

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API şu adreste çalışacak: `http://localhost:8000`

### 🌐 Web Arayüzü

Basit bir web arayüzü de oluşturduk. İki terminal açın:

**Terminal 1 - API Sunucusu:**
```bash
python api.py
```

**Terminal 2 - Web Arayüzü:**
```bash
python web_app.py
```

Web arayüzü: `http://localhost:5000`

## API Endpoint'leri

### 1. Ana Sayfa
- **GET** `/`
- API'nin çalıştığını kontrol eder

### 2. Sağlık Kontrolü
- **GET** `/health`
- Sunucunun durumunu kontrol eder

### 3. Metin Üretme (Ana Endpoint)
- **POST** `/generate`
- Gemini AI ile metin üretir

**İstek Formatı:**
```json
{
    "prompt": "Yapay zeka hakkında bir paragraf yaz",
    "max_tokens": 1000,
    "temperature": 0.7
}
```

**Yanıt Formatı:**
```json
{
    "generated_text": "Üretilen metin burada olacak...",
    "model": "gemini-pro"
}
```

### 4. Mevcut Modeller
- **GET** `/models`
- Kullanılabilir modelleri listeler

## Kullanım Örnekleri

### Python ile İstek Gönderme

```python
import requests

# API URL'i
url = "http://localhost:8000/generate"

# İstek verisi
data = {
    "prompt": "Python programlama dilinin avantajları nelerdir?",
    "max_tokens": 500,
    "temperature": 0.8
}

# POST isteği gönder
response = requests.post(url, json=data)

# Yanıtı yazdır
print(response.json())
```

### cURL ile İstek Gönderme

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Geleceğin teknolojileri hakkında düşüncelerini paylaş",
       "max_tokens": 300,
       "temperature": 0.6
     }'
```

## Dokümantasyon

Otomatik API dokümantasyonu için:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Güvenlik Notları

⚠️ **Dikkat**: Şu anda CORS ayarları tüm origin'lere izin veriyor (`allow_origins=["*"]`). Production ortamında güvenliği artırmak için belirli domain'leri belirtin:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "https://anotherdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Sorun Giderme

### Yaygın Hatalar

1. **GEMINI_API_KEY hatası**: API anahtarını doğru ayarladığınızdan emin olun
2. **Bağlantı reddedildi**: Sunucunun çalıştığından emin olun
3. **500 Internal Server Error**: Prompt'unuzu kontrol edin, boş olmamalı

### Debug Modu

Daha detaylı loglama için:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## Lisans

Bu proje eğitim amaçlı hazırlanmıştır.
