# TutorlyAI - Fal.ai + RAG Sistemi

Bu uygulama Fal.ai platformu üzerinden Gemini 2.5 Flash modelini kullanarak gelişmiş metin üretme ve RAG (Retrieval-Augmented Generation) sistemi sağlar.

## Özellikler

- ✅ **Fal.ai Entegrasyonu** - Gemini 2.5 Flash modeli
- ✅ **RAG Sistemi** - 9-10-11-12. sınıf kitapları ile arama
- ✅ **Tools/Fonksiyon Desteği** - Model kitap arama yapabilir
- ✅ **Vektör Database** - ChromaDB ile hızlı arama
- ✅ **PDF İşleme** - Kitapları otomatik olarak işler
- ✅ **Streaming Desteği** - Gerçek zamanlı metin üretimi
- ✅ **FastAPI** - Hızlı ve modern API
- ✅ **CORS Desteği** - Tüm origin'lere açık
- ✅ **Otomatik Dokümantasyon** - Swagger UI ve ReDoc

## Kurulum


### 1. Gerekli Paketleri Yükleyin

```bash
pip install -r requirements.txt
```

Bu komut aşağıdaki paketleri yükleyecek:
- FastAPI, Uvicorn (Web framework)
- fal-client (Fal.ai entegrasyonu)
- LangChain, ChromaDB (RAG sistemi)
- Sentence Transformers (Embedding modelleri)
- PyPDF2 (PDF işleme)
- python-dotenv (Ortam değişkenleri)

### 2. Ortam Değişkenlerini Ayarlayın

`.env` dosyası oluşturun veya ortam değişkenlerini ayarlayın:

```bash
# .env dosyası
FAL_KEY=your_fal_api_key_here
```

**Fal.ai API anahtarını almak için:**
1. [Fal.ai](https://fal.ai)'ya kayıt olun
2. Dashboard'dan API anahtarınızı alın
3. `.env` dosyasına `FAL_KEY=your_key_here` şeklinde ekleyin

#### Windows Ortam Değişkeni Ayarlama:
```bash
# CMD
set FAL_KEY=your_api_key_here

# PowerShell
$env:FAL_KEY="your_api_key_here"
```

### 3. Kitapları Hazırlayın

`books/` klasörüne 9-10-11-12. sınıf kitaplarınızı PDF formatında koyun:
```
books/
├── matematik_9_sinif.pdf
├── fizik_10_sinif.pdf
├── kimya_11_sinif.pdf
└── biyoloji_12_sinif.pdf
```

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
- API ve RAG sistemi durumu hakkında bilgi verir

### 2. Sağlık Kontrolü
- **GET** `/health`
- Sunucunun durumunu kontrol eder

### 3. RAG Sistemi Durumu
- **GET** `/rag-status`
- RAG sisteminin aktif olup olmadığını ve kaç document yüklendiğini gösterir

### 4. Kitap Yükleme
- **POST** `/load-books`
- Books klasöründeki kitapları yükler ve vektör database'ine ekler
- **Optimizasyon**: Kitaplar zaten yüklüyse tekrar yüklenmez (performans için)
- Eğer kitapları yeniden yüklemek istiyorsanız `chroma_db` klasörünü silin

### 5. Ana Metin Üretme (RAG + Tools ile)
- **POST** `/generate`
- Tools/fonksiyon desteği ile metin üretir

**İstek Formatı (RAG ile):**
```json
{
    "prompt": "9. sınıf matematik fonksiyonlar hakkında bilgi ver",
    "max_tokens": 1000,
    "temperature": 0.7,
    "tools": null,
    "system_prompt": null
}
```

**İstek Formatı (Sadece Tools olmadan):**
```json
{
    "prompt": "Normal bir soru sor",
    "max_tokens": 500,
    "temperature": 0.8
}
```

**Yanıt Formatı:**
```json
{
    "generated_text": "Üretilen metin burada olacak...",
    "model": "google/gemini-2.5-flash",
    "gateway": "fal-ai/any-llm",
    "tools_used": true,
    "message_history": [...],
    "total_iterations": 2
}
```

### 6. Streaming Metin Üretme
- **POST** `/generate/stream`
- Server-Sent Events ile gerçek zamanlı metin üretir

### 7. Model Bilgisi
- **GET** `/models`
- Kullanılan model hakkında bilgi verir

## Kullanım Örnekleri

### 1. Basit Metin Üretme

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
result = response.json()
print("Üretilen Metin:", result["generated_text"])
```

### 2. RAG Sistemi ile Kitap Arama

```python
import requests

url = "http://localhost:8000/generate"

# RAG sistemi ile kitap arama
data = {
    "prompt": "9. sınıf matematik fonksiyonlar hakkında detaylı bilgi ver",
    "max_tokens": 1000,
    "temperature": 0.7
    # tools ve system_prompt varsayılan olarak kullanılacak
}

response = requests.post(url, json=data)
result = response.json()

print("Tools kullanıldı:", result["tools_used"])
print("Üretilen Metin:")
print(result["generated_text"])
```

### 3. Kitap Yükleme

```python
import requests

# Kitapları yükle
response = requests.post("http://localhost:8000/load-books")
print(response.json())

# RAG durumu kontrol et
response = requests.get("http://localhost:8000/rag-status")
print(response.json())
```

### 4. cURL ile RAG Sistemi Kullanımı

```bash
# Basit RAG sorgusu
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "10. sınıf fizik yasaları hakkında özet bilgi ver",
       "max_tokens": 800,
       "temperature": 0.6
     }'

# Streaming ile RAG
curl -X POST "http://localhost:8000/generate/stream" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "11. sınıf kimya konuları nelerdir?",
       "max_tokens": 500,
       "temperature": 0.7
     }'
```

### 5. Test Dosyasını Çalıştırma

```bash
python example_usage.py
```

Bu dosya tüm özellikleri test eder ve örnek kullanım gösterir.

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

## RAG Sistemi Nasıl Çalışır?

1. **İlk Başlatma**: `books/` klasöründeki PDF'ler okunur ve metin parçalara bölünür
2. **Vektör Oluşturma**: Her metin parçası için embedding vektörü oluşturulur
3. **ChromaDB'ye Kaydetme**: Vektörler ve metinler ChromaDB'ye kaydedilir
4. **Sonraki Başlatmalar**: Kitaplar zaten yüklüyse tekrar yüklenmez (performans için)
5. **Arama**: Kullanıcı sorgusu için benzerlik araması yapılır
6. **Sonuçları Model'e Gönder**: Arama sonuçları model'e context olarak verilir
7. **Akıllı Yanıt**: Model, context'i kullanarak doğru ve detaylı yanıt üretir

### 🚀 Performans Optimizasyonu
- **Persistent Storage**: ChromaDB disk üzerinde kalıcı olarak saklanır
- **Akıllı Yükleme**: Kitaplar yüklendiyse tekrar yüklenmez
- **Hızlı Arama**: Vektör benzerlik araması çok hızlıdır
- **Yeniden Yükleme**: `chroma_db` klasörünü silerek kitapları yeniden yükleyebilirsiniz

### Desteklenen Dosya Formatları
- ✅ PDF dosyaları
- ✅ Metin tabanlı içerikler
- ✅ 9-10-11-12. sınıf müfredatına uygun kitaplar

## Sorun Giderme

### Yaygın Hatalar

1. **FAL_KEY hatası**: Fal.ai API anahtarını doğru ayarladığınızdan emin olun
2. **RAG sistemi başlatılamadı**: Gerekli paketlerin yüklü olduğundan emin olun
3. **Kitap yüklenemedi**: PDF dosyalarının `books/` klasöründe olduğundan emin olun
4. **500 Internal Server Error**: Prompt'unuzu kontrol edin, boş olmamalı

### Debug Modu

Daha detaylı loglama için:

```bash
python fal-api.py
```

### RAG Sistemi Sorunları

```bash
# RAG durumunu kontrol et
curl http://localhost:8000/rag-status

# Kitapları yeniden yükle
curl -X POST http://localhost:8000/load-books

# ChromaDB'yi sıfırla (chroma_db klasörünü silin)
rm -rf chroma_db
```

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## Lisans

Bu proje eğitim amaçlı hazırlanmıştır. Apache License 2.0 altında lisanslanmıştır.
