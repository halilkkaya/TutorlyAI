#!/usr/bin/env python3
"""
TutorlyAI RAG Sistemi Kullanım Örneği

Bu dosya, RAG sistemi ile kitap arama özelliklerinin nasıl kullanılacağını gösterir.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_rag_system():
    """RAG sistemini test eder"""
    print("=== RAG Sistemi Testi ===")

    # RAG durumunu kontrol et
    response = requests.get(f"{BASE_URL}/rag-status")
    rag_status = response.json()
    print(f"RAG Durumu: {json.dumps(rag_status, indent=2, ensure_ascii=False)}")

    # Kitapları yükle (sadece gerekirse)
    print("\nKitap yükleme kontrolü...")
    response = requests.post(f"{BASE_URL}/load-books")
    load_result = response.json()
    print(f"Yükleme Sonucu: {json.dumps(load_result, indent=2, ensure_ascii=False)}")

    if load_result.get("status") == "already_loaded":
        print("✅ Kitaplar zaten yüklü - performans optimizasyonu çalışıyor!")

def search_books_example():
    """Kitap arama örneği"""
    print("\n=== Kitap Arama Örneği ===")

    prompt = "Matematik fonksiyonlar hakkında bilgi ver"

    payload = {
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7,
        "tools": None,  # Varsayılan tools'ları kullan
        "system_prompt": None  # Varsayılan system prompt'u kullan
    }

    response = requests.post(
        f"{BASE_URL}/generate",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    result = response.json()
    print(f"Sorgu: {prompt}")
    print(f"Tools kullanıldı: {result.get('tools_used', False)}")
    print(f"Üretilen Yanıt:\n{result.get('generated_text', 'Yanıt yok')}")

def direct_search_example():
    """Doğrudan kitap arama örneği"""
    print("\n=== Doğrudan Kitap Arama ===")

    # search_books fonksiyonunu test et
    from fal_api import search_books

    result = search_books("fizik yasaları", k=3)
    print(f"Arama Sonucu:\n{result}")

def streaming_example():
    """Streaming örneği"""
    print("\n=== Streaming Örneği ===")

    prompt = "9. sınıf matematik konularını özetle"

    payload = {
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.7
    }

    response = requests.post(
        f"{BASE_URL}/generate/stream",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True
    )

    print(f"Streaming Sorgu: {prompt}")
    print("Streaming Yanıt:")
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data = json.loads(line_str[6:])
                if data.get('done'):
                    print(f"\n[Final Response] {data.get('full_response', '')}")
                    break
                else:
                    print(data.get('text', ''), end='', flush=True)

if __name__ == "__main__":
    print("TutorlyAI RAG Sistemi Örnek Kullanımı")
    print("=" * 50)

    try:
        # RAG sistemini test et
        test_rag_system()

        # Kitap arama örneği
        search_books_example()

        # Doğrudan arama örneği
        direct_search_example()

        # Streaming örneği
        streaming_example()

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        print("Lütfen API'nin çalıştığından emin olun (python fal-api.py)")
