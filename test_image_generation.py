#!/usr/bin/env python3
"""
TutorlyAI Image Generation Endpoint Test Script
Bu script generate/image endpoint'ini test eder
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
API_BASE_URL = "http://81.214.137.5:8000"
ENDPOINT = "/generate/image"

def test_image_generation():
    """Görsel üretimi endpoint'ini test eder"""

    print("=" * 60)
    print("TutorlyAI Image Generation Test")
    print("=" * 60)
    print(f"API URL: {API_BASE_URL}{ENDPOINT}")
    print(f"Test Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Test data
    test_cases = [
        {
            "name": "Basit Matematik Görseli",
            "data": {
                "prompt": "Bir matematik dersi için basit toplama işlemi gösteren renkli görsel",
                "workflow_id": "workflows/halillllibrahim58/teach-img-model",
                "max_tokens": 1000,
                "temperature": 0.7
            }
        },
        {
            "name": "Fen Bilgisi Görseli",
            "data": {
                "prompt": "Güneş sistemi ve gezegenleri gösteren eğitici görsel",
                "workflow_id": "workflows/halillllibrahim58/teach-img-model",
                "max_tokens": 1000,
                "temperature": 0.7
            }
        },
        {
            "name": "Türkçe Dersi Görseli",
            "data": {
                "prompt": "Türkçe alfabesi ve harfleri gösteren eğitici görsel",
                "workflow_id": "workflows/halillllibrahim58/teach-img-model",
                "max_tokens": 1000,
                "temperature": 0.7
            }
        }
    ]

    # Her test case'i çalıştır
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Test: {test_case['name']}")
        print("-" * 40)

        try:
            # Request gönder
            print("📤 Request gönderiliyor...")
            start_time = time.time()

            response = requests.post(
                f"{API_BASE_URL}{ENDPOINT}",
                json=test_case['data'],
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                timeout=60  # 60 saniye timeout
            )

            end_time = time.time()
            duration = end_time - start_time

            print(f"⏱️  Süre: {duration:.2f} saniye")
            print(f"📊 Status Code: {response.status_code}")

            # Response'u kontrol et
            if response.status_code == 200:
                result = response.json()

                print("✅ Başarılı!")
                print(f"🎯 Success: {result.get('success', 'N/A')}")
                print(f"🖼️  Ana Image URL: {result.get('image_url', 'N/A')}")
                print(f"📊 Toplam Görsel: {result.get('total_images', 'N/A')}")
                print(f"🔄 Workflow ID: {result.get('workflow_id', 'N/A')}")
                print(f"📝 Prompt: {result.get('prompt', 'N/A')[:50]}...")
                print(f"⏰ Generated At: {result.get('generated_at', 'N/A')}")

                if result.get('error_message'):
                    print(f"⚠️  Error Message: {result['error_message']}")

                # Tüm görselleri göster
                all_images = result.get('all_images', [])
                if all_images:
                    print(f"\n🖼️  Tüm Görseller ({len(all_images)} adet):")
                    for i, img_url in enumerate(all_images, 1):
                        print(f"  {i}. {img_url}")
                elif result.get('image_url'):
                    print(f"\n🔗 Ana Görsel: {result['image_url']}")

            else:
                print(f"❌ Hata! Status Code: {response.status_code}")
                print(f"📄 Response: {response.text}")

        except requests.exceptions.Timeout:
            print("⏰ Timeout! Request çok uzun sürdü")
        except requests.exceptions.ConnectionError:
            print("🔌 Bağlantı Hatası! Sunucuya ulaşılamıyor")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Hatası: {str(e)}")
        except json.JSONDecodeError:
            print("📄 JSON Parse Hatası! Response geçersiz JSON")
        except Exception as e:
            print(f"💥 Beklenmeyen Hata: {str(e)}")

    print("\n" + "=" * 60)
    print("Test Tamamlandı!")
    print("=" * 60)

def test_endpoint_info():
    """Endpoint bilgilerini test eder"""
    print("\n🔍 Endpoint Bilgileri Test Ediliyor...")

    try:
        response = requests.get(f"{API_BASE_URL}/generate/image/info", timeout=10)

        if response.status_code == 200:
            info = response.json()
            print("✅ Endpoint bilgileri alındı:")
            print(json.dumps(info, indent=2, ensure_ascii=False))
        else:
            print(f"❌ Endpoint bilgileri alınamadı: {response.status_code}")

    except Exception as e:
        print(f"❌ Endpoint bilgileri test hatası: {str(e)}")

def test_health_check():
    """Sağlık kontrolü yapar"""
    print("\n🏥 Sağlık Kontrolü...")

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)

        if response.status_code == 200:
            health = response.json()
            print("✅ API sağlıklı:")
            print(json.dumps(health, indent=2, ensure_ascii=False))
        else:
            print(f"❌ API sağlık kontrolü başarısız: {response.status_code}")

    except Exception as e:
        print(f"❌ Sağlık kontrolü hatası: {str(e)}")

if __name__ == "__main__":
    print("🚀 TutorlyAI Image Generation Test Script Başlatılıyor...")

    # Önce sağlık kontrolü
    test_health_check()

    # Endpoint bilgilerini al
    test_endpoint_info()

    # Ana test
    test_image_generation()

    print("\n🎉 Tüm testler tamamlandı!")
    print("📋 Sonuçları kontrol edin ve görsel URL'lerini tarayıcıda açın.")
