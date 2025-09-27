#!/usr/bin/env python3
"""
TutorlyAI Image Generation Stream Test Script
Bu script generate/image/stream endpoint'ini test eder
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
API_BASE_URL = "http://81.214.137.5:8000"
ENDPOINT = "/generate/image/stream"

def test_stream_image_generation():
    """Streaming görsel üretimi endpoint'ini test eder"""

    print("=" * 60)
    print("TutorlyAI Image Generation Stream Test")
    print("=" * 60)
    print(f"API URL: {API_BASE_URL}{ENDPOINT}")
    print(f"Test Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Test data
    test_data = {
        "prompt": "Bir matematik dersi için basit toplama işlemi gösteren renkli görsel",
        "workflow_id": "workflows/halillllibrahim58/teach-img-model",
        "max_tokens": 1000,
        "temperature": 0.7
    }

    print(f"📝 Test Prompt: {test_data['prompt']}")
    print(f"🔄 Workflow ID: {test_data['workflow_id']}")
    print("-" * 60)

    try:
        print("📤 Stream request başlatılıyor...")
        start_time = time.time()

        # Stream request gönder
        response = requests.post(
            f"{API_BASE_URL}{ENDPOINT}",
            json=test_data,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            },
            stream=True,  # Stream mode
            timeout=120  # 2 dakika timeout
        )

        print(f"📊 Status Code: {response.status_code}")

        if response.status_code == 200:
            print("✅ Stream bağlantısı başarılı!")
            print("📡 Stream verileri alınıyor...")
            print("-" * 60)

            event_count = 0
            image_urls = []

            # Stream verilerini oku
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith('data: '):
                        event_count += 1
                        data_str = line[6:]  # 'data: ' kısmını çıkar

                        try:
                            # JSON parse et
                            event_data = json.loads(data_str)
                            event_type = event_data.get('type', 'unknown')

                            print(f"[{event_count}] Event Type: {event_type}")

                            if event_type == 'start':
                                print(f"  🚀 Workflow: {event_data.get('workflow_id', 'N/A')}")
                                print(f"  📝 Prompt: {event_data.get('prompt', 'N/A')[:50]}...")

                            elif event_type == 'text':
                                content = event_data.get('content', '')
                                print(f"  📄 Text: {content[:100]}{'...' if len(content) > 100 else ''}")

                            elif event_type == 'image':
                                url = event_data.get('url', '')
                                image_urls.append(url)
                                print(f"  🖼️  Image URL: {url}")

                            elif event_type == 'event':
                                print(f"  📊 Event Data: {event_data.get('data', 'N/A')[:100]}...")

                            elif event_type == 'error':
                                print(f"  ❌ Error: {event_data.get('message', 'N/A')}")

                            else:
                                print(f"  ❓ Unknown Event: {event_data}")

                        except json.JSONDecodeError:
                            print(f"  ⚠️  JSON Parse Error: {data_str[:100]}...")
                        except Exception as e:
                            print(f"  💥 Event Parse Error: {str(e)}")

            end_time = time.time()
            duration = end_time - start_time

            print("-" * 60)
            print(f"⏱️  Toplam Süre: {duration:.2f} saniye")
            print(f"📊 Toplam Event: {event_count}")
            print(f"🖼️  Bulunan Görsel: {len(image_urls)}")

            if image_urls:
                print(f"\n🎯 Görsel URL'leri ({len(image_urls)} adet):")
                for i, url in enumerate(image_urls, 1):
                    print(f"  {i}. {url}")
            else:
                print("\n⚠️  Hiç görsel URL'si bulunamadı")

        else:
            print(f"❌ Stream bağlantısı başarısız! Status Code: {response.status_code}")
            print(f"📄 Response: {response.text}")

    except requests.exceptions.Timeout:
        print("⏰ Timeout! Stream çok uzun sürdü")
    except requests.exceptions.ConnectionError:
        print("🔌 Bağlantı Hatası! Sunucuya ulaşılamıyor")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Hatası: {str(e)}")
    except Exception as e:
        print(f"💥 Beklenmeyen Hata: {str(e)}")

if __name__ == "__main__":
    print("🚀 TutorlyAI Image Generation Stream Test Script Başlatılıyor...")
    test_stream_image_generation()
    print("\n🎉 Stream test tamamlandı!")
