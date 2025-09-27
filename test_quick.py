#!/usr/bin/env python3
"""
TutorlyAI Quick Test Script
Hızlı test için basit script
"""

import requests
import json

# API Configuration
API_BASE_URL = "http://81.214.137.5:8000"

def quick_test():
    """Hızlı test"""
    print("🚀 Hızlı Test Başlatılıyor...")

    # Test data
    data = {
        "prompt": "Bir matematik dersi için basit toplama işlemi gösteren renkli görsel",
        "workflow_id": "workflows/halillllibrahim58/teach-img-model",
        "max_tokens": 1000,
        "temperature": 0.7
    }

    try:
        print("📤 Request gönderiliyor...")
        response = requests.post(
            f"{API_BASE_URL}/generate/image",
            json=data,
            timeout=60
        )

        print(f"📊 Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Başarılı!")
            print(f"🖼️  Ana Image URL: {result.get('image_url', 'N/A')}")
            print(f"📊 Toplam Görsel: {result.get('total_images', 'N/A')}")
            print(f"🎯 Success: {result.get('success', 'N/A')}")

            # Tüm görselleri göster
            all_images = result.get('all_images', [])
            if all_images:
                print(f"\n🖼️  Tüm Görseller ({len(all_images)} adet):")
                for i, img_url in enumerate(all_images, 1):
                    print(f"  {i}. {img_url}")
        else:
            print(f"❌ Hata: {response.text}")

    except Exception as e:
        print(f"💥 Hata: {str(e)}")

if __name__ == "__main__":
    quick_test()
