"""
Quiz Generation Sistemi
fal_client kullanarak quiz soruları oluşturur
"""

import fal_client
import json
import re
import traceback
from datetime import datetime
from typing import List, Dict, Any
from tools.classes import QuizRequest, QuizResponse
from tools.quiz_prompts import get_quiz_system_prompt, build_quiz_prompt
from tools.subject_normalizer import normalize_subject_name, validate_subject

MODEL_NAME = "google/gemini-2.5-flash"
FAL_MODEL_GATEWAY = "fal-ai/any-llm"

async def generate_quiz(request: QuizRequest) -> QuizResponse:
    """
    Quiz soruları oluştur
    
    Args:
        request: Quiz parametreleri
        
    Returns:
        QuizResponse: Oluşturulan quiz
    """
    try:
        print(f"[QUIZ] Quiz generation başlatılıyor: {request.soru_sayisi} {request.soru_tipi} soru")
        
        # 1. Input validation - Pydantic otomatik yapar, bu adım gereksiz
        # Eğer request buraya kadar geldiyse zaten valid demektir
        
        # 2. Subject normalization
        normalized_subject = normalize_subject_name(request.ders)
        if not normalized_subject:
            print(f"[QUIZ] ⚠️  Ders adı normalize edilemedi: '{request.ders}', orijinal kullanılıyor")
            normalized_subject = request.ders
        else:
            print(f"[QUIZ] Ders adı normalize edildi: '{request.ders}' -> '{normalized_subject}'")
        
        # 3. System prompt ve user prompt hazırla
        system_prompt = get_quiz_system_prompt(request.soru_tipi)
        user_prompt = build_quiz_prompt(
            sinif=request.sinif,
            ders=normalized_subject, 
            konu=request.konu,
            soru_sayisi=request.soru_sayisi,
            zorluk=request.zorluk
        )
        
        print(f"[QUIZ] Prompt hazırlandı - Sınıf: {request.sinif}, Ders: {normalized_subject}, Konu: {request.konu}")
        
        # 4. Model'i çağır
        result = await fal_client.run_async(
            FAL_MODEL_GATEWAY,
            arguments={
                "model": MODEL_NAME,
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "max_tokens": min(3000, request.soru_sayisi * 200),  # Soru sayısına göre token limiti
                "temperature": 0.3,  # Tutarlı sonuçlar için düşük temperature
            },
        )
        
        response_text = result.get("output", "[]").strip()
        print(f"[QUIZ] Model yanıtı alındı: {len(response_text)} karakter")
        
        # 5. JSON parse et
        questions_data = _parse_quiz_response(response_text, request.soru_tipi)
        
        if not questions_data:
            raise ValueError("Model'den geçerli soru formatı alınamadı")
        
        # 6. Response oluştur
        quiz_response = QuizResponse(
            quiz_info={
                "sinif": request.sinif,
                "ders": normalized_subject,
                "original_ders": request.ders,
                "konu": request.konu,
                "zorluk": request.zorluk,
                "dil": request.dil,
                "generated_by": "AI Quiz Generator",
                "model": MODEL_NAME
            },
            sorular=questions_data,
            total_soru=len(questions_data),
            soru_tipi=request.soru_tipi,
            created_at=datetime.now().isoformat()
        )
        
        print(f"[QUIZ] ✅ Quiz başarıyla oluşturuldu: {len(questions_data)} soru")
        return quiz_response
        
    except Exception as e:
        print(f"[QUIZ ERROR] Quiz oluşturma hatası: {str(e)}")
        traceback.print_exc()
        raise


# _validate_quiz_request fonksiyonu kaldırıldı - Pydantic otomatik validation yapıyor


def _parse_quiz_response(response_text: str, question_type: str) -> List[Dict[str, Any]]:
    """Model yanıtından JSON soruları parse et"""
    try:
        # JSON array'i bul
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        
        if not json_match:
            print("[QUIZ] JSON array bulunamadı, response'un tamamını parse etmeye çalışılıyor...")
            # Tüm response'u parse etmeye çalış
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                # Tek soru varsa array'e çevir
                response_text = f"[{json_match.group(0)}]"
            else:
                raise ValueError("JSON formatı bulunamadı")
        else:
            response_text = json_match.group(0)
        
        # JSON parse et
        questions_data = json.loads(response_text)
        
        # List olup olmadığını kontrol et
        if not isinstance(questions_data, list):
            if isinstance(questions_data, dict):
                questions_data = [questions_data]
            else:
                raise ValueError("Geçersiz JSON formatı")
        
        # Her soruyu validate et
        validated_questions = []
        for i, question in enumerate(questions_data):
            if _validate_question_format(question, question_type):
                validated_questions.append(question)
                print(f"[QUIZ] ✓ Soru {i+1} geçerli")
            else:
                print(f"[QUIZ] ✗ Soru {i+1} geçersiz format, atlanıyor")
        
        return validated_questions
        
    except json.JSONDecodeError as e:
        print(f"[QUIZ] JSON parse hatası: {e}")
        print(f"[QUIZ] Response text: {response_text[:500]}...")
        return []
    except Exception as e:
        print(f"[QUIZ] Parse hatası: {e}")
        return []


def _validate_question_format(question: Dict[str, Any], question_type: str) -> bool:
    """Tek sorunun formatını validate et"""
    try:
        if question_type == "coktan_secmeli":
            required_fields = ["soru", "a", "b", "c", "d", "cevap", "aciklama"]
            for field in required_fields:
                if field not in question or not question[field]:
                    print(f"[QUIZ] Eksik field: {field}")
                    return False
            
            # Cevap şıkkı kontrolü
            if question["cevap"].lower() not in ["a", "b", "c", "d"]:
                print(f"[QUIZ] Geçersiz cevap şıkkı: {question['cevap']}")
                return False
                
        elif question_type == "acik_uclu":
            required_fields = ["soru", "cevap", "aciklama"]
            for field in required_fields:
                if field not in question or not question[field]:
                    print(f"[QUIZ] Eksik field: {field}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"[QUIZ] Validation error: {e}")
        return False

