import fal_client
from typing import Dict, Any
import json
import re
from tools.system_prompt import QUERY_PLANNER_SYSTEM_PROMPT
from tools.subject_normalizer import normalize_subject_name, validate_subject
from tools.resilience_utils import resilient_client, create_fallback_response
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "google/gemini-2.5-flash"
FAL_MODEL_GATEWAY = "fal-ai/any-llm"
 
async def get_search_plan(user_prompt: str) -> Dict[str, Any]:
    """Kullanıcı sorgusundan arama planı oluşturur"""
    logger.info(f"[PLANNER] Sorgu analizi: '{user_prompt}'")
    
    try:
        # Fallback response for search plan
        fallback_response = {"output": '{"query": "' + user_prompt + '", "filters": {}}'}

        result = await resilient_client.run_async_with_resilience(
            FAL_MODEL_GATEWAY,
            arguments={
                "model": MODEL_NAME,
                "prompt": user_prompt,
                "system_prompt": QUERY_PLANNER_SYSTEM_PROMPT,
                "max_tokens": 150,
                "temperature": 0.1,
            },
            fallback_response=fallback_response,
            operation_type="search_plan"
        )
        
        response_text = result.get("output", "{}").strip()
        logger.info(f"[PLANNER] Model yanıtı: {response_text}")
        
        # JSON'u çıkar
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        
        if not json_match:
            logger.warning("[PLANNER] JSON bulunamadı, varsayılan plan kullanılıyor")
            return {"query": user_prompt, "filters": {}}
        
        clean_json = json_match.group(0)
        plan = json.loads(clean_json)
        
        # Plan doğrulama ve normalizasyon
        if "query" not in plan:
            plan["query"] = user_prompt
        if "filters" not in plan:
            plan["filters"] = {}
        
        # Ders adını normalize et
        if "ders" in plan["filters"] and plan["filters"]["ders"]:
            original_subject = plan["filters"]["ders"]
            normalized_subject = normalize_subject_name(original_subject)
            
            if normalized_subject:
                plan["filters"]["ders"] = normalized_subject
                if original_subject != normalized_subject:
                    logger.info(f"[PLANNER] Ders adı normalize edildi: '{original_subject}' -> '{normalized_subject}'")
            else:
                logger.warning(f"[PLANNER] ⚠️  Geçersiz ders adı kaldırıldı: '{original_subject}'")
                del plan["filters"]["ders"]
        
        # Sınıf doğrulaması
        if "sinif" in plan["filters"]:
            try:
                sinif = int(plan["filters"]["sinif"])
                if sinif not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                    logger.warning(f"[PLANNER] ⚠️  Geçersiz sınıf kaldırıldı: {sinif}")
                    del plan["filters"]["sinif"]
                else:
                    plan["filters"]["sinif"] = sinif
            except (ValueError, TypeError):
                logger.warning(f"[PLANNER] ⚠️  Geçersiz sınıf formatı kaldırıldı: {plan['filters']['sinif']}")
                del plan["filters"]["sinif"]
        
        # Boş filtreleri temizle
        plan["filters"] = {k: v for k, v in plan["filters"].items() 
                          if v is not None and str(v).strip() != ""}
        
        logger.info(f"[PLANNER] Final plan: {plan}")
        return plan
        
    except Exception as e:
        logger.error(f"[PLANNER] Hata: {e}")
        return {"query": user_prompt, "filters": {}}
