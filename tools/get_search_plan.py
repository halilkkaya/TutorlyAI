import fal_client
from typing import Dict, Any
import json
import re
from tools.system_prompt import QUERY_PLANNER_SYSTEM_PROMPT

MODEL_NAME = "google/gemini-2.5-flash"
FAL_MODEL_GATEWAY = "fal-ai/any-llm"
 
async def get_search_plan(user_prompt: str) -> Dict[str, Any]:
    """Kullanıcı sorgusundan arama planı oluşturur"""
    print(f"[PLANNER] Sorgu analizi: '{user_prompt}'")
    
    try:
        result = await fal_client.run_async(
            FAL_MODEL_GATEWAY,
            arguments={
                "model": MODEL_NAME,
                "prompt": user_prompt,
                "system_prompt": QUERY_PLANNER_SYSTEM_PROMPT,
                "max_tokens": 150,
                "temperature": 0.1,
            },
        )
        
        response_text = result.get("output", "{}").strip()
        print(f"[PLANNER] Model yanıtı: {response_text}")
        
        # JSON'u çıkar
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        
        if not json_match:
            print("[PLANNER] JSON bulunamadı, varsayılan plan kullanılıyor")
            return {"query": user_prompt, "filters": {}}
        
        clean_json = json_match.group(0)
        plan = json.loads(clean_json)
        
        # Plan doğrulama
        if "query" not in plan:
            plan["query"] = user_prompt
        if "filters" not in plan:
            plan["filters"] = {}
        
        # Boş filtreleri temizle
        plan["filters"] = {k: v for k, v in plan["filters"].items() 
                          if v is not None and str(v).strip() != ""}
        
        print(f"[PLANNER] Final plan: {plan}")
        return plan
        
    except Exception as e:
        print(f"[PLANNER] Hata: {e}")
        return {"query": user_prompt, "filters": {}}
