from pydantic import BaseModel
from typing import Optional, List, Dict, Any



class TextGenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    tools: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    
    # Hibrit arama parametreleri
    use_hybrid: Optional[bool] = True
    semantic_weight: Optional[float] = 0.7
    keyword_weight: Optional[float] = 0.3
    score_threshold: Optional[float] = 0.25
    search_k: Optional[int] = 5


# RAG sistemi için yanıt modeli
class ToolResponse(BaseModel):
    tool_call_id: str
    result: str
