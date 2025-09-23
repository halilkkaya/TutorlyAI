from pydantic import BaseModel
from typing import Optional, List, Dict, Any



class TextGenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    tools: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None


# RAG sistemi için yanıt modeli
class ToolResponse(BaseModel):
    tool_call_id: str
    result: str
