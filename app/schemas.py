from pydantic import BaseModel
from typing import List, Optional

class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

class AskResponse(BaseModel):
    answer: str
    source: List[str]