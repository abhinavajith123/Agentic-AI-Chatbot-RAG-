from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    question: str

class ContextChunk(BaseModel):
    text: str
    page: int
    score: float

class ChatResponse(BaseModel):
    answer: str
    context: List[ContextChunk]
    confidence: float
