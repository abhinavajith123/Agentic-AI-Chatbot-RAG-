from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse
from app.rag.graph import rag_app

app = FastAPI(title="Agentic AI RAG Chatbot")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = rag_app.invoke({"question": req.question})
    
    # Return only the final LLM answer
    return {
        "answer": result["answer"],
        "context": [],           
        "confidence": result["confidence"]
    }
