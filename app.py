from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import build_graph


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="LangGraph + Pinecone grounded RAG chatbot",
    version="1.0.0",
)


graph = build_graph()


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.get("/")
def root():
    return {
        "message": "Agentic AI RAG API is running"
    }


@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "confidence_score": 0.0,
    }

    result = graph.invoke(initial_state)

    return ChatResponse(
        query=request.query,
        final_answer=result["answer"],
        retrieved_context_chunks=result["context"],
        confidence_score=result["confidence_score"],
    )