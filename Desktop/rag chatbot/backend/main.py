from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os
from rag_core import RAGPipeline

app = FastAPI(title="Mini RAG API")

# Add CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Warning: Could not initialize RAG pipeline. Error: {e}")

class QueryRequest(BaseModel):
    question: str

class ContextResult(BaseModel):
    text: str
    source: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    contexts: List[ContextResult]

@app.post("/rag/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline not initialized. Check server logs.")
        
    contexts = pipeline.retrieve(request.question, top_k=3)
    answer = pipeline.generate_answer(request.question, contexts)
    
    return QueryResponse(answer=answer, contexts=contexts)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
