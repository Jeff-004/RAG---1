from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import time

from retriever import retriever, reranker
from llm_handler import llm_handler
from citation_manager import citation_manager
from chunking import chunker
from embeddings import embedding_model
from vector_store import get_vector_store

app = FastAPI(title="RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class DocumentUploadRequest(BaseModel):
    content: str
    source: str
    title: str
    section: str = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[dict]
    sources: List[dict]
    execution_time: float
    token_estimate: int

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    start_time = time.time()
    
    try:
        retrieved = retriever.retrieve(request.query, top_k=10)
        reranked = reranker.rerank(request.query, retrieved, top_k=5)
        context = citation_manager.build_context(reranked)
        answer = llm_handler.generate_answer(request.query, context)
        formatted_answer, citations = citation_manager.format_citations(answer, reranked)
        
        execution_time = time.time() - start_time
        token_estimate = len(formatted_answer.split()) + len(context.split())
        
        return QueryResponse(
            answer=formatted_answer,
            citations=citations,
            sources=[
                {
                    "id": doc["id"],
                    "score": doc.get("rerank_score", doc.get("score", 0)),
                    "metadata": doc["metadata"]
                }
                for doc in reranked
            ],
            execution_time=execution_time,
            token_estimate=token_estimate
        )
    
    except Exception as e:
        import traceback
        print(f"Query error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(request: DocumentUploadRequest):
    try:
        chunks = chunker.chunk_document(
            text=request.content,
            source=request.source,
            title=request.title,
            section=request.section
        )
        
        vector_store = get_vector_store()
        
        for chunk in chunks:
            chunk["embedding"] = embedding_model.embed_text(chunk["content"]).tolist()
        
        vector_store.upsert(chunks)
        
        return {
            "status": "success",
            "chunks_created": len(chunks),
            "source": request.source
        }
    except Exception as e:
        import traceback
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}