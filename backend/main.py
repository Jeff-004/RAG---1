from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import time

app = FastAPI(title="RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy-loaded modules
_retriever = None
_reranker = None
_llm_handler = None
_citation_manager = None
_chunker = None
_embedding_model = None
_vector_store = None

def get_retriever():
    global _retriever
    if _retriever is None:
        from retriever import retriever
        _retriever = retriever
    return _retriever

def get_reranker():
    global _reranker
    if _reranker is None:
        from retriever import reranker
        _reranker = reranker
    return _reranker

def get_llm_handler():
    global _llm_handler
    if _llm_handler is None:
        from llm_handler import llm_handler
        _llm_handler = llm_handler
    return _llm_handler

def get_citation_manager():
    global _citation_manager
    if _citation_manager is None:
        from citation_manager import citation_manager
        _citation_manager = citation_manager
    return _citation_manager

def get_chunker():
    global _chunker
    if _chunker is None:
        from chunking import chunker
        _chunker = chunker
    return _chunker

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from embeddings import embedding_model
        _embedding_model = embedding_model
    return _embedding_model

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        from vector_store import get_vector_store as _get_vs
        _vector_store = _get_vs()
    return _vector_store

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
        retriever = get_retriever()
        reranker = get_reranker()
        llm_handler = get_llm_handler()
        citation_manager = get_citation_manager()
        
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
        chunker = get_chunker()
        embedding_model = get_embedding_model()
        vector_store = get_vector_store()
        
        chunks = chunker.chunk_document(
            text=request.content,
            source=request.source,
            title=request.title,
            section=request.section
        )
        
        # Batch embed all chunks at once (more efficient with Cohere API)
        texts_to_embed = [chunk["content"] for chunk in chunks]
        embeddings = embedding_model.embed_batch(texts_to_embed, input_type="search_document")
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
       
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