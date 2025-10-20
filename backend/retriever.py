from vector_store import get_vector_store
from embeddings import embedding_model
from config import settings
from typing import List, Dict
import cohere

class Retriever:
    def __init__(self):
        self.vector_store = get_vector_store()
    
    def retrieve(self, query: str, top_k: int = settings.TOP_K) -> List[Dict]:
        """Retrieve top-k documents"""
        query_vector = embedding_model.embed_text(query)
        results = self.vector_store.query(query_vector, top_k=top_k)
        return results

class Reranker:
    def __init__(self):
        self.co = cohere.Client(api_key=settings.COHERE_API_KEY)
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank documents using Cohere"""
        if not documents:
            return []
        
        # Extract actual chunk content for reranking
        texts = []
        for doc in documents:
            # Try to get content from metadata first, then from doc itself
            content = doc["metadata"].get("content", "")
            if not content:
                content = doc.get("content", "")
            texts.append(content if content else doc["metadata"].get("source", ""))
        
        try:
            results = self.co.rerank(
                query=query,
                documents=texts,
                top_n=min(top_k, len(documents)),
                model="rerank-english-v3.0"
            )
            
            reranked = []
            for result in results:
                doc_copy = documents[result.index].copy()
                doc_copy["rerank_score"] = result.relevance_score
                reranked.append(doc_copy)
            
            print(f"Reranking successful: {len(reranked)} documents reranked")
            return reranked
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            print(f"Returning original {top_k} documents without reranking")
            return documents[:top_k]

retriever = Retriever()
reranker = Reranker()