from abc import ABC, abstractmethod
from config import settings
from pinecone import Pinecone
from typing import List, Dict

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, documents: List[Dict]) -> None:
        pass
    
    @abstractmethod
    def query(self, query_vector: list, top_k: int = 5) -> List[Dict]:
        pass

class PineconeStore(VectorStore):
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
    
    def upsert(self, documents: List[Dict]) -> None:
        vectors_to_upsert = []
        for doc in documents:
            from embeddings import embedding_model
            vector = embedding_model.embed_text(doc["content"])
            metadata = {k: v for k, v in doc["metadata"].items() if v is not None}
            vectors_to_upsert.append((
                doc["id"],
                vector,
                metadata
            ))
        self.index.upsert(vectors=vectors_to_upsert)
    
    def query(self, query_vector: list, top_k: int = 5) -> List[Dict]:
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            for match in results.matches
        ]

def get_vector_store() -> VectorStore:
    if settings.VECTOR_DB == "pinecone":
        return PineconeStore()
    # Add Weaviate/Supabase implementations similarly
    raise ValueError(f"Unknown vector DB: {settings.VECTOR_DB}")