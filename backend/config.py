from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # Vector DB
    VECTOR_DB: Literal["pinecone", "weaviate", "supabase"] = "pinecone"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "rag-index"
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    
    # Retrieval
    TOP_K: int = 5
    RERANKER_MODEL: str = "cohere"
    COHERE_API_KEY: str = ""
    
    # LLM - Groq 
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.1-8b-instant"
    
    class Config:
        env_file = ".env"

settings = Settings()