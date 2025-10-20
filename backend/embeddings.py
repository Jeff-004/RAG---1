import cohere
import os
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()

class EmbeddingModel:
    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")
        self.client = cohere.Client(api_key=api_key)
        self.model = "embed-english-light-v3.0"
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string"""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_document"
            )
            return response.embeddings[0]
        except Exception as e:
            print(f"Error embedding text: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query (slightly different from document embedding)"""
        try:
            response = self.client.embed(
                texts=[query],
                model=self.model,
                input_type="search_query"
            )
            return response.embeddings[0]
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """Embed multiple texts at once (more efficient)"""
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type=input_type
            )
            return response.embeddings
        except Exception as e:
            print(f"Error embedding batch: {str(e)}")
            raise

# Initialize the embedding model
embedding_model = EmbeddingModel()