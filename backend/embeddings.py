from sentence_transformers import SentenceTransformer
from config import settings
import numpy as np

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: list) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts, convert_to_numpy=True)

embedding_model = EmbeddingModel()