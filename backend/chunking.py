from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from config import settings

class DocumentChunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(self, 
                      text: str, 
                      source: str,
                      title: str,
                      section: str = None) -> List[Dict]:
        """
        Chunk document and preserve metadata including content
        """
        chunks = self.splitter.split_text(text)
        chunked_docs = []
        
        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                "id": f"{source}_{idx}",
                "content": chunk,
                "metadata": {
                    "content": chunk,  # Store the actual chunk text
                    "source": source,
                    "title": title,
                    "section": section or "N/A",
                    "position": idx,
                    "chunk_size": len(chunk)
                }
            })
        
        return chunked_docs

chunker = DocumentChunker()