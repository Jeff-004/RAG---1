from typing import List, Dict, Tuple

class CitationManager:
    def format_citations(self, 
                         answer: str, 
                         sources: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Extract and format citations in answer
        Returns: (formatted_answer, citation_list)
        """
        citations = []
        
        for idx, source in enumerate(sources, 1):
            citations.append({
                "index": idx,
                "source": source["metadata"].get("source", "Unknown"),
                "title": source["metadata"].get("title", "Unknown")
            })
        
        return answer, citations
    
    def build_context(self, documents: List[Dict]) -> str:
        """Build context string from retrieved documents"""
        context = ""
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "Unknown")
            content = metadata.get("content", "")
            context += f"\n[{i}] {title}:\n{content}\n"
        
        return context

citation_manager = CitationManager()