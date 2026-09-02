RAG System - Retrieval Augmented Generation

A complete end-to-end retrieval-augmented generation system with vector database storage, semantic search, reranking, and LLM-powered answers with citations.

Architecture

Documents → Chunking (1000 tokens) → Embeddings (all-MiniLM-L6-v2)
                                           ↓
                                    Pinecone Vector DB
                                           ↓
Query → Embedding → Semantic Search (Top-5) → Cohere Reranking 
                                           ↓
                    Context Building → Groq LLM → Answer + Citations

Usage
Upload Documents

    Paste document text
    Enter source name (e.g., "doc1")
    Enter document title
    Click "Upload"
    System chunks the document and stores in Pinecone

Query

    Type a question in the "Ask a Question" box
    Click "Search"
    View:
        Answer: LLM-generated response with citations
        Citations: [1], [2], etc. with source and title
        Retrieved Sources: Chunks ranked by relevance score

Chunking Configuration

CHUNK_SIZE=1000        # Tokens per chunk
CHUNK_OVERLAP=100      # Tokens overlapping between chunks

Reduces storage and improves query speed. Adjust based on document type:

    Long documents: 1000-2000 tokens
    Short documents: 500-800 tokens

Retrieval & Reranking Process
Step 1: Semantic Search

    Query embedded using all-MiniLM-L6-v2
    Top-5 chunks retrieved from Pinecone using cosine similarity
    Fast (~50-100ms)

Step 2: Reranking

    Cohere reranker reorders top-5 by relevance
    Uses deeper semantic understanding than embeddings
    Improves precision significantly
    Scores 0.0-1.0 displayed as percentages

Step 3: LLM Generation

    Top-5 reranked chunks passed as context
    Groq generates grounded answer with citations [1-5]
    Temperature 0.3 (factual, not creative)
    Max tokens: 1000


Built with: FastAPI, Pinecone, Groq, Cohere
