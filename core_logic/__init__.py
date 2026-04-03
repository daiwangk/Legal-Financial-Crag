"""
Core Logic Package for Legal & Financial Document Intelligence System.

This package contains the CRAG (Corrective RAG) pipeline components:
- ingestion: Document parsing, chunking, embedding, and storage
- retriever: Semantic search over ChromaDB
- grader: LLM-based relevance grading
- rewriter: Query rewriting for improved retrieval
- generator: Answer generation with citations
- graph: LangGraph CRAG state machine orchestration
"""

# Python 3.14 + Pydantic v2 compatibility — must run before any llama imports

