"""
models.py — Pydantic v2 request/response models for all API endpoints.

Every payload flowing through the API is validated by one of these models,
ensuring strict typing, descriptive error messages, and self-documenting
OpenAPI schemas.
"""

from pydantic import BaseModel, Field


# ── Upload ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response returned after successfully uploading and ingesting a document."""

    status: str = Field(..., description="'success' or 'error'")
    filename: str = Field(..., description="Name of the uploaded file")
    chunks_stored: int = Field(
        ..., description="Number of chunks stored in ChromaDB"
    )
    message: str = Field(..., description="Human-readable status message")


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the /chat_query endpoint."""

    query: str = Field(
        ..., min_length=1, description="Natural-language question about the documents"
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of chunks to retrieve"
    )


class CitationModel(BaseModel):
    """Structured citation reference pointing back to a source document."""

    source_filename: str = Field(..., description="Original PDF filename")
    page_number: int = Field(..., description="Page number in the source PDF")
    section_header: str = Field(..., description="Nearest section heading")


class ChatResponse(BaseModel):
    """Response returned by the CRAG chat pipeline."""

    query: str = Field(..., description="The original user query")
    answer: str = Field(..., description="Generated answer grounded in context")
    citations: list[str] = Field(
        default_factory=list, description="Inline citation strings"
    )
    chunks_used: int = Field(
        ..., description="Number of context chunks used for generation"
    )
    loop_count: int = Field(
        ..., description="Number of rewrite loops executed (0 = first retrieval was sufficient)"
    )


# ── Clause Extraction ───────────────────────────────────────────────────────

class ClauseItem(BaseModel):
    """A single extracted clause with risk assessment."""

    clause_type: str = Field(
        ..., description="Category of the clause (e.g., Termination, Liability)"
    )
    clause_text: str = Field(..., description="Full text of the extracted clause")
    page_number: int = Field(
        ..., description="Page where the clause was found"
    )
    risk_level: str = Field(
        ..., description="Risk assessment: 'low', 'medium', or 'high'"
    )


class ClauseExtractionRequest(BaseModel):
    """Request body for the /extract_clauses endpoint."""

    query: str = Field(
        default="Extract all important clauses",
        description="Extraction instruction (can be customized)",
    )


class ClauseExtractionResponse(BaseModel):
    """Response containing all extracted clauses from uploaded documents."""

    filename: str = Field(
        ..., description="Source document filename (or 'multiple')"
    )
    clauses: list[ClauseItem] = Field(
        default_factory=list, description="List of extracted clauses"
    )
    total_found: int = Field(
        ..., description="Total number of clauses identified"
    )
