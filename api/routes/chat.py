"""
chat.py — /chat_query endpoint.

Accepts a user question, runs it through the full Corrective RAG (CRAG)
LangGraph pipeline, and returns a grounded answer with citations.
"""

import logging

from fastapi import APIRouter, HTTPException

from api.models import ChatRequest, ChatResponse
from core_logic.graph import run_crag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/chat_query",
    response_model=ChatResponse,
    summary="Ask a question about uploaded documents",
    description=(
        "Runs the Corrective RAG pipeline: retrieve → grade → "
        "(rewrite if needed) → generate. Returns a cited answer."
    ),
)
async def chat_query(request: ChatRequest) -> ChatResponse:
    """
    Process a user query through the CRAG pipeline.

    Args:
        request: ChatRequest with query string and optional top_k.

    Returns:
        ChatResponse with answer, citations, chunks_used, and loop_count.

    Raises:
        HTTPException 400: If the query is empty.
        HTTPException 500: If the CRAG pipeline fails.
    """
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        logger.info("chat_query — processing: '%s' (top_k=%d)", query[:80], request.top_k)

        # Run the full CRAG pipeline
        result = run_crag_pipeline(query)

        return ChatResponse(
            query=result["query"],
            answer=result["answer"],
            citations=result.get("citations", []),
            chunks_used=result.get("chunks_used", 0),
            loop_count=result.get("loop_count", 0),
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Configuration error in chat pipeline")
        raise HTTPException(status_code=500, detail=str(exc))
    except RuntimeError as exc:
        logger.exception("CRAG pipeline runtime error")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in chat_query")
        raise HTTPException(
            status_code=500,
            detail=f"Chat pipeline error: {exc}",
        )
