"""
graph.py — LangGraph Corrective RAG (CRAG) state machine.

Orchestrates the full CRAG pipeline:
  START → retrieve → grade → (conditional) → generate / rewrite → …

Loop protection: maximum 2 query rewrites before forcing generation.
"""

import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from core_logic.retriever import retrieve_chunks
from core_logic.grader import grade_relevance
from core_logic.rewriter import rewrite_query
from core_logic.generator import generate_answer

logger = logging.getLogger(__name__)


# ── State schema ─────────────────────────────────────────────────────────────
class CRAGState(TypedDict, total=False):
    """Typed state flowing through the CRAG graph."""

    query: str
    rewritten_query: str
    retrieved_chunks: list[dict[str, Any]]
    grade: str  # "yes" or "no"
    answer: str
    citations: list[str]
    loop_count: int
    final: bool
    chunks_used: int


# ── Node functions ───────────────────────────────────────────────────────────

def retrieve_node(state: CRAGState) -> dict[str, Any]:
    """
    Retrieve relevant chunks from ChromaDB.

    Uses the rewritten query if one exists, otherwise the original query.
    """
    try:
        search_query = state.get("rewritten_query") or state["query"]
        logger.info(
            "retrieve_node — searching for: '%s'", search_query[:80]
        )
        chunks = retrieve_chunks(search_query, k=5)
        return {"retrieved_chunks": chunks}
    except Exception as exc:
        logger.exception("retrieve_node failed")
        return {"retrieved_chunks": [], "grade": "no"}


def grade_node(state: CRAGState) -> dict[str, Any]:
    """
    Grade the relevance of retrieved chunks with respect to the query.
    """
    try:
        query = state.get("rewritten_query") or state["query"]
        chunks = state.get("retrieved_chunks", [])
        grade = grade_relevance(query, chunks)
        logger.info("grade_node — relevance: %s", grade)
        return {"grade": grade}
    except Exception as exc:
        logger.exception("grade_node failed — defaulting to 'no'")
        return {"grade": "no"}


def rewrite_node(state: CRAGState) -> dict[str, Any]:
    """
    Rewrite the query to improve retrieval results.

    Increments loop_count and forces generation if the limit is reached.
    """
    try:
        loop_count = state.get("loop_count", 0) + 1

        # Safety valve: prevent infinite loops
        if loop_count >= 2:
            logger.warning(
                "rewrite_node — loop_count=%d reached limit; forcing generation.",
                loop_count,
            )
            return {
                "loop_count": loop_count,
                "grade": "yes",  # force through to generate
            }

        original = state["query"]
        rewritten = rewrite_query(original, attempt=loop_count)
        logger.info(
            "rewrite_node — attempt %d: '%s' → '%s'",
            loop_count,
            original[:50],
            rewritten[:80],
        )
        return {
            "rewritten_query": rewritten,
            "loop_count": loop_count,
        }
    except Exception as exc:
        logger.exception("rewrite_node failed")
        return {
            "loop_count": state.get("loop_count", 0) + 1,
            "grade": "yes",  # fail-open: try generating anyway
        }


def generate_node(state: CRAGState) -> dict[str, Any]:
    """
    Generate the final answer with citations from retrieved chunks.
    """
    try:
        query = state.get("rewritten_query") or state["query"]
        chunks = state.get("retrieved_chunks", [])
        result = generate_answer(query, chunks)
        logger.info(
            "generate_node — answer length: %d chars, citations: %d",
            len(result["answer"]),
            len(result["citations"]),
        )
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "chunks_used": result["chunks_used"],
            "final": True,
        }
    except Exception as exc:
        logger.exception("generate_node failed")
        return {
            "answer": (
                "An error occurred while generating the answer. "
                "Please try again or rephrase your query."
            ),
            "citations": [],
            "chunks_used": 0,
            "final": True,
        }


# ── Conditional edge ────────────────────────────────────────────────────────

def decide_after_grading(state: CRAGState) -> str:
    """
    Decide next node after grading.

    Routes to 'generate' if chunks are relevant OR the rewrite budget is
    exhausted.  Otherwise routes to 'rewrite'.
    """
    grade = state.get("grade", "no")
    loop_count = state.get("loop_count", 0)

    if grade == "yes" or loop_count >= 2:
        logger.info("Routing → generate_node (grade=%s, loops=%d)", grade, loop_count)
        return "generate"
    else:
        logger.info("Routing → rewrite_node (grade=%s, loops=%d)", grade, loop_count)
        return "rewrite"


# ── Build and compile the graph ──────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """Construct the CRAG state machine graph."""
    graph = StateGraph(CRAGState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    # Add edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph


# Compile once at module level
_compiled_graph = _build_graph().compile()


def run_crag_pipeline(query: str) -> dict[str, Any]:
    """
    Execute the full Corrective RAG pipeline for a user query.

    Args:
        query: The user's natural-language question.

    Returns:
        Dict with keys: query, answer, citations, chunks_used, loop_count,
        retrieved_chunks.

    Raises:
        RuntimeError: If the pipeline execution fails.
    """
    try:
        if not query or not query.strip():
            return {
                "query": query,
                "answer": "Please provide a valid query.",
                "citations": [],
                "chunks_used": 0,
                "loop_count": 0,
                "retrieved_chunks": [],
            }

        initial_state: CRAGState = {
            "query": query.strip(),
            "rewritten_query": "",
            "retrieved_chunks": [],
            "grade": "",
            "answer": "",
            "citations": [],
            "loop_count": 0,
            "final": False,
            "chunks_used": 0,
        }

        logger.info("Running CRAG pipeline for query: '%s'", query[:100])
        final_state = _compiled_graph.invoke(initial_state)

        result = {
            "query": query.strip(),
            "answer": final_state.get("answer", "No answer generated."),
            "citations": final_state.get("citations", []),
            "chunks_used": final_state.get("chunks_used", 0),
            "loop_count": final_state.get("loop_count", 0),
            "retrieved_chunks": final_state.get("retrieved_chunks", []),
        }

        logger.info(
            "CRAG pipeline complete — loops: %d, citations: %d",
            result["loop_count"],
            len(result["citations"]),
        )
        return result

    except Exception as exc:
        logger.exception("CRAG pipeline failed for query: %s", query[:100])
        raise RuntimeError(f"CRAG pipeline error: {exc}") from exc
