"""
grader.py — LLM relevance grading node.

Uses Groq with the lightweight "llama-3.1-8b-instant" model to determine
whether retrieved chunks contain enough factual information to answer the
user's query.  Returns strictly "yes" or "no".
"""

import os
import logging
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

# ── Grading prompt ───────────────────────────────────────────────────────────
_GRADER_SYSTEM_PROMPT = (
    "You are a relevance grader. Given the user query and retrieved document "
    "chunks, determine strictly whether the chunks contain enough factual "
    "information to answer the query. Respond with ONLY 'yes' or 'no'. "
    "No explanation."
)


def grade_relevance(query: str, chunks: list[dict[str, Any]]) -> str:
    """
    Grade whether the retrieved chunks are relevant enough to answer the query.

    Args:
        query: The user's original or rewritten query.
        chunks: List of retrieved chunk dicts (each with "text" and "metadata").

    Returns:
        "yes" if chunks are relevant, "no" otherwise.

    Raises:
        RuntimeError: If the Groq API call fails.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

        if not chunks:
            logger.warning("No chunks provided for grading — returning 'no'.")
            return "no"

        # Build the context string from retrieved chunks
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            context_parts.append(
                f"--- Chunk {i} ---\n"
                f"Source: {meta.get('source_filename', 'N/A')}, "
                f"Page: {meta.get('page_number', 'N/A')}, "
                f"Section: {meta.get('section_header', 'N/A')}\n"
                f"{chunk.get('text', '')}\n"
            )
        context_str = "\n".join(context_parts)

        user_message = (
            f"User query: {query}\n\n"
            f"Retrieved document chunks:\n{context_str}\n\n"
            "Do these chunks contain enough information to answer the query? "
            "Respond with ONLY 'yes' or 'no'."
        )

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=5,
        )

        raw_answer = response.choices[0].message.content.strip().lower()
        grade = raw_answer if raw_answer in ("yes", "no") else "no"

        logger.info(
            "Relevance grade for query '%s': %s (raw: '%s')",
            query[:60],
            grade,
            raw_answer,
        )
        return grade

    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Relevance grading failed for query: %s", query[:80])
        raise RuntimeError(f"Grading error: {exc}") from exc
