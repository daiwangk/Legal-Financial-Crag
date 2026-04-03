"""
rewriter.py — Query rewriter node.

Uses Groq with "llama-3.1-8b-instant" to rewrite queries that failed
relevance grading.  Applies legal/financial domain optimizations such as
expanding abbreviations, adding synonyms, and restructuring compound questions.
"""

import os
import logging

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

# ── Rewriter prompt template ────────────────────────────────────────────────
_REWRITER_SYSTEM_PROMPT = (
    "You are a legal/financial search query optimizer. The original query "
    "failed to retrieve relevant document sections. Rewrite it to be more "
    "specific using:\n"
    "- Expand abbreviations and acronyms\n"
    "- Add legal/financial domain synonyms\n"
    "- Break compound questions into focused sub-questions\n"
    "- Use terminology likely found in formal documents\n"
    "Return ONLY the rewritten query. No explanation."
)


def rewrite_query(original_query: str, attempt: int) -> str:
    """
    Rewrite a query to improve retrieval results.

    Args:
        original_query: The query that failed relevance grading.
        attempt: Which rewrite attempt this is (1-based).

    Returns:
        The rewritten query string.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
        RuntimeError: If the Groq API call fails.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

        user_message = (
            f"Original query: {original_query}\n"
            f"Attempt number: {attempt}\n"
            "Return ONLY the rewritten query. No explanation."
        )

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _REWRITER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=256,
        )

        rewritten = response.choices[0].message.content.strip()

        if not rewritten:
            logger.warning(
                "Rewriter returned empty response — using original query."
            )
            return original_query

        logger.info(
            "Query rewrite attempt %d: '%s' → '%s'",
            attempt,
            original_query[:60],
            rewritten[:100],
        )
        return rewritten

    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Query rewrite failed (attempt %d)", attempt)
        raise RuntimeError(f"Query rewrite error: {exc}") from exc
