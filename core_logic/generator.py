"""
generator.py — Final answer generator with citations.

Uses Groq with "llama-3.3-70b-versatile" to generate precise, cited answers
strictly grounded in the retrieved document context.  Extracts inline
citations and returns them as a structured list.
"""

import os
import re
import logging
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

# ── Generator system prompt — strict grounding rules ─────────────────────────
_GENERATOR_SYSTEM_PROMPT = (
    "You are a Legal and Financial Document Analysis AI. You MUST follow these rules:\n"
    "1. Answer ONLY using information present in the provided document context below.\n"
    "2. If the answer is not in the context, respond with exactly:\n"
    "   'The provided documents do not contain the answer to this query.'\n"
    "3. After EVERY factual claim, append a citation in this exact format:\n"
    "   [Source: {filename}, Page {page_number}, Section: {section_header}]\n"
    "4. Never use prior knowledge. Never hallucinate. Never guess.\n"
    "5. Be precise and professional."
)


def generate_answer(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate a grounded answer with citations from retrieved document chunks.

    Args:
        query: The user's question.
        chunks: List of retrieved chunk dicts (each with "text" and "metadata").

    Returns:
        Dict with keys:
          - answer (str): The generated answer text.
          - citations (list[str]): Extracted citation strings.
          - chunks_used (int): Number of chunks provided as context.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
        RuntimeError: If the Groq API call fails.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

        if not chunks:
            logger.warning("No chunks provided — returning fallback answer.")
            return {
                "answer": "The provided documents do not contain the answer to this query.",
                "citations": [],
                "chunks_used": 0,
            }

        # ── Build context block ─────────────────────────────────────────
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            context_parts.append(
                f"--- Document Chunk {i} ---\n"
                f"Filename: {meta.get('source_filename', 'N/A')}\n"
                f"Page: {meta.get('page_number', 'N/A')}\n"
                f"Section: {meta.get('section_header', 'N/A')}\n"
                f"Content:\n{chunk.get('text', '')}\n"
            )
        context_str = "\n".join(context_parts)

        user_message = (
            f"Document Context:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            "Provide a detailed, cited answer following the rules above."
        )

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=2048,
        )

        answer_text = response.choices[0].message.content.strip()

        # ── Extract citations ───────────────────────────────────────────
        citation_pattern = r"\[Source:\s*[^\]]+\]"
        citations = re.findall(citation_pattern, answer_text)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_citations: list[str] = []
        for c in citations:
            if c not in seen:
                seen.add(c)
                unique_citations.append(c)

        result = {
            "answer": answer_text,
            "citations": unique_citations,
            "chunks_used": len(chunks),
        }

        logger.info(
            "Generated answer (%d chars, %d citations) for query: '%s'",
            len(answer_text),
            len(unique_citations),
            query[:60],
        )
        return result

    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Answer generation failed for query: %s", query[:80])
        raise RuntimeError(f"Generation error: {exc}") from exc
