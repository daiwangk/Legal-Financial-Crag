"""
clauses.py — /extract_clauses endpoint.

Structured extraction of legal/financial clauses from uploaded documents.
This is NOT routed through CRAG — it retrieves chunks directly and uses
Groq with JSON-mode to extract structured clause data.
"""

import os
import json
import logging
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from groq import Groq

from api.models import (
    ClauseExtractionRequest,
    ClauseExtractionResponse,
    ClauseItem,
)
from core_logic.retriever import retrieve_chunks

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Extraction prompt ────────────────────────────────────────────────────────
_CLAUSE_EXTRACTION_PROMPT = """\
You are a Legal Document Clause Extraction AI.

Analyze the following document chunks and extract ALL important legal and financial clauses.

For EACH clause, provide:
- clause_type: The category (e.g., "Termination", "Payment Terms", "Liability", \
"Indemnification", "Confidentiality", "Non-Compete", "Force Majeure", \
"Governing Law", "Dispute Resolution", "Warranty", "Limitation of Liability", \
"Intellectual Property", "Data Protection", "Insurance", "Penalty")
- clause_text: The full relevant text of the clause
- page_number: The page number where the clause appears
- risk_level: Your assessment — "low", "medium", or "high"

Risk assessment guidelines:
- "high": Clauses with significant financial exposure, unlimited liability, \
harsh penalties, or one-sided termination rights
- "medium": Clauses with moderate obligations or conditional risks
- "low": Standard boilerplate clauses with minimal risk

Return your response as a JSON object with this exact structure:
{
  "clauses": [
    {
      "clause_type": "string",
      "clause_text": "string",
      "page_number": integer,
      "risk_level": "low" | "medium" | "high"
    }
  ]
}

Document Chunks:
"""


@router.post(
    "/extract_clauses",
    response_model=ClauseExtractionResponse,
    summary="Extract key clauses from uploaded documents",
    description=(
        "Retrieves top chunks from ChromaDB and uses Groq LLM with "
        "structured JSON output to extract and classify legal/financial clauses."
    ),
)
async def extract_clauses(
    request: ClauseExtractionRequest = ClauseExtractionRequest(),
) -> ClauseExtractionResponse:
    """
    Extract structured clause data from the uploaded documents.

    Args:
        request: Optional extraction instruction (defaults to extracting all clauses).

    Returns:
        ClauseExtractionResponse with list of ClauseItem objects.

    Raises:
        HTTPException 500: If extraction fails.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY is not set. Add it to your .env file.",
            )

        # Retrieve top 10 chunks for maximum clause coverage
        logger.info("extract_clauses — retrieving chunks for: '%s'", request.query[:80])
        chunks = retrieve_chunks(request.query, k=10)

        if not chunks:
            return ClauseExtractionResponse(
                filename="N/A",
                clauses=[],
                total_found=0,
            )

        # Determine primary filename from chunks
        filenames: set[str] = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            fn = meta.get("source_filename", "unknown")
            filenames.add(fn)
        primary_filename = filenames.pop() if len(filenames) == 1 else "multiple"

        # Build context from retrieved chunks
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            context_parts.append(
                f"--- Chunk {i} (Page {meta.get('page_number', '?')}, "
                f"Section: {meta.get('section_header', 'N/A')}) ---\n"
                f"{chunk.get('text', '')}\n"
            )
        context_str = "\n".join(context_parts)

        full_prompt = _CLAUSE_EXTRACTION_PROMPT + context_str

        # Call Groq with JSON mode
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal clause extraction assistant. "
                        "Always respond with valid JSON only."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content.strip()
        logger.debug("Groq raw clause extraction response: %s", raw_content[:500])

        # Parse JSON response
        parsed: dict[str, Any] = json.loads(raw_content)

        clauses_data = parsed.get("clauses", [])
        validated_clauses: list[ClauseItem] = []
        for c in clauses_data:
            try:
                validated_clauses.append(
                    ClauseItem(
                        clause_type=c.get("clause_type", "Unknown"),
                        clause_text=c.get("clause_text", ""),
                        page_number=int(c.get("page_number", 0)),
                        risk_level=c.get("risk_level", "medium").lower(),
                    )
                )
            except (ValueError, TypeError) as parse_err:
                logger.warning("Skipping malformed clause: %s — %s", c, parse_err)
                continue

        result = ClauseExtractionResponse(
            filename=primary_filename,
            clauses=validated_clauses,
            total_found=len(validated_clauses),
        )

        logger.info(
            "Extracted %d clauses from '%s'",
            result.total_found,
            primary_filename,
        )
        return result

    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse Groq JSON response")
        raise HTTPException(
            status_code=500,
            detail=f"LLM returned invalid JSON: {exc}",
        )
    except Exception as exc:
        logger.exception("Clause extraction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Clause extraction error: {exc}",
        )
