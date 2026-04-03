"""
upload.py — /upload_document endpoint.

Handles multipart file upload, saves the PDF to disk, and triggers the
full ingestion pipeline (parse → chunk → embed → store).
"""

import os
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
import aiofiles

from api.models import UploadResponse
from core_logic.ingestion import ingest_document

logger = logging.getLogger(__name__)

router = APIRouter()

# Resolve upload directory relative to project root
_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "uploaded_docs",
)


@router.post(
    "/upload_document",
    response_model=UploadResponse,
    summary="Upload and ingest a PDF document",
    description=(
        "Accepts a PDF file via multipart/form-data, parses it with LlamaParse, "
        "chunks it hierarchically, embeds the chunks, and stores them in ChromaDB."
    ),
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a PDF document for ingestion into the RAG pipeline.

    Args:
        file: The uploaded PDF file (multipart/form-data).

    Returns:
        UploadResponse with status, filename, and chunks_stored.

    Raises:
        HTTPException 400: If the file is not a PDF.
        HTTPException 500: If ingestion fails.
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are accepted. Got: '{file.filename}'",
            )

        # Ensure upload directory exists
        os.makedirs(_UPLOAD_DIR, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(_UPLOAD_DIR, file.filename)
        logger.info("Saving uploaded file to: %s", file_path)

        async with aiofiles.open(file_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        # Run the ingestion pipeline
        result = await ingest_document(file_path)

        return UploadResponse(
            status=result["status"],
            filename=result["filename"],
            chunks_stored=result["chunks_stored"],
            message=(
                f"Successfully ingested '{result['filename']}' — "
                f"{result['chunks_stored']} chunks stored in ChromaDB."
            ),
        )

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.exception("File not found during upload")
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        logger.exception("Configuration error during upload")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Upload/ingestion failed for: %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        )
