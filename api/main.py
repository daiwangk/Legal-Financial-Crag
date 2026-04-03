"""
main.py — FastAPI application entry point.

Configures CORS, includes all API routers under /api/v1, and provides a
root health-check endpoint.
"""



import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import upload, chat, clauses

# ── Logging configuration ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Legal & Financial Document Intelligence API",
    description=(
        "A Corrective RAG (CRAG) system for analyzing legal and financial "
        "documents.  Upload PDFs, ask questions with cited answers, and "
        "extract structured clause data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow all origins for local development ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers ─────────────────────────────────────────────────────────
app.include_router(upload.router, prefix="/api/v1", tags=["Document Upload"])
app.include_router(chat.router, prefix="/api/v1", tags=["Document Chat"])
app.include_router(clauses.router, prefix="/api/v1", tags=["Clause Extraction"])


# ── Health check ─────────────────────────────────────────────────────────────
@app.get(
    "/",
    summary="Health check",
    description="Returns API status. Use to verify the server is running.",
)
async def health_check() -> dict[str, str]:
    """Root health-check endpoint."""
    return {"status": "ok", "service": "Legal & Financial Document Intelligence API"}


# ── Run with uvicorn ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting API server on http://0.0.0.0:8000")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
