"""
ingestion.py — Phase 1: Parse → Chunk → Embed → Store

Uses LlamaParse for document parsing, LlamaIndex HierarchicalNodeParser for
hierarchical chunking, SentenceTransformer for local embeddings, and ChromaDB
PersistentClient for vector storage.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Any



from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

logger = logging.getLogger(__name__)

# ── Singleton model cache ────────────────────────────────────────────────────
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer instance (all-MiniLM-L6-v2)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_chroma_collection() -> chromadb.Collection:
    """Return the ChromaDB collection using PersistentClient."""
    chroma_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name="legal_financial_docs",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


async def ingest_document(file_path: str) -> dict[str, Any]:
    """
    Ingest a document through the full pipeline:
      1. Parse PDF with LlamaParse (markdown output)
      2. Chunk with LlamaIndex HierarchicalNodeParser
      3. Embed with SentenceTransformer
      4. Store in ChromaDB with rich metadata

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        Dict with keys: status, chunks_stored, filename.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the LLAMA_CLOUD_API_KEY is not set.
        RuntimeError: On any pipeline failure.
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY is not set. Add it to your .env file."
        )

    filename = Path(file_path).name
    logger.info("Starting ingestion for: %s", filename)

    try:
        # ── Step 1: Parse PDF with LlamaParse ────────────────────────────
        logger.info("Step 1/4 — Parsing PDF with LlamaParse …")
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            verbose=False,
        )
        parsed_documents = await parser.aload_data(file_path)

        if not parsed_documents:
            raise RuntimeError(
                f"LlamaParse returned no content for: {filename}"
            )

        # Convert LlamaParse output to LlamaIndex Document objects
        llama_docs: list[Document] = []
        for idx, doc in enumerate(parsed_documents):
            llama_docs.append(
                Document(
                    text=doc.text,
                    metadata={
                        "source_filename": filename,
                        "page_number": idx + 1,
                    },
                )
            )

        logger.info(
            "Parsed %d page(s) from %s", len(llama_docs), filename
        )

        # ── Step 2: Hierarchical chunking ────────────────────────────────
        logger.info("Step 2/4 — Chunking with HierarchicalNodeParser …")
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128],
        )
        all_nodes = node_parser.get_nodes_from_documents(llama_docs)
        leaf_nodes = get_leaf_nodes(all_nodes)

        if not leaf_nodes:
            raise RuntimeError(
                "HierarchicalNodeParser produced zero leaf nodes."
            )

        logger.info(
            "Created %d leaf nodes from %d total nodes",
            len(leaf_nodes),
            len(all_nodes),
        )

        # ── Step 3: Attach metadata and embed ───────────────────────────
        logger.info("Step 3/4 — Generating embeddings …")
        model = get_embedding_model()

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for chunk_idx, node in enumerate(leaf_nodes):
            text = node.get_content()
            if not text.strip():
                continue

            page_number = node.metadata.get("page_number", 0)
            section_header = _extract_section_header(text)

            meta = {
                "source_filename": filename,
                "page_number": int(page_number),
                "section_header": section_header,
                "chunk_index": chunk_idx,
            }

            texts.append(text)
            metadatas.append(meta)
            ids.append(f"{filename}_{chunk_idx}")

        if not texts:
            raise RuntimeError("All leaf nodes were empty after filtering.")

        # Batch-encode all texts in one call
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # ── Step 4: Store in ChromaDB ────────────────────────────────────
        logger.info("Step 4/4 — Storing %d chunks in ChromaDB …", len(texts))
        collection = get_chroma_collection()

        # Upsert in batches of 500 to avoid oversized requests
        batch_size = 500
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            collection.upsert(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

        result = {
            "status": "success",
            "chunks_stored": len(texts),
            "filename": filename,
        }
        logger.info("Ingestion complete: %s", result)
        return result

    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed for %s", filename)
        raise RuntimeError(f"Ingestion pipeline error: {exc}") from exc


def _extract_section_header(text: str) -> str:
    """
    Attempt to extract the nearest Markdown heading from a chunk's text.

    Falls back to the first 60 characters if no heading is found.
    """
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    # Fallback: first meaningful line truncated
    first_line = text.strip().split("\n")[0].strip()
    return first_line[:60] if first_line else "Unknown Section"


# ── Helper for synchronous callers ──────────────────────────────────────────
def ingest_document_sync(file_path: str) -> dict[str, Any]:
    """Synchronous wrapper around the async ingest_document function."""
    return asyncio.run(ingest_document(file_path))
