"""
retriever.py — ChromaDB semantic search logic.

Embeds the user query with the same SentenceTransformer model used during
ingestion and performs cosine-similarity search over the persistent
ChromaDB collection.
"""

import logging
from typing import Any

from core_logic.ingestion import get_embedding_model, get_chroma_collection

logger = logging.getLogger(__name__)


def retrieve_chunks(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks from ChromaDB for a given query.

    Args:
        query: Natural-language search query.
        k: Number of results to return (default 5).

    Returns:
        List of dicts, each containing:
          - text (str): The chunk content.
          - metadata (dict): source_filename, page_number, section_header, chunk_index.
          - distance (float): Cosine distance (lower = more similar).

    Raises:
        RuntimeError: If the retrieval process fails.
    """
    try:
        if not query or not query.strip():
            logger.warning("Empty query received — returning no results.")
            return []

        # Embed the query with the same model used at ingestion time
        model = get_embedding_model()
        query_embedding = model.encode([query], show_progress_bar=False).tolist()[0]

        # Query ChromaDB
        collection = get_chroma_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB batch format into a flat list of dicts
        chunks: list[dict[str, Any]] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_text, meta, dist in zip(documents, metadatas, distances):
            chunks.append(
                {
                    "text": doc_text,
                    "metadata": meta,
                    "distance": float(dist),
                }
            )

        logger.info(
            "Retrieved %d chunks for query (first 80 chars): '%s'",
            len(chunks),
            query[:80],
        )
        return chunks

    except Exception as exc:
        logger.exception("Retrieval failed for query: %s", query[:120])
        raise RuntimeError(f"ChromaDB retrieval error: {exc}") from exc
