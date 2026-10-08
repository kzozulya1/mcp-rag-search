"""FastMCP server for RAG search operations on Qdrant."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from . import config

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="rag-search",
    version="0.1.0",

)


def _get_embeddings() -> OpenAIEmbeddings:
    """Create and return an OpenAIEmbeddings instance from config."""
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_base=config.EMBEDDING_BASE_URL,
    )


def _get_qdrant_client() -> QdrantClient:
    """Create and return a QdrantClient instance from config."""
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def _ensure_collection(client: QdrantClient, embedding_dim: int) -> None:
    """Create the target collection if it does not already exist."""
    try:
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )
    except Exception:
        logger.debug("Collection '%s' already exists or creation failed", config.COLLECTION_NAME)


@mcp.tool(
    annotations={
        "title": "Add Document",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
def add_document(text: str) -> str:
    """Load text into Qdrant, splitting it into chunks.

    Args:
        text: The document text to load.
    """
    logger.info("add_document: text=%s", text)
    if not text or not text.strip():
        return "Error: text is empty."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)

    if not chunks:
        return "Warning: No chunks produced (text too short or empty after splitting)."

    embeddings = _get_embeddings()
    # Determine embedding dimension from the first chunk
    first_vec = embeddings.embed_query(chunks[0])
    embedding_dim = len(first_vec)

    client = _get_qdrant_client()
    _ensure_collection(client, embedding_dim)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_texts(chunks)

    return f"Added {len(chunks)} chunk(s) to collection '{config.COLLECTION_NAME}'."


@mcp.tool(
    annotations={
        "title": "List Documents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def list_documents() -> list[dict[str, Any]]:
    """List all collections in Qdrant with their vector count and dimension.

    Returns:
        A list of dicts, each with 'name', 'vectors_count', and 'dimensions'.
    """
    client = _get_qdrant_client()
    collections = client.get_collections().collections

    result: list[dict[str, Any]] = []
    for coll in collections:
        info = client.get_collection(coll.name)
        params = info.config.params
        if params is None:
            size = None
        elif isinstance(params.vectors, dict):
            # named vector configuration: {name: VectorParams}
            first = next(iter(params.vectors.values()), None)
            size = getattr(first, "size", None) if first is not None else None
        else:
            # single unnamed vector configuration: VectorParams
            size = params.vectors.size if params.vectors is not None else None
        result.append({
            "name": coll.name,
            "dimensions": size,
            "vectors_count": info.points_count,
        })

    return result


@mcp.tool(
    annotations={
        "title": "Clear Documents",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def clear_documents() -> str:
    """Remove all documents from the documents collection.

    Returns:
        A confirmation message.
    """
    logger.info("clear_documents: removing all documents from '%s'", config.COLLECTION_NAME)
    client = _get_qdrant_client()

    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        return f"Collection '{config.COLLECTION_NAME}' does not exist."

    embedding_dim = _get_embeddings().embed_query("dim probe")
    _ensure_collection(client, len(embedding_dim))

    return f"Collection '{config.COLLECTION_NAME}' cleared and recreated."


@mcp.tool(
    annotations={
        "title": "Semantic Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def semantic_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Perform a semantic search over indexed documents.

    Args:
        query: The search query string.
        k: Number of top results to return (default 5).
    """
    logger.info("semantic_search: query=%s, k=%d", query, k)
    if not query or not query.strip():
        return []

    embeddings = _get_embeddings()
    client = _get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
    )
    results = vector_store.similarity_search_with_score(query, k=k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        }
        for doc, score in results
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run("sse")


def main() -> None:
    """Entry point: run the MCP server with SSE transport."""
    logging.basicConfig(level=logging.INFO)
    mcp.run("sse")
