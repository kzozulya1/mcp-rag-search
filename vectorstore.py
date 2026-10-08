"""Vector store helpers for Qdrant-backed RAG operations."""

from __future__ import annotations

import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import config

logger = logging.getLogger(__name__)


def get_embeddings() -> OpenAIEmbeddings:
    """Create and return an OpenAIEmbeddings instance from config."""
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_base=config.EMBEDDING_BASE_URL,
    )


def get_qdrant_client() -> QdrantClient:
    """Create and return a QdrantClient instance from config."""
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def collection_exists() -> bool:
    """Check if the target collection exists in Qdrant."""
    client = get_qdrant_client()
    collections = client.get_collections().collections
    return any(c.name == config.COLLECTION_NAME for c in collections)


def ensure_collection(client: QdrantClient, embedding_dim: int) -> None:
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


async def load_document(text: str) -> None:
    """Split text into chunks and load into Qdrant.

    Args:
        text: Document text to index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)

    if not chunks:
        logger.warning("No chunks produced from document text (empty or too small).")
        return

    embeddings = get_embeddings()
    embedding_dim = len(embeddings.embed_query(chunks[0]))

    client = get_qdrant_client()
    ensure_collection(client, embedding_dim)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_texts(chunks)


def search_similar(query: str, k: int = 5) -> list[str]:
    """Find the most similar chunks for a query.

    Args:
        query: Search query string.
        k: Number of results to return.

    Returns:
        List of matching page contents.
    """
    embeddings = get_embeddings()
    client = get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
    )
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
