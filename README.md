# RAG Search MCP Server

An MCP server for RAG operations backed by Qdrant. Exposes tools for adding documents, listing collections, clearing the document collection, and performing semantic search.

## Dependencies

- Python >= 3.10 (pinned to 3.14)
- `fastmcp` — MCP server library
- `qdrant-client` — Qdrant client
- `langchain`, `langchain-qdrant` — vector store
- `langchain-openai` — embeddings
- `langchain-text-splitters` — chunking
- `python-dotenv` — `.env` loading

## Installation

```bash
uv sync
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-api-key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=documents

CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

`config.py` calls `load_dotenv()` at import time; all values have sensible defaults if `.env` is absent.

## Running

Start Qdrant (if not already running):

```bash
docker compose up -d
```

### SSE transport (default)

```bash
uv run rag-search
```

### Stdio transport

```bash
uv run python -m rag_search.mcp_server
```

## Tools

### `add_document`

Loads text into Qdrant, splitting it into chunks.

**Parameters:**
- `text` (str, required) — the document text to load

**Returns:** a result string (e.g. `"Added 3 chunk(s) to collection 'documents'."`).

### `list_documents`

Returns all Qdrant collections with vector count and dimension.

**Returns:** `list[dict]` — each entry contains `name`, `vectors_count`, `dimensions`.

### `clear_documents`

Removes all documents from the `documents` collection and recreates it empty.

**Returns:** a confirmation string (e.g. `"Collection 'documents' cleared and recreated."`).

### `semantic_search`

Performs a semantic search over indexed documents.

**Parameters:**
- `query` (str, required) — the search query
- `k` (int, default=5) — number of results to return

**Returns:** `list[dict]` — each entry contains `content`, `metadata`, `score`.

## Project Structure

```
rag-search/
├── pyproject.toml           # deps, console script, build backend
├── docker-compose.yml       # Qdrant service
├── .env                     # local secrets (not committed)
├── vectorstore.py           # legacy module (dead code, not imported by package)
├── src/rag_search/
│   ├── __init__.py          # empty
│   ├── config.py            # env-driven settings (OpenAI, Qdrant, chunking)
│   └── mcp_server.py        # FastMCP server, all tools, entry point
└── README.md
```

## Security Notes

- API keys are stored in `.env` and must not be committed to the repository.
- The Qdrant connection uses plain HTTP by default — configure TLS or use gRPC for production.
- Tool inputs are validated by FastMCP based on type hints.
