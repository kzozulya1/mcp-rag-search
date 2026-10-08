# AGENTS.md

## Project Overview

RAG search MCP server: a FastMCP-based Model Context Protocol server exposing RAG operations against a Qdrant vector database. Agents can add documents, list collections, clear the document collection, and run semantic search. Backed by LangChain vector-store integrations (`langchain-qdrant`) and OpenAI-compatible embeddings (`langchain-openai`, pointed at an OpenAI-compatible endpoint via `EMBEDDING_BASE_URL`).

## Architecture & Data Flow

Single-package MCP server, no framework beyond FastMCP.

```
MCP client (stdio/SSE)
    └── FastMCP instance `mcp` (src/rag_search/mcp_server.py:19)
          ├── add_document(text)        → RecursiveCharacterTextSplitter → OpenAIEmbeddings → QdrantVectorStore.add_texts
          ├── list_documents()          → QdrantClient.get_collections / get_collection
          ├── clear_documents()         → QdrantClient.delete_collection → recreate via _ensure_collection
          └── semantic_search(query,k)  → QdrantVectorStore.similarity_search_with_score
```

Key data flow for `add_document` (`mcp_server.py:63`):
1. Validate non-empty text (return error string).
2. Split with `RecursiveCharacterTextSplitter` using `config.CHUNK_SIZE`/`CHUNK_OVERLAP`.
3. Build `OpenAIEmbeddings`, probe dimension with `embed_query` on first chunk.
4. `_ensure_collection` creates the Qdrant collection (COSINE distance) if absent.
5. `QdrantVectorStore(client, collection_name, embedding).add_texts(chunks)`.

`clear_documents` deletes the collection then re-creates it empty (dimension probed via `embed_query("dim probe")`).

## Key Directories

- `src/rag_search/` — the only real source package:
  - `mcp_server.py` — FastMCP server, all tools, entry point (`main()`).
  - `config.py` — env-driven settings (OpenAI, Qdrant, chunking).
  - `__init__.py` — empty.
- `vectorstore.py` (repo root) — **legacy/duplicated module**. Contains `get_embeddings`, `get_qdrant_client`, `collection_exists`, `ensure_collection`, `load_document` (async), `search_similar`. It imports a root-level `config` module that no longer exists (`import config`) — this file is not used by the package (package uses `from . import config`) and will fail to import. Treat as dead code.
- `docker-compose.yml` — Qdrant service (`qdrant/qdrant:latest`, ports 6333/6334, persistent `qdrant_data` volume).

## Development Commands

Package manager: **uv**. Python >= 3.10 (`.python-version` pins 3.14).

```bash
uv sync                 # install deps
uv run rag-search       # console script (rag_search.mcp_server:main) — SSE transport
uv run -m fastmcp.serve sse rag_search.mcp_server:mcp   # SSE transport
# stdio transport:
uv run python -m rag_search.mcp_server
docker compose up -d    # start Qdrant (localhost:6333)
## Code Conventions & Common Patterns

- **Logging**: module-level `logger = logging.getLogger(__name__)`; `logging.basicConfig(level=logging.INFO)` called in both `__main__` and `main()`. Tools log inputs at INFO (`add_document`, `clear_documents`, `semantic_search`).
- **Error handling**: defensive, non-exceptional. Empty/whitespace inputs return an error/warning string or `[]` rather than raising. `_ensure_collection` swallows `Exception` on create (collection may already exist) and logs at DEBUG. `clear_documents` catches delete failure and returns a "does not exist" message.
- **Dependency injection**: no DI framework. Client construction via private factories `_get_embeddings()` and `_get_qdrant_client()` (`mcp_server.py:26-37`) built fresh per tool call from `config`. Qdrant collection is auto-created on demand.
- **MCP tool pattern**: `@mcp.tool(annotations={...})` with OpenAI-style tool annotations (`title`, `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`). Tools are **synchronous** plain functions; FastMCP handles async bridging. Return types: `str` for mutations, `list[dict[str, Any]]` for queries.
- **Config**: `config.py` calls `load_dotenv()` at import time and exposes module-level typed constants read from env with defaults: `EMBEDDING_BASE_URL`, `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `QDRANT_HOST`, `QDRANT_PORT`, `COLLECTION_NAME`, `CHUNK_SIZE` (500), `CHUNK_OVERLAP` (50). No settings class — plain module attributes.
- **Naming**: `snake_case`; private helpers prefixed `_` (`_get_embeddings`, `_get_qdrant_client`, `_ensure_collection`). Docstrings on every tool and helper.

## Important Files

- `src/rag_search/mcp_server.py` — entry point (`main()`, `if __name__ == "__main__"`), `mcp = FastMCP(name="rag-search", version="0.1.0")`, all four tools.
- `src/rag_search/config.py` — all configuration.
- `pyproject.toml` — deps (fastmcp>=2.0, qdrant-client>=1.7, langchain>=0.3, langchain-qdrant>=0.2, langchain-openai>=0.3, python-dotenv>=1.0, langchain-text-splitters>=1.1.3), console script `rag-search = "rag_search.mcp_server:main"`, build backend `uv_build`.
- `.env` — local secrets (OpenAI-compatible base URL `http://192.168.0.203:5051/v1`, Jina embedding model `jina-embeddings-v4-text-retrieval-Q4_K_M`, Qdrant localhost:6333, collection `documents`).
- `docker-compose.yml` — Qdrant service definition.
- `vectorstore.py` — root-level legacy helper module; not imported by the package (dead code, `import config` broken).
- `README.md` — user docs (Russian); documents `uv run rag-search` (stdio) and SSE invocation, tool schemas.

## Runtime/Tooling Preferences

- Runtime: Python >= 3.10, pinned to 3.14 (`.python-version`).
- Package manager: **uv** only (`uv sync`, `uv run`). `uv.lock` present; build backend `uv_build`.
- Qdrant required as an external service (HTTP, default `localhost:6333`); no TLS by default.
- Embeddings endpoint is OpenAI-compatible — not necessarily OpenAI; configured via `EMBEDDING_BASE_URL`/`EMBEDDING_MODEL`/`OPENAI_API_KEY`.
- MCP transport: `mcp.run("sse")` in `main()`; stdio also supported via `mcp.run()` default / `python -m`.
- No linting/formatting tooling configured (no ruff/black/mypy config, no pre-commit).

## Testing & QA

- **No test suite exists.** No `tests/` directory, no test framework in `pyproject.toml` dependencies, no pytest configuration.
- QA is manual: start Qdrant via `docker compose up -d`, run the server, and exercise tools through an MCP client.
- If adding tests, use `uv add --dev pytest` and place tests under a new `tests/` package; run with `uv run pytest`.
