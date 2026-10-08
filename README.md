# RAG Search MCP Server

MCP-сервер для работы с RAG-системой на базе Qdrant. Предоставляет инструменты для загрузки документов, управления коллекциями и семантического поиска.

## Зависимости

- Python >= 3.10
- `fastmcp` — MCP-серверная библиотека
- `qdrant-client` — клиент Qdrant
- `langchain`, `langchain-qdrant` — векторное хранилище
- `langchain-openai` — эмбеддинги
- `python-dotenv` — загрузка `.env`

## Установка

```bash
uv sync
```

## Конфигурация

Создайте `.env` в корне проекта:

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

## Запуск

### Stdio-транспорт (по умолчанию)

```bash
uv run rag-search
```

### SSE-транспорт

```bash
uv run -m fastmcp.serve sse rag_search.mcp_server:mcp
```

## Инструменты

### `add_document`

Загружает текст в Qdrant, разбивая на чанки.

**Параметры:**
- `text` (str, required) — текст документа
- `metadata` (dict | None, optional) — метаданные для чанков

**Возвращает:** строку с результатом операции.

### `list_documents`

Возвращает список всех коллекций Qdrant с метриками.

**Возвращает:** `list[dict]` — каждая запись содержит `name`, `vectors_count`, `dimensions`.

### `semantic_search`

Семантический поиск по индексированным документам.

**Параметры:**
- `query` (str, required) — поисковый запрос
- `k` (int, default=5) — количество результатов

**Возвращает:** `list[dict]` — каждая запись содержит `content`, `metadata`, `score`.

## Структура проекта

```
rag-search/
├── pyproject.toml
├── config.py              # Конфигурация (OpenAI, Qdrant, Chunking)
├── vectorstore.py         # Векторные операции (load, search, collection)
├── src/rag_search/
│   ├── __init__.py        # Entry-point MCP-сервера
│   └── mcp_server.py      # FastMCP сервер с инструментами
└── README.md
```

## Security Notes

- API-ключи хранятся в `.env` и не должны попадать в репозиторий.
- Qdrant-подключение использует HTTP без TLS по умолчанию — для продакшена настройте HTTPS или используйте gRPC.
- Ввод инструментов валидируется FastMCP на основе type hints.
