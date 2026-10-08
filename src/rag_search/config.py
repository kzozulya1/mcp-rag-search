import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI-compatible
EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Models
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Qdrant
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")

# Chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
