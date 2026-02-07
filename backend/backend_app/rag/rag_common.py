"""
RAG Common Module

Shared utilities and functions for RAG MCP and RAG Injection services.
Contains common database operations, embeddings management, and configuration.
"""
import os
import logging
import psycopg

# LangChain imports for embeddings
try:
    from langchain_ollama import OllamaEmbeddings
    from langchain_postgres import PGVector
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_ERROR = str(e)

logger = logging.getLogger(__name__)

# =========================
# Environment Configuration
# =========================
DATABASE_URL = os.getenv("DATABASE_URL", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
COLLECTION_DOCS = os.getenv("COLLECTION_DOCS", "document_embeddings")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

# PostgreSQL DSN (convert from SQLAlchemy format to psycopg format)
PG_DSN = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

# =========================
# Embeddings Cache
# =========================
# Global cache for embeddings instances by model name
_embeddings_cache = {}
_vector_store_cache = {}


def get_embeddings_for_model(model_name: str):
    """
    Get or create embeddings instance for a specific model.
    
    This function maintains a cache of embeddings instances to avoid
    recreating them for each request, which improves performance.
    
    Args:
        model_name: Name of the embedding model (e.g., "nomic-embed-text")
        
    Returns:
        OllamaEmbeddings instance or None if LangChain not available
    """
    if not LANGCHAIN_AVAILABLE:
        logger.error(f"LangChain not available: {LANGCHAIN_ERROR}")
        return None
    
    if model_name not in _embeddings_cache:
        logger.info(f"Creating embeddings instance for model: {model_name}")
        _embeddings_cache[model_name] = OllamaEmbeddings(
            model=model_name,
            base_url=OLLAMA_BASE_URL
        )
    return _embeddings_cache[model_name]


def get_vector_store_for_model(model_name: str):
    """
    Get vector store instance for a specific embedding model.
    
    Creates a PGVector store configured with the appropriate embedding model.
    This ensures that documents embedded with a specific model are searched
    using the same model for consistency.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        PGVector instance or None if LangChain not available
    """
    if not LANGCHAIN_AVAILABLE:
        logger.error(f"LangChain not available: {LANGCHAIN_ERROR}")
        return None
    
    if model_name not in _vector_store_cache:
        embeddings = get_embeddings_for_model(model_name)
        if not embeddings:
            return None
        
        logger.info(f"Creating vector store for model: {model_name}")
        _vector_store_cache[model_name] = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_DOCS,
            connection=DATABASE_URL,
            use_jsonb=True,
        )
    return _vector_store_cache[model_name]


# =========================
# Database Utilities
# =========================
def db_exec(query: str, params: tuple = ()):
    """
    Execute a database query and return results.
    
    This is a convenience wrapper around psycopg that handles
    connection management and provides a consistent interface
    for both read and write operations.
    
    Args:
        query: SQL query string (can include placeholders)
        params: Tuple of parameters for the query
        
    Returns:
        List of result rows for SELECT queries, None for other queries
    """
    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            try:
                return cur.fetchall()
            except psycopg.ProgrammingError:
                # Query didn't return results (INSERT, UPDATE, DELETE)
                return None


def get_rag_group_by_name(name: str, scope: str = "global"):
    """
    Get RAG group information by name and scope.
    
    Args:
        name: RAG group name
        scope: RAG group scope (default: "global")
        
    Returns:
        Tuple of (id, embed_model, doc_count) or None if not found
    """
    rows = db_exec("""
        SELECT id, embed_model, doc_count
        FROM rag_groups
        WHERE name = %s AND scope = %s
    """, (name, scope))
    
    if not rows:
        return None
    
    return rows[0]


def get_rag_group_embed_model(group_id: str):
    """
    Get the embedding model for a specific RAG group.
    
    Args:
        group_id: UUID of the RAG group
        
    Returns:
        Embedding model name or None if not found
    """
    rows = db_exec("""
        SELECT embed_model FROM rag_groups WHERE id = %s
    """, (group_id,))
    
    if not rows:
        return None
    
    return rows[0][0]


def list_rag_groups(scope: str = "global", owner: Optional[str] = None):
    """
    List RAG groups with optional filtering.
    
    Args:
        scope: Filter by scope (default: "global")
        owner: Optional owner identifier to filter by
        
    Returns:
        List of tuples: (id, name, scope, owner, description, embed_model, doc_count, created_at, updated_at)
    """
    query = """
        SELECT id, name, scope, owner, description, embed_model,
               doc_count, created_at, updated_at
        FROM rag_groups
        WHERE scope = %s
    """
    params = [scope]
    
    if owner:
        query += " AND owner = %s"
        params.append(owner)
    
    query += " ORDER BY name"
    
    return db_exec(query, tuple(params))


# =========================
# Module Initialization
# =========================
def init_rag_common():
    """
    Initialize the RAG common module.
    
    Call this at startup to verify configuration and dependencies.
    
    Returns:
        Dict with initialization status
    """
    status = {
        "langchain_available": LANGCHAIN_AVAILABLE,
        "database_url": DATABASE_URL[:30] + "..." if len(DATABASE_URL) > 30 else DATABASE_URL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "default_embed_model": OLLAMA_EMBED_MODEL,
        "collection_name": COLLECTION_DOCS,
    }
    
    if not LANGCHAIN_AVAILABLE:
        status["error"] = LANGCHAIN_ERROR
        logger.warning(f"⚠ RAG Common: LangChain not available - {LANGCHAIN_ERROR}")
    else:
        logger.info(f"✓ RAG Common: Initialized with model {OLLAMA_EMBED_MODEL}")
    
    return status


# Optional typing support
from typing import Optional, Tuple, List

__all__ = [
    # Configuration
    "DATABASE_URL",
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBED_MODEL",
    "COLLECTION_DOCS",
    "PG_DSN",
    "LANGCHAIN_AVAILABLE",
    
    # Embeddings functions
    "get_embeddings_for_model",
    "get_vector_store_for_model",
    
    # Database utilities
    "db_exec",
    "get_rag_group_by_name",
    "get_rag_group_embed_model",
    "list_rag_groups",
    
    # Initialization
    "init_rag_common",
]
