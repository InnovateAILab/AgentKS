import hashlib
import json
import os
import uuid
from typing import Any, Dict, Optional

import psycopg
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Config (moved here as common shared resources)
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_DOCS = os.getenv("COLLECTION_DOCS", "kb_docs")
COLLECTION_TOOLS = os.getenv("COLLECTION_TOOLS", "tool_catalog")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
CDS_BASE_URL = os.getenv("CDS_BASE_URL", "https://cds.cern.ch")
INSPIRE_BASE_URL = os.getenv("INSPIRE_BASE_URL", "https://inspirehep.net")
ARXIV_API_URL = os.getenv("ARXIV_API_URL", "http://export.arxiv.org/api/query")

TOOL_SELECT_TOPK = int(os.getenv("TOOL_SELECT_TOPK", "6"))
FETCH_TIMEOUT_SECONDS = int(os.getenv("FETCH_TIMEOUT_SECONDS", "25"))
MAX_CHARS_PER_DOC = int(os.getenv("MAX_CHARS_PER_DOC", "250000"))
MAX_URLS_PER_REQUEST = int(os.getenv("MAX_URLS_PER_REQUEST", "30"))

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

PG_DSN = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")


# OpenGPTs integration (optional)
# If OPENGPTS_URL is set and USE_OPENGPTS=true, the backend will call a
# running OpenGPTs instance to run assistants instead of the local agent.
OPENGPTS_URL = os.getenv("OPENGPTS_URL")
OPENGPTS_ASSISTANT_ID = os.getenv("OPENGPTS_ASSISTANT_ID")
USE_OPENGPTS = os.getenv("USE_OPENGPTS", "false").lower() in ("1", "true", "yes")
OPENGPTS_JWT = os.getenv("OPENGPTS_JWT")  # optional Bearer token for auth


# LLM + Embeddings
llm = ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


# Vector stores
doc_vs = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_DOCS,
    connection=DATABASE_URL,
    use_jsonb=True,
)

tool_vs = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_TOOLS,
    connection=DATABASE_URL,
    use_jsonb=True,
)

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)


# DB helpers
def db_exec(query: str, params: tuple = ()):
    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            try:
                return cur.fetchall()
            except psycopg.ProgrammingError:
                return None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def tool_run_log(
    tool_id: str,
    user_id: str,
    request_obj: Dict[str, Any],
    response_obj: Any,
    status: str,
    error: Optional[str],
    latency_ms: Optional[int],
):
    db_exec(
        """
        INSERT INTO tool_runs (id, tool_id, user_id, request, response, status, error, latency_ms, created_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now())
        """,
        (
            str(uuid.uuid4()),
            tool_id,
            user_id,
            json.dumps(request_obj),
            json.dumps(response_obj) if response_obj is not None else None,
            status,
            error,
            latency_ms,
        ),
    )


def upsert_source(scope: str, user_id: Optional[str], url: str) -> str:
    url_sha = sha256_text(url)
    new_id = str(uuid.uuid4())
    rows = db_exec(
        """
        INSERT INTO sources (id, scope, user_id, url, url_sha256, status, error, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, 'queued', NULL, now(), now())
        ON CONFLICT (scope, user_id, url_sha256)
        DO UPDATE SET status='queued', error=NULL, updated_at=now()
        RETURNING id
        """,
        (new_id, scope, user_id, url, url_sha),
    )
    return str(rows[0][0])


def mark_source_status(source_id: str, status: str, error: Optional[str] = None, content_sha: Optional[str] = None):
    db_exec(
        """
        UPDATE sources
        SET status=%s,
            error=%s,
            content_sha256=COALESCE(%s, content_sha256),
            fetched_at=CASE WHEN %s IN ('ingested','failed') THEN now() ELSE fetched_at END,
            updated_at=now()
        WHERE id=%s
        """,
        (status, error, content_sha, status, source_id),
    )


def tool_upsert(
    tool_id: str,
    body: Any,
    created_by: str,
    provider: str = "native",
    mcp_server: Optional[str] = None,
    mcp_tool: Optional[str] = None,
):
    db_exec(
        """
        INSERT INTO tools (id, name, description, type, config, enabled, scope, provider, mcp_server, mcp_tool, created_by, created_at, updated_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now())
        ON CONFLICT (name)
        DO UPDATE SET
          description=EXCLUDED.description,
          type=EXCLUDED.type,
          config=EXCLUDED.config,
          enabled=EXCLUDED.enabled,
          scope=EXCLUDED.scope,
          provider=EXCLUDED.provider,
          mcp_server=EXCLUDED.mcp_server,
          mcp_tool=EXCLUDED.mcp_tool,
          updated_at=now()
        """,
        (
            tool_id,
            body.name,
            body.description,
            body.type,
            json.dumps(body.config),
            body.enabled,
            body.scope,
            provider,
            mcp_server,
            mcp_tool,
            created_by,
        ),
    )


def tool_patch(tool_id: str, patch: Any):
    sets = []
    params = []
    if patch.description is not None:
        sets.append("description=%s")
        params.append(patch.description)
    if patch.enabled is not None:
        sets.append("enabled=%s")
        params.append(patch.enabled)
    if patch.scope is not None:
        sets.append("scope=%s")
        params.append(patch.scope)
    if patch.config is not None:
        sets.append("config=%s")
        params.append(json.dumps(patch.config))
    sets.append("updated_at=now()")
    params.append(tool_id)
    db_exec(f"UPDATE tools SET {', '.join(sets)} WHERE id=%s", tuple(params))


def tool_get_by_name(name: str):
    rows = db_exec(
        "SELECT id,name,description,type,config,enabled,scope,provider,mcp_server,mcp_tool FROM tools WHERE name=%s",
        (name,),
    )
    return rows[0] if rows else None


def tool_get_by_ids(ids: list):
    if not ids:
        return []
    placeholders = ",".join(["%s"] * len(ids))
    rows = (
        db_exec(
            f"SELECT id,name,description,type,config,enabled,scope,provider,mcp_server,mcp_tool FROM tools WHERE id IN ({placeholders})",
            tuple(ids),
        )
        or []
    )
    return rows


def tool_list_for_role(role: str):
    if role == "admin":
        return (
            db_exec(
                "SELECT id,name,description,type,config,enabled,scope,provider,mcp_server,mcp_tool FROM tools ORDER BY name"
            )
            or []
        )
    return (
        db_exec(
            "SELECT id,name,description,type,config,enabled,scope,provider,mcp_server,mcp_tool FROM tools WHERE scope='global' ORDER BY name"
        )
        or []
    )


def index_tool_desc(tool_id: str, name: str, description: str, enabled: bool, scope: str):
    # Simple append index; selection de-dupes by tool_id
    tool_vs.add_documents(
        [
            Document(
                page_content=description,
                metadata={"tool_id": tool_id, "name": name, "enabled": enabled, "scope": scope},
            )
        ]
    )
