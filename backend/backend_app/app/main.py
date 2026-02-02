import ast
import asyncio
import hashlib
import json
import os
import time
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Literal
from urllib.parse import urlparse

import psycopg
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool

from .langgraph_adapter import run_agent_with_langgraph
from .mcp import run_mcp_tool_async


# RAG helpers live in app.rag (imported after PGVector setup)

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


# =========================
# LLM + Embeddings
# =========================
llm = ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


# =========================
# Vector stores
# =========================
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

# RAG helpers (split into app.rag)
from .rag import add_documents, rag_search, fetch_url_text, sha256_text, splitter  # type: ignore

app = FastAPI(title="OpenWebUI Backend: RAG + Auto Tools + MCP (CDS + arXiv + INSPIRE-HEP + SearxNG)")


# =========================
# Schemas
# =========================
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str = "rag-agent"
    messages: List[ChatMessage]
    stream: Optional[bool] = False


class ToolCreate(BaseModel):
    name: str
    description: str
    type: Literal["searxng_search", "cds_search", "arxiv_search", "inspirehep_search", "http_json", "mcp_tool"]
    scope: Literal["global", "admin_only"] = "global"
    enabled: bool = True
    config: Dict[str, Any] = {}


class ToolPatch(BaseModel):
    description: Optional[str] = None
    enabled: Optional[bool] = None
    scope: Optional[Literal["global", "admin_only"]] = None
    config: Optional[Dict[str, Any]] = None


class MCPSyncRequest(BaseModel):
    server_name: str
    url: str
    headers: Dict[str, str] = {}
    scope: Literal["global", "admin_only"] = "global"
    enabled: bool = True


class IngestUrlsRequest(BaseModel):
    urls: List[str]
    scope: Literal["private", "global"] = "private"


# =========================
# Auth helpers (Authentik forwarded headers)
# =========================
def user_from_headers(request: Request) -> dict:
    return {
        "email": request.headers.get("X-Authentik-Email", "unknown"),
        "name": request.headers.get("X-Authentik-Name", ""),
        "groups": request.headers.get("X-Authentik-Groups", ""),
    }


def is_admin(request: Request) -> bool:
    return "admin" in (request.headers.get("X-Authentik-Groups", "") or "").lower()


def require_admin(request: Request):
    """Raise HTTP 403 if the request's Authentik groups do not contain 'admin'."""
    groups = request.headers.get("X-Authentik-Groups", "")
    if "admin" not in groups.lower():
        raise HTTPException(status_code=403, detail="admin group required")


# Backwards-compatible helpers to derive user id / role from either
# Authentik forwarded headers or OpenWebUI headers (or Authorization token).
def get_user_id(
    x_authentik_email: Optional[str] = None,
    x_openwebui_user_id: Optional[str] = None,
    authorization: Optional[str] = None,
) -> str:
    # Prefer Authentik email when available (acts as canonical user id)
    if x_authentik_email:
        return x_authentik_email
    if x_openwebui_user_id:
        return x_openwebui_user_id
    # Try to extract a token-based user id from Authorization if present (bearer token)
    if authorization:
        return authorization
    return "anonymous"


def get_user_role(x_authentik_groups: Optional[str] = None, x_openwebui_user_role: Optional[str] = None) -> str:
    # Authentik groups contain group names; prefer them if present
    if x_authentik_groups:
        groups = (x_authentik_groups or "").lower()
        return "admin" if "admin" in groups else "user"
    if x_openwebui_user_role:
        return x_openwebui_user_role
    return "user"


def assert_admin(role: str):
    if role != "admin":
        raise HTTPException(status_code=403, detail="admin role required for this endpoint")


# =========================
# DB helpers
# =========================
def db_exec(query: str, params: tuple = ()):
    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            try:
                return cur.fetchall()
            except psycopg.ProgrammingError:
                return None



def tool_run_log(
    tool_id: str,
    user_id: str,
    request_obj: Dict[str, Any],
    response_obj: Any,
    status: str,
    error: Optional[str],
    latency_ms: Optional[int],
):
    # Align with migration-defined tool_runs columns: id, tool_id, input, output, status, started_at, finished_at
    db_exec(
        """
        INSERT INTO tool_runs (id, tool_id, input, output, status, started_at, finished_at)
        VALUES (%s, %s, %s, %s, %s, now(), now())
        """,
        (
            str(uuid.uuid4()),
            tool_id,
            json.dumps(request_obj),
            json.dumps(response_obj) if response_obj is not None else None,
            status,
        ),
    )


def tool_upsert(
    tool_id: str,
    body: ToolCreate,
    created_by: str,
    provider: str = "native",
    mcp_server: Optional[str] = None,
    mcp_tool: Optional[str] = None,
):
    # Migration defines tools as: id, name, kind, mcp_id, metadata JSONB, tags JSONB, created_at, updated_at
    mcp_id = mcp_server
    metadata = json.dumps(body.config or {})
    tags = json.dumps([])
    db_exec(
        """
        INSERT INTO tools (id, name, kind, mcp_id, metadata, tags, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, now(), now())
        ON CONFLICT (id)
        DO UPDATE SET
          name = EXCLUDED.name,
          kind = EXCLUDED.kind,
          mcp_id = EXCLUDED.mcp_id,
          metadata = EXCLUDED.metadata,
          tags = EXCLUDED.tags,
          updated_at = now()
        """,
        (
            tool_id,
            body.name,
            body.type,
            mcp_id,
            metadata,
            tags,
        ),
    )


def tool_patch(tool_id: str, patch: ToolPatch):
    sets = []
    params = []
    if patch.description is not None:
        # map description into metadata.description
        sets.append("metadata = jsonb_strip_nulls(coalesce(metadata, '{}'::jsonb) || %s::jsonb)")
        params.append(json.dumps({"description": patch.description}))
    if patch.enabled is not None:
        # no explicit enabled column in current schema; store in metadata
        sets.append("metadata = jsonb_strip_nulls(coalesce(metadata, '{}'::jsonb) || %s::jsonb)")
        params.append(json.dumps({"enabled": patch.enabled}))
    if patch.scope is not None:
        # store scope in metadata
        sets.append("metadata = jsonb_strip_nulls(coalesce(metadata, '{}'::jsonb) || %s::jsonb)")
        params.append(json.dumps({"scope": patch.scope}))
    if patch.config is not None:
        sets.append("metadata = %s")
        params.append(json.dumps(patch.config))
    sets.append("updated_at=now()")
    params.append(tool_id)
    db_exec(f"UPDATE tools SET {', '.join(sets)} WHERE id=%s", tuple(params))


def tool_get_by_name(name: str):
    rows = db_exec("SELECT id,name,kind,mcp_id,metadata,tags FROM tools WHERE name=%s", (name,))
    return rows[0] if rows else None


def tool_get_by_ids(ids: List[str]):
    if not ids:
        return []
    placeholders = ",".join(["%s"] * len(ids))
    query = (
        "SELECT id, name, kind, mcp_id, metadata, tags "
        f"FROM tools WHERE id IN ({placeholders})"
    )
    rows = db_exec(query, tuple(ids)) or []
    return rows


def tool_list_for_role(role: str):
    if role == "admin":
        q_admin = "SELECT id, name, kind, mcp_id, metadata, tags FROM tools ORDER BY name"
        return db_exec(q_admin) or []
    q = "SELECT id, name, kind, mcp_id, metadata, tags FROM tools WHERE (metadata->>'scope') = 'global' OR (metadata->>'scope') IS NULL ORDER BY name"
    return db_exec(q) or []


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


# =========================
# Sources table helpers (for URL ingestion tracking)
# =========================
def upsert_source(scope: str, user_id: Optional[str], url: str) -> str:
    """Upsert into `urls` table and return the url id.

    The project migrations define a canonical `urls` table. Older code used a
    separate `sources` table; convert that usage to `urls` so ingestion and
    status tracking use the normalized schema.
    """
    new_id = str(uuid.uuid4())
    # Insert or update by unique url; keep scope and set status='queued'.
    rows = db_exec(
        """
        INSERT INTO urls (id, url, scope, status, created_at)
        VALUES (%s, %s, %s, 'queued', now())
        ON CONFLICT (url)
        DO UPDATE SET scope = EXCLUDED.scope, status = 'queued'
        RETURNING id
        """,
        (new_id, url, scope),
    )
    return str(rows[0][0])


def mark_source_status(source_id: str, status: str, error: Optional[str] = None, content_sha: Optional[str] = None):
    # Map source status to the urls table columns (last_fetched_at, last_error)
    if status in ("ingested", "failed"):
        fetched_at_clause = "last_fetched_at = now(),"
    else:
        fetched_at_clause = ""

    db_exec(
        f"""
        UPDATE urls
        SET status=%s,
            last_error=%s,
            {fetched_at_clause}
            -- keep created_at untouched
            -- no updated_at column on urls in current schema
            url = url
        WHERE id=%s
        """,
        (status, error, source_id),
    )


# =========================
# Seed default tools into DB + tool_vs
# =========================
def ensure_default_tools():
    def _ensure(name: str, desc: str, typ: str, config: Dict[str, Any]):
        if tool_get_by_name(name):
            return
        tid = str(uuid.uuid4())
        body = ToolCreate(
            name=name,
            description=desc,
            type=typ,  # type: ignore
            scope="global",
            enabled=True,
            config=config,
        )
        tool_upsert(tid, body, created_by="system")
        index_tool_desc(tid, body.name, body.description, body.enabled, body.scope)

    _ensure(
        "cds_search",
        "Search CERN Document Server (CDS) for reports/papers and return top records.",
        "cds_search",
        {"base_url": CDS_BASE_URL, "rg": 5},
    )
    _ensure(
        "arxiv_search",
        "Search arXiv (Atom API) for papers and return top results.",
        "arxiv_search",
        {"api_url": ARXIV_API_URL, "max_results": 5, "sortBy": "submittedDate", "sortOrder": "descending"},
    )
    _ensure(
        "inspirehep_search",
        "Search INSPIRE-HEP literature for high-energy physics papers and return top results.",
        "inspirehep_search",
        {"base_url": INSPIRE_BASE_URL, "size": 5},
    )
    _ensure(
        "web_search",
        "Search the web via SearxNG and return top results.",
        "searxng_search",
        {"base_url": SEARXNG_URL},
    )

    # Also register available basic tools defined in app.basic_tools. This is
    # best-effort and idempotent: we only create a DB row when the tool name
    # is not already present.
    try:
        from . import basic_tools

        tool_classes = [
            basic_tools.DDGSearch,
            basic_tools.Arxiv,
            basic_tools.YouSearch,
            basic_tools.SecFilings,
            basic_tools.PressReleases,
            basic_tools.PubMed,
            basic_tools.Wikipedia,
            basic_tools.Tavily,
            basic_tools.TavilyAnswer,
            basic_tools.DallE,
        ]

        for cls in tool_classes:
            try:
                name = getattr(cls, "name")
                desc = getattr(cls, "description")
                typ = getattr(cls, "type").value if getattr(cls, "type", None) is not None else None
                if not name or not typ:
                    continue
                _ensure(name, desc, typ, {})
            except Exception:
                # Non-fatal: skip problematic tool
                continue
    except Exception:
        # basic_tools unavailable in some environments (tests/editor); ignore
        pass


@app.on_event("startup")
def _startup():
    ensure_default_tools()


# =========================
# RAG store + retrieve
# =========================
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)


def add_documents(scope: Literal["global", "private"], user_id: str, docs: List[Document]) -> int:
    chunks = splitter.split_documents(docs)
    for d in chunks:
        d.metadata["scope"] = scope
        d.metadata["user_id"] = user_id if scope == "private" else None
    doc_vs.add_documents(chunks)
    return len(chunks)


def rag_search(user_id: str, query: str, k: int = 6) -> List[Document]:
    filt = {
        "$or": [{"scope": {"$eq": "global"}}, {"$and": [{"scope": {"$eq": "private"}}, {"user_id": {"$eq": user_id}}]}]
    }
    return doc_vs.similarity_search(query, k=k, filter=filt)


# =========================
# URL fetch + clean
# =========================
def fetch_url_text(url: str) -> str:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs allowed")
    r = requests.get(url, timeout=FETCH_TIMEOUT_SECONDS, headers={"User-Agent": "Mozilla/5.0 (RAGBot)"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join([ln for ln in lines if ln])
    return text[:MAX_CHARS_PER_DOC]


# =========================
# Calculator: rule + tool
# =========================
_ALLOWED = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd}


def safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval").body

    def _eval(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED:
            v = _eval(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED:
            a = _eval(n.left)
            b = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow):
                return a**b
            if isinstance(n.op, ast.Mod):
                return a % b
        raise ValueError("Unsupported expression")

    return float(_eval(node))


def looks_like_math(text: str) -> bool:
    allowed = set("0123456789+-*/().% ^\n\t ")
    t = text.strip()
    return t and all(ch in allowed for ch in t)


@tool
def calculator(expression: str) -> str:
    """Do arithmetic exactly. Input like '12*(3+4)'."""
    return str(safe_eval(expression))


# =========================
# Auto tool selection (Top-K)
# =========================
def select_tools_for_query(role: str, query: str, k: int) -> List[str]:
    filt = {"enabled": {"$eq": True}} if role == "admin" else {"enabled": {"$eq": True}, "scope": {"$eq": "global"}}
    docs = tool_vs.similarity_search(query, k=k, filter=filt)
    tool_ids: List[str] = []
    for d in docs:
        tid = d.metadata.get("tool_id")
        if tid and tid not in tool_ids:
            tool_ids.append(tid)
    return tool_ids


# =========================
# Tool implementations
# =========================
def _remember_url(used_urls: Dict[str, int], url: Optional[str]) -> None:
    if not url:
        return
    if url not in used_urls:
        used_urls[url] = len(used_urls) + 1


def run_searxng(base_url: str, query: str, used_urls: Dict[str, int]) -> Dict[str, Any]:
    r = requests.get(f"{base_url.rstrip('/')}/search", params={"q": query, "format": "json"}, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = []
    for x in (data.get("results") or [])[:5]:
        url = x.get("url")
        _remember_url(used_urls, url)
        results.append({"title": x.get("title"), "url": url, "snippet": x.get("content")})
    return {"results": results}


def run_cds_search(base_url: str, query: str, rg: int, used_urls: Dict[str, int]) -> Dict[str, Any]:
    params = {
        "p": query,
        "of": "recjson",
        "rg": str(rg),
        "jrec": "1",
        "ot": "recid,creation_date,authors[0],number_of_authors,title,abstract",
    }
    r = requests.get(f"{base_url.rstrip('/')}/search", params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    out = []
    for rec in (data or [])[:rg]:
        recid = rec.get("recid")
        title = None
        t = rec.get("title")
        if isinstance(t, dict):
            title = t.get("title") or t.get("subtitle")
        elif isinstance(t, str):
            title = t
        record_url = f"{base_url.rstrip('/')}/record/{recid}" if recid else None
        _remember_url(used_urls, record_url)
        out.append({"recid": recid, "title": title, "creation_date": rec.get("creation_date"), "url": record_url})
    return {"results": out}


def _xml_text(el: Optional[ET.Element]) -> Optional[str]:
    return None if el is None else (el.text or "").strip()


def run_arxiv_search(
    api_url: str, query: str, max_results: int, sortBy: str, sortOrder: str, used_urls: Dict[str, int]
) -> Dict[str, Any]:
    params = {
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }
    r = requests.get(api_url, params=params, timeout=25)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("a:entry", ns):
        title = _xml_text(entry.find("a:title", ns))
        summary = _xml_text(entry.find("a:summary", ns))
        link_el = entry.find("a:link[@rel='alternate']", ns)
        url = link_el.attrib.get("href") if link_el is not None else None
        _remember_url(used_urls, url)
        results.append(
            {
                "title": title,
                "url": url,
                "summary": (summary[:400] + "...") if summary and len(summary) > 400 else summary,
            }
        )
    return {"results": results}


def run_inspirehep_search(base_url: str, query: str, size: int, used_urls: Dict[str, int]) -> Dict[str, Any]:
    r = requests.get(
        f"{base_url.rstrip('/')}/api/literature",
        params={"q": query, "size": str(size)},
        headers={"Accept": "application/json"},
        timeout=25,
    )
    r.raise_for_status()
    data = r.json()
    hits = ((data or {}).get("hits") or {}).get("hits") or []
    out = []
    for h in hits[:size]:
        meta = h.get("metadata") or {}
        titles = meta.get("titles") or []
        title = titles[0].get("title") if titles and isinstance(titles[0], dict) else None
        links = h.get("links") or {}
        url = links.get("html") or links.get("self")
        _remember_url(used_urls, url)
        out.append({"title": title, "url": url})
    return {"results": out}


def run_db_tool(row: tuple, user_id: str, payload: Dict[str, Any], used_urls: Dict[str, int]) -> Any:
    tool_id, name, desc, typ, config_json, enabled, scope, provider, mcp_server, mcp_tool = row
    config = config_json if isinstance(config_json, dict) else json.loads(config_json or "{}")

    t0 = time.time()
    try:
        if not enabled:
            tool_run_log(tool_id, user_id, payload, None, "skipped", "tool disabled", 0)
            return {"error": "tool disabled"}

        if typ == "searxng_search":
            out = run_searxng(config.get("base_url") or SEARXNG_URL, payload.get("query", ""), used_urls)
        elif typ == "cds_search":
            out = run_cds_search(
                config.get("base_url") or CDS_BASE_URL, payload.get("query", ""), int(config.get("rg") or 5), used_urls
            )
        elif typ == "arxiv_search":
            out = run_arxiv_search(
                config.get("api_url") or ARXIV_API_URL,
                payload.get("query", ""),
                int(config.get("max_results") or 5),
                str(config.get("sortBy") or "submittedDate"),
                str(config.get("sortOrder") or "descending"),
                used_urls,
            )
        elif typ == "inspirehep_search":
            out = run_inspirehep_search(
                config.get("base_url") or INSPIRE_BASE_URL,
                payload.get("query", ""),
                int(config.get("size") or 5),
                used_urls,
            )
        elif typ == "mcp_tool":
            mcp_url = config.get("url")
            headers = config.get("headers") or {}
            toolname = config.get("tool") or mcp_tool
            if not mcp_url or not toolname:
                raise ValueError("mcp_tool missing config.url or tool name")
            out = asyncio.run(run_mcp_tool_async(mcp_url, headers, toolname, payload))
        else:
            out = {"error": f"unknown tool type: {typ}"}

        tool_run_log(tool_id, user_id, payload, out, "ok", None, int((time.time() - t0) * 1000))
        return out
    except Exception as e:
        tool_run_log(tool_id, user_id, payload, None, "error", str(e), int((time.time() - t0) * 1000))
        return {"error": str(e)}


# =========================
# Agent toolset = base tools + auto-selected DB tools
# =========================
def make_agent_tools(user_id: str, role: str, user_query: str, used_urls: Dict[str, int]):
    @tool
    def kb_search(query: str) -> str:
        """Search KB (global + your private memory). Return numbered evidence blocks."""
        docs = rag_search(user_id, query, k=6)
        if not docs:
            return "No relevant passages found."
        blocks = []
        for d in docs:
            url = d.metadata.get("source_url") or d.metadata.get("source") or "unknown"
            _remember_url(used_urls, url)
            idx = used_urls.get(url, 0)
            scope = d.metadata.get("scope", "unknown")
            blocks.append(f"[{idx}] (scope={scope}) URL={url}\n{d.page_content}")
        return "\n\n---\n\n".join(blocks)

    @tool
    def save_memory(text: str) -> str:
        """Save private memory ONLY when user explicitly asks to remember."""
        n = add_documents(
            "private", user_id, [Document(page_content=text, metadata={"source_url": "memory", "type": "note"})]
        )
        return f"Saved to private memory. chunks={n}"

    @tool
    def ingest_urls(urls: List[str], scope: Literal["private", "global"] = "private") -> str:
        """Fetch URLs and ingest into RAG. Only admins may ingest GLOBAL."""
        if len(urls) > MAX_URLS_PER_REQUEST:
            return f"Too many URLs. Max is {MAX_URLS_PER_REQUEST}."

        if scope == "global":
            if role != "admin":
                return "Denied: only admin can ingest GLOBAL knowledge."
            target_scope = "global"
            target_user = None
        else:
            target_scope = "private"
            target_user = user_id

        ok, fail = [], []
        for u in urls:
            sid = None
            try:
                sid = upsert_source(target_scope, target_user, u)
                mark_source_status(sid, "fetching")
                txt = fetch_url_text(u)
                csha = sha256_text(txt)
                n = add_documents(
                    target_scope,
                    user_id,
                    [Document(page_content=txt, metadata={"source_url": u, "source_id": sid, "type": "url"})],
                )
                mark_source_status(sid, "ingested", content_sha=csha)
                ok.append(f"{u} (chunks={n})")
            except Exception as e:
                if sid:
                    mark_source_status(sid, "failed", error=str(e))
                fail.append(f"{u} (error={e})")

        out = []
        if ok:
            out.append("Ingested:\n" + "\n".join(ok))
        if fail:
            out.append("Failed:\n" + "\n".join(fail))
        return "\n\n".join(out) if out else "Nothing ingested."

    selected_tool_ids = select_tools_for_query(role, user_query, k=TOOL_SELECT_TOPK)
    rows = tool_get_by_ids(selected_tool_ids)

    db_tools = []
    for row in rows:

        def _factory(r):
            rid, rname, rdesc, *_ = r

            @tool(name=rname, description=rdesc)
            def _t(**kwargs):
                return run_db_tool(r, user_id, kwargs, used_urls)

            return _t

        db_tools.append(_factory(row))

    return [kb_search, calculator, save_memory, ingest_urls] + db_tools


def run_agent(user_id: str, role: str, messages: List[ChatMessage]) -> str:
    """Delegate to the langgraph adapter (with safe fallback).

    The adapter is intentionally conservative: if langgraph primitives are
    available it may use them, otherwise it falls back to the existing
    create_agent flow. We pass `llm` and `make_agent_tools` so the adapter
    can reuse the same tool wiring.
    """
    # Normalize messages to simple dicts for the adapter
    try:
        msgs = [
            (
                m.model_dump()
                if hasattr(m, "model_dump")
                else (m if isinstance(m, dict) else {"role": getattr(m, "role", "user"), "content": str(m)})
            )
            for m in messages
        ]
    except Exception:
        msgs = messages

    return run_agent_with_langgraph(user_id, role, msgs, llm, make_agent_tools)


# =========================
# OpenAI-compatible endpoints (for Open WebUI)
# =========================
@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "rag-agent", "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionsRequest,
    x_openwebui_user_id: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Id"),
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_authentik_email: Optional[str] = Header(default=None, alias="X-Authentik-Email"),
    x_authentik_groups: Optional[str] = Header(default=None, alias="X-Authentik-Groups"),
):
    user_id = get_user_id(x_authentik_email, x_openwebui_user_id, authorization)
    role = get_user_role(x_authentik_groups, x_openwebui_user_role)
    answer = run_agent(user_id, role, req.messages)

    if req.stream:

        def gen():
            cid = f"chatcmpl-{int(time.time())}"
            chunk1 = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            chunk2 = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
            }
            chunk3 = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield f"data: {json.dumps(chunk3)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
    }
