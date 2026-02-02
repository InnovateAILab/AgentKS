import ast
import asyncio
import hashlib
import json
import os
import time
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Literal
from typing_extensions import TypedDict
from urllib.parse import urlparse

import psycopg
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Header, HTTPException, Query, Request, Form
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from authlib.integrations.starlette_client import OAuth, OAuthError
from pydantic import BaseModel

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_agent

from langchain_mcp_adapters.client import MultiServerMCPClient

# Try to import StateGraph (support either `langchain.graph` or legacy `langgraph`)
try:
    from langchain.graph import StateGraph, START, END  # type: ignore
except Exception:
    try:
        from langgraph.graph import StateGraph, START, END  # type: ignore
    except Exception:
        StateGraph = None  # type: ignore


# =========================
# Config
# =========================
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

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)

app = FastAPI(title="OpenWebUI Backend: RAG + Auto Tools + MCP (CDS + arXiv + INSPIRE-HEP + SearxNG)")

# Simple web UI assets (templates + static)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Simple admin password for web UI (insecure default; set ADMIN_PASSWORD in env)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
# OIDC configuration (optional). Provide OIDC_DISCOVERY_URL, OIDC_CLIENT_ID and OIDC_CLIENT_SECRET to enable.
OIDC_CLIENT_ID = os.getenv("OIDC_CLIENT_ID")
OIDC_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET")
OIDC_DISCOVERY_URL = os.getenv("OIDC_DISCOVERY_URL")
OIDC_SCOPES = os.getenv("OIDC_SCOPES", "openid email profile")
ADMIN_USERS = [s.strip().lower() for s in (os.getenv("ADMIN_USERS", "").split(",") if os.getenv("ADMIN_USERS") else [])]

_oauth = None
if OIDC_CLIENT_ID and OIDC_CLIENT_SECRET and OIDC_DISCOVERY_URL:
    oauth = OAuth()
    oauth.register(
        name="oidc",
        client_id=OIDC_CLIENT_ID,
        client_secret=OIDC_CLIENT_SECRET,
        server_metadata_url=OIDC_DISCOVERY_URL,
        client_kwargs={"scope": OIDC_SCOPES},
    )
    _oauth = oauth
else:
    _oauth = None


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
    type: Literal[
        "searxng_search", "cds_search", "arxiv_search", "inspirehep_search", "http_json", "mcp_tool", "travis_search"
    ]
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
# Auth helpers (Open WebUI forwarded headers)
# =========================
def get_user_id(x_openwebui_user_id: Optional[str], authorization: Optional[str]) -> str:
    if x_openwebui_user_id:
        return x_openwebui_user_id
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    raise HTTPException(401, "Missing user identity (X-OpenWebUI-User-Id or Bearer token)")


def get_user_role(x_openwebui_user_role: Optional[str]) -> str:
    return (x_openwebui_user_role or "user").lower()


def assert_admin(role: str):
    if role != "admin":
        raise HTTPException(403, "Admin only")


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


def tool_upsert(
    tool_id: str,
    body: ToolCreate,
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


def tool_patch(tool_id: str, patch: ToolPatch):
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


def tool_get_by_ids(ids: List[str]):
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


# =========================
# Sources table helpers (for URL ingestion tracking)
# =========================
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
    _ensure(
        "travis_search",
        "Search the Travis search engine for results and return top matches.",
        "travis_search",
        {"base_url": "https://travis-search.example/search"},
    )


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


def run_travis_search(base_url: str, query: str, used_urls: Dict[str, int]) -> Dict[str, Any]:
    """Search a Travis-like search endpoint that returns JSON results.
    Expected to accept `q` or `query` and return an object with a `results` array
    where each item has at least `url` and `title`/`content` fields. This is a
    best-effort adapter — adjust `base_url` and params in tool config as needed.
    """
    try:
        # prefer `/search` path similar to searxng; allow base_url to include path
        r = requests.get(f"{base_url.rstrip('/')}/search", params={"q": query, "format": "json"}, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        # fallback: try GET with `query` param
        r = requests.get(base_url, params={"query": query}, timeout=20)
        r.raise_for_status()
        data = r.json()

    results = []
    for x in (data.get("results") or [])[:10]:
        url = x.get("url") or x.get("link") or x.get("href")
        _remember_url(used_urls, url)
        results.append(
            {
                "title": x.get("title") or x.get("name") or "",
                "url": url,
                "snippet": x.get("content") or x.get("snippet") or x.get("description"),
            }
        )
    return {"results": results}


async def run_mcp_tool_async(mcp_url: str, headers: Dict[str, str], tool_name: str, payload: Dict[str, Any]) -> Any:
    client = MultiServerMCPClient({"srv": {"transport": "http", "url": mcp_url, "headers": headers}})
    tools = await client.get_tools()
    target = next((t for t in tools if t.name == tool_name), None)
    if not target:
        raise ValueError(f"MCP tool not found: {tool_name}")
    return await target.ainvoke(payload)


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
        elif typ == "travis_search":
            out = run_travis_search(
                config.get("base_url") or config.get("url") or "", payload.get("query", ""), used_urls
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


# -------------------------
# StateGraph: small wrapper to run the agent as a graph node
# -------------------------
class GraphState(TypedDict):
    graph_state: Dict[str, Any]


# Cached compiled graph (or None if import failed)
_COMPILED_GRAPH = None


def _agent_invoke_direct(user_id: str, role: str, messages: List[Dict[str, Any]], used_urls: Dict[str, int]) -> str:
    """Core agent invocation logic extracted from run_agent.
    Accepts messages as list-of-dicts (same shape as .model_dump()) and a mutable used_urls dict.
    Returns the assistant answer string.
    """
    user_query = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")

    if looks_like_math(user_query):
        try:
            return str(safe_eval(user_query))
        except Exception:
            pass

    tools = make_agent_tools(user_id, role, user_query, used_urls)

    system = (
        "You are a helpful assistant with tools.\n"
        "- Use kb_search for internal knowledge.\n"
        "- Use cds_search for CERN Document Server.\n"
        "- Use arxiv_search for arXiv.\n"
        "- Use inspirehep_search for HEP literature.\n"
        "- Use web_search for general web queries.\n"
        "- Use calculator for arithmetic.\n"
        "- Only save_memory if user explicitly asks to remember.\n"
        "- Only admins can ingest GLOBAL knowledge.\n"
        "- Cite sources with [1], [2] etc.\n"
        "- Do NOT reveal hidden reasoning.\n"
    )

    agent = create_agent(model=llm, tools=tools, system_prompt=system)
    result = agent.invoke(
        {"messages": [{"role": "system", "content": system}] + list(messages)},
        config={"recursion_limit": 25},
    )

    answer = ""
    for m in reversed(result["messages"]):
        if getattr(m, "type", "") == "ai":
            answer = m.content
            break
        if isinstance(m, dict) and m.get("role") == "assistant":
            answer = m.get("content", "")
            break

    # Attach discovered URLs as side-effect already recorded in used_urls
    return answer or "No response produced."


def _node_agent(state: GraphState) -> GraphState:
    """Graph node that runs the agent and returns updated state containing answer + used_urls."""
    payload = state.get("graph_state") or {}
    user_id = payload.get("user_id", "")
    role = payload.get("role", "user")
    messages = payload.get("messages", [])
    used_urls = payload.get("used_urls", {}) or {}

    answer = _agent_invoke_direct(user_id, role, messages, used_urls)

    return {"graph_state": {"answer": answer, "used_urls": used_urls}}


def _build_graph():
    global _COMPILED_GRAPH
    if not StateGraph:
        return None
    builder = StateGraph(GraphState)
    builder.add_node("agent_node", _node_agent)
    builder.add_edge(START, "agent_node")
    builder.add_edge("agent_node", END)
    _COMPILED_GRAPH = builder.compile()
    return _COMPILED_GRAPH


# Build the graph at import-time (best-effort). If StateGraph isn't available, we'll fall back.
_COMPILED_GRAPH = _build_graph()


def run_agent(user_id: str, role: str, messages: List[ChatMessage]) -> str:
    user_query = next((m.content for m in reversed(messages) if m.role == "user"), "")

    if looks_like_math(user_query):
        try:
            return str(safe_eval(user_query))
        except Exception:
            pass

    used_urls: Dict[str, int] = {}
    tools = make_agent_tools(user_id, role, user_query, used_urls)

    system = (
        "You are a helpful assistant with tools.\n"
        "- Use kb_search for internal knowledge.\n"
        "- Use cds_search for CERN Document Server.\n"
        "- Use arxiv_search for arXiv.\n"
        "- Use inspirehep_search for HEP literature.\n"
        "- Use web_search for general web queries.\n"
        "- Use calculator for arithmetic.\n"
        "- Only save_memory if user explicitly asks to remember.\n"
        "- Only admins can ingest GLOBAL knowledge.\n"
        "- Cite sources with [1], [2] etc.\n"
        "- Do NOT reveal hidden reasoning.\n"
    )

    agent = create_agent(model=llm, tools=tools, system_prompt=system)
    result = agent.invoke(
        {"messages": [{"role": "system", "content": system}] + [m.model_dump() for m in messages]},
        config={"recursion_limit": 25},
    )

    answer = ""
    for m in reversed(result["messages"]):
        if getattr(m, "type", "") == "ai":
            answer = m.content
            break
        if isinstance(m, dict) and m.get("role") == "assistant":
            answer = m.get("content", "")
            break

    if used_urls:
        refs = [f"- [{idx}]({url})" for url, idx in sorted(used_urls.items(), key=lambda x: x[1]) if url]
        if refs:
            answer += "\n\n---\n\n**References**\n" + "\n".join(refs)

    return answer or "No response produced."


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
):
    user_id = get_user_id(x_openwebui_user_id, authorization)
    role = get_user_role(x_openwebui_user_role)
    answer = run_agent(user_id, role, req.messages)

    if req.stream:

        def gen():
            cid = f"chatcmpl-{int(time.time())}"
            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'model': req.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': answer}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
    }


# =========================
# Tool management APIs
# =========================
@app.get("/tools")
def list_tools(
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
):
    role = get_user_role(x_openwebui_user_role)
    rows = tool_list_for_role(role)
    return [
        {"id": r[0], "name": r[1], "description": r[2], "type": r[3], "enabled": r[5], "scope": r[6], "provider": r[7]}
        for r in rows
    ]


@app.post("/admin/tools")
def admin_create_tool(
    body: ToolCreate,
    x_openwebui_user_id: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Id"),
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
):
    role = get_user_role(x_openwebui_user_role)
    assert_admin(role)
    created_by = x_openwebui_user_id or "admin"
    tid = str(uuid.uuid4())
    tool_upsert(tid, body, created_by=created_by)
    index_tool_desc(tid, body.name, body.description, body.enabled, body.scope)
    return {"id": tid, "name": body.name}


@app.patch("/admin/tools/{tool_id}")
def admin_patch_tool(
    tool_id: str,
    patch: ToolPatch,
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
):
    role = get_user_role(x_openwebui_user_role)
    assert_admin(role)
    tool_patch(tool_id, patch)

    row = db_exec("SELECT id,name,description,enabled,scope FROM tools WHERE id=%s", (tool_id,))
    if row:
        _id, name, desc, enabled, scope = row[0]
        index_tool_desc(tool_id, name, desc, enabled, scope)
    return {"ok": True}


@app.get("/admin/tool-runs")
def admin_tool_runs(
    limit: int = Query(100, ge=1, le=500),
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
):
    role = get_user_role(x_openwebui_user_role)
    assert_admin(role)
    rows = (
        db_exec(
            """
        SELECT tr.id, tr.tool_id, t.name, tr.user_id, tr.status, tr.error, tr.latency_ms, tr.created_at
        FROM tool_runs tr
        JOIN tools t ON t.id = tr.tool_id
        ORDER BY tr.created_at DESC
        LIMIT %s
        """,
            (limit,),
        )
        or []
    )
    keys = ["id", "tool_id", "tool_name", "user_id", "status", "error", "latency_ms", "created_at"]
    return [dict(zip(keys, r)) for r in rows]


# ------------------------
# Web UI: simple cookie-based login + admin pages
# ------------------------
def _web_get_role(request: Request) -> str:
    # Prefer explicit header, fall back to cookie
    hdr = request.headers.get("X-OpenWebUI-User-Role")
    if hdr:
        return hdr.lower()
    c = request.cookies.get("openwebui_user_role")
    return (c or "user").lower()


def _require_admin_web(request: Request):
    role = _web_get_role(request)
    if role != "admin":
        # redirect to login page
        raise HTTPException(status_code=403, detail="Admin required")


@app.get("/admin/ui/login")
def ui_login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None, "oidc_enabled": bool(_oauth)})


@app.post("/admin/ui/login")
def ui_login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    # Very small auth: if password matches ADMIN_PASSWORD, user becomes admin
    resp = RedirectResponse(url="/admin/ui/", status_code=303)
    if password == ADMIN_PASSWORD:
        resp.set_cookie("openwebui_user_id", username or "admin", httponly=True)
        resp.set_cookie("openwebui_user_role", "admin", httponly=True)
        return resp
    # invalid
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


@app.get("/admin/ui/logout")
def ui_logout(request: Request):
    resp = RedirectResponse(url="/admin/ui/login", status_code=303)
    resp.delete_cookie("openwebui_user_id")
    resp.delete_cookie("openwebui_user_role")
    return resp


@app.get("/admin/ui/oidc/login")
async def ui_oidc_login(request: Request):
    if not _oauth:
        return RedirectResponse(url="/admin/ui/login")
    redirect_uri = str(request.url_for("ui_oidc_callback"))
    return await _oauth.oidc.authorize_redirect(request, redirect_uri)


@app.route("/admin/ui/oidc/callback", methods=["GET", "POST"])
async def ui_oidc_callback(request: Request):
    if not _oauth:
        return RedirectResponse(url="/admin/ui/login")
    try:
        token = await _oauth.oidc.authorize_access_token(request)
    except OAuthError:
        return RedirectResponse(url="/admin/ui/login")

    # Try parse id_token into claims
    claims = None
    try:
        claims = await _oauth.oidc.parse_id_token(request, token)
    except Exception:
        # fallback to userinfo endpoint
        try:
            claims = await _oauth.oidc.userinfo(token=token)
        except Exception:
            claims = {}

    # Determine user identity and role
    uid = (claims.get("email") or claims.get("preferred_username") or claims.get("sub")) if claims else None
    uid = uid or "oidc-user"
    roles = []
    if claims:
        r = claims.get("role") or claims.get("roles") or claims.get("groups")
        if isinstance(r, str):
            roles = [r]
        elif isinstance(r, (list, tuple)):
            roles = list(r)

    is_admin = False
    # Admin if email is in ADMIN_USERS env or token includes admin role
    if uid and ADMIN_USERS and uid.lower() in ADMIN_USERS:
        is_admin = True
    for rr in roles:
        if isinstance(rr, str) and rr.lower() == "admin":
            is_admin = True

    resp = RedirectResponse(url="/admin/ui/", status_code=303)
    resp.set_cookie("openwebui_user_id", uid, httponly=True)
    resp.set_cookie("openwebui_user_role", "admin" if is_admin else "user", httponly=True)
    return resp


@app.get("/admin/ui/")
def ui_index(request: Request):
    # simple index page linking to features
    role = _web_get_role(request)
    return templates.TemplateResponse("base.html", {"request": request, "role": role})


@app.get("/admin/ui/urls")
def ui_list_urls(request: Request):
    # require admin via cookie/header
    role = _web_get_role(request)
    if role != "admin":
        return RedirectResponse(url="/admin/ui/login")
    rows = (
        db_exec(
            "SELECT id, scope, user_id, url, status, error, created_at, updated_at FROM sources ORDER BY created_at DESC LIMIT 200"
        )
        or []
    )
    sources = [
        dict(id=r[0], scope=r[1], user_id=r[2], url=r[3], status=r[4], error=r[5], created_at=r[6], updated_at=r[7])
        for r in rows
    ]
    return templates.TemplateResponse("urls.html", {"request": request, "sources": sources})


@app.get("/admin/ui/mcp")
def ui_list_mcp(request: Request):
    role = _web_get_role(request)
    if role != "admin":
        return RedirectResponse(url="/admin/ui/login")
    rows = (
        db_exec(
            "SELECT id,name,description,provider,mcp_server,mcp_tool,enabled,scope FROM tools WHERE provider=%s ORDER BY name",
            ("mcp",),
        )
        or []
    )
    tools = [
        dict(
            id=r[0],
            name=r[1],
            description=r[2],
            provider=r[3],
            mcp_server=r[4],
            mcp_tool=r[5],
            enabled=r[6],
            scope=r[7],
        )
        for r in rows
    ]
    # group by mcp_server
    servers = {}
    for t in tools:
        srv = t.get("mcp_server") or "unknown"
        servers.setdefault(srv, []).append(t)
    return templates.TemplateResponse("mcp.html", {"request": request, "servers": servers})


# =========================
# Admin: URL ingestion API (no need to rely on agent calling)
# =========================
@app.post("/admin/ingest/urls")
def admin_ingest_urls(
    body: IngestUrlsRequest,
    x_openwebui_user_id: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Id"),
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
):
    role = get_user_role(x_openwebui_user_role)
    user_id = x_openwebui_user_id or "admin"

    if body.scope == "global":
        assert_admin(role)
        scope = "global"
        owner = None
    else:
        scope = "private"
        owner = user_id

    if len(body.urls) > MAX_URLS_PER_REQUEST:
        raise HTTPException(400, f"Too many URLs. Max is {MAX_URLS_PER_REQUEST}.")

    results = {"ingested": [], "failed": []}
    for u in body.urls:
        sid = None
        try:
            sid = upsert_source(scope, owner, u)
            mark_source_status(sid, "fetching")
            txt = fetch_url_text(u)
            csha = sha256_text(txt)
            n = add_documents(
                scope,
                user_id,
                [Document(page_content=txt, metadata={"source_url": u, "source_id": sid, "type": "url"})],
            )
            mark_source_status(sid, "ingested", content_sha=csha)
            results["ingested"].append({"url": u, "chunks": n})
        except Exception as e:
            if sid:
                mark_source_status(sid, "failed", error=str(e))
            results["failed"].append({"url": u, "error": str(e)})

    return results


# =========================
# MCP sync: admin imports all tools from an MCP server into DB
# =========================
@app.post("/admin/mcp/sync")
def admin_mcp_sync(
    body: MCPSyncRequest,
    x_openwebui_user_id: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Id"),
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
):
    role = get_user_role(x_openwebui_user_role)
    assert_admin(role)
    created_by = x_openwebui_user_id or "admin"

    async def _sync():
        client = MultiServerMCPClient(
            {body.server_name: {"transport": "http", "url": body.url, "headers": body.headers}}
        )
        return await client.get_tools()

    mcp_tools = asyncio.run(_sync())
    imported = []

    for t in mcp_tools:
        name = f"{body.server_name}__{t.name}"
        tid = str(uuid.uuid4())
        tool_create = ToolCreate(
            name=name,
            description=t.description or f"MCP tool {t.name}",
            type="mcp_tool",
            scope=body.scope,
            enabled=body.enabled,
            config={"url": body.url, "headers": body.headers, "tool": t.name},
        )
        tool_upsert(
            tid, tool_create, created_by=created_by, provider="mcp", mcp_server=body.server_name, mcp_tool=t.name
        )
        index_tool_desc(tid, tool_create.name, tool_create.description, tool_create.enabled, tool_create.scope)
        imported.append({"name": name, "mcp_tool": t.name})

    return {"count": len(imported), "imported": imported}
