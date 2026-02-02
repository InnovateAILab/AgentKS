import json
import time
import uuid
from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import common
from common import Document
from .rag.rag import add_documents, fetch_url_text
from .tools.tools import run_agent as tools_run_agent
from .opengpts_local.stream import stream_local_assistant
from .mcp.mcp import get_mcp_tools

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


# Delegate DB / tool helpers to common to avoid duplication after refactor
def db_exec(query: str, params: tuple = ()):
    return common.db_exec(query, params)


def sha256_text(s: str) -> str:
    return common.sha256_text(s)


def tool_run_log(
    tool_id: str,
    user_id: str,
    request_obj: Dict[str, Any],
    response_obj: Any,
    status: str,
    error: Optional[str],
    latency_ms: Optional[int],
):
    return common.tool_run_log(tool_id, user_id, request_obj, response_obj, status, error, latency_ms)


def tool_upsert(
    tool_id: str,
    body: ToolCreate,
    created_by: str,
    provider: str = "native",
    mcp_server: Optional[str] = None,
    mcp_tool: Optional[str] = None,
):
    return common.tool_upsert(tool_id, body, created_by, provider, mcp_server, mcp_tool)


def tool_patch(tool_id: str, patch: ToolPatch):
    return common.tool_patch(tool_id, patch)


def tool_get_by_name(name: str):
    return common.tool_get_by_name(name)


def tool_get_by_ids(ids: List[str]):
    return common.tool_get_by_ids(ids)


def tool_list_for_role(role: str):
    return common.tool_list_for_role(role)


def index_tool_desc(tool_id: str, name: str, description: str, enabled: bool, scope: str):
    return common.index_tool_desc(tool_id, name, description, enabled, scope)


def upsert_source(scope: str, user_id: Optional[str], url: str) -> str:
    return common.upsert_source(scope, user_id, url)


def mark_source_status(source_id: str, status: str, error: Optional[str] = None, content_sha: Optional[str] = None):
    return common.mark_source_status(source_id, status, error, content_sha)


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
        {"base_url": common.CDS_BASE_URL, "rg": 5},
    )
    _ensure(
        "arxiv_search",
        "Search arXiv (Atom API) for papers and return top results.",
        "arxiv_search",
        {"api_url": common.ARXIV_API_URL, "max_results": 5, "sortBy": "submittedDate", "sortOrder": "descending"},
    )
    _ensure(
        "inspirehep_search",
        "Search INSPIRE-HEP literature for high-energy physics papers and return top results.",
        "inspirehep_search",
        {"base_url": common.INSPIRE_BASE_URL, "size": 5},
    )
    _ensure(
        "web_search",
        "Search the web via SearxNG and return top results.",
        "searxng_search",
        {"base_url": common.SEARXNG_URL},
    )


@app.on_event("startup")
def _startup():
    ensure_default_tools()


# run_agent is delegated to tools.run_agent
def run_agent(user_id: str, role: str, messages: List[ChatMessage]) -> str:
    return tools_run_agent(user_id, role, messages)


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
    # If streaming and the local OpenGPTs clone is enabled, use the local
    # stream generator which yields SSE chunks. Otherwise fallback to the
    # non-streaming run_agent path.
    if req.stream and getattr(common, "USE_OPENGPTS", False):
        # Map incoming messages (Pydantic ChatMessage) to OpenGPTs-style dicts
        og_msgs = []
        for m in req.messages:
            if m.role == "user":
                og_msgs.append({"content": m.content, "type": "human"})
        assistant_id = getattr(common, "OPENGPTS_ASSISTANT_ID", "local-assistant")
        return StreamingResponse(
            stream_local_assistant(assistant_id, user_id, role, og_msgs), media_type="text/event-stream"
        )

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

    if len(body.urls) > common.MAX_URLS_PER_REQUEST:
        raise HTTPException(400, f"Too many URLs. Max is {common.MAX_URLS_PER_REQUEST}.")

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

    # Use mcp helper to fetch tools
    mcp_tools = get_mcp_tools(body.url, body.headers)
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
