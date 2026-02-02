# APIs not used anymore
# these APIs are deprecated and no longer used in the current application.
# They are replaced by the web interface and other mechanisms.

# =========================
# Tool management APIs
# =========================
@app.get("/api/tools")
def list_tools(
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
    x_authentik_groups: Optional[str] = Header(default=None, alias="X-Authentik-Groups"),
):
    role = get_user_role(x_authentik_groups, x_openwebui_user_role)
    rows = tool_list_for_role(role)
    return [
        {"id": r[0], "name": r[1], "description": r[2], "type": r[3], "enabled": r[5], "scope": r[6], "provider": r[7]}
        for r in rows
    ]


@app.post("/api/admin/tools")
def admin_create_tool(
    body: ToolCreate,
    x_openwebui_user_id: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Id"),
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
    x_authentik_email: Optional[str] = Header(default=None, alias="X-Authentik-Email"),
    x_authentik_groups: Optional[str] = Header(default=None, alias="X-Authentik-Groups"),
):
    role = get_user_role(x_authentik_groups, x_openwebui_user_role)
    assert_admin(role)
    created_by = get_user_id(x_authentik_email, x_openwebui_user_id)
    tid = str(uuid.uuid4())
    tool_upsert(tid, body, created_by=created_by)
    index_tool_desc(tid, body.name, body.description, body.enabled, body.scope)
    return {"id": tid, "name": body.name}


@app.patch("/api/admin/tools/{tool_id}")
def admin_patch_tool(
    tool_id: str,
    patch: ToolPatch,
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
    x_authentik_groups: Optional[str] = Header(default=None, alias="X-Authentik-Groups"),
):
    role = get_user_role(x_authentik_groups, x_openwebui_user_role)
    assert_admin(role)
    tool_patch(tool_id, patch)

    row = db_exec("SELECT id,name,description,enabled,scope FROM tools WHERE id=%s", (tool_id,))
    if row:
        _id, name, desc, enabled, scope = row[0]
        index_tool_desc(tool_id, name, desc, enabled, scope)
    return {"ok": True}


@app.get("/api/admin/tool-runs")
def admin_tool_runs(
    limit: int = Query(100, ge=1, le=500),
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
    x_authentik_groups: Optional[str] = Header(default=None, alias="X-Authentik-Groups"),
):
    role = get_user_role(x_authentik_groups, x_openwebui_user_role)
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
@app.post("/api/admin/ingest/urls")
def admin_ingest_urls(
    body: IngestUrlsRequest,
    x_openwebui_user_id: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Id"),
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
    x_authentik_email: Optional[str] = Header(default=None, alias="X-Authentik-Email"),
    x_authentik_groups: Optional[str] = Header(default=None, alias="X-Authentik-Groups"),
):
    role = get_user_role(x_authentik_groups, x_openwebui_user_role)
    user_id = get_user_id(x_authentik_email, x_openwebui_user_id)

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
@app.post("/api/admin/mcp/sync")
def admin_mcp_sync(
    body: MCPSyncRequest,
    x_openwebui_user_id: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Id"),
    x_openwebui_user_role: Optional[str] = Header(default=None, alias="X-OpenWebUI-User-Role"),
    x_authentik_email: Optional[str] = Header(default=None, alias="X-Authentik-Email"),
    x_authentik_groups: Optional[str] = Header(default=None, alias="X-Authentik-Groups"),
):
    role = get_user_role(x_authentik_groups, x_openwebui_user_role)
    assert_admin(role)
    created_by = get_user_id(x_authentik_email, x_openwebui_user_id)

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



