from __future__ import annotations

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uuid
import datetime
import os
import json

import psycopg
from starlette.middleware.sessions import SessionMiddleware
import requests
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Admin UI (Flowbite-style)", version="0.1.0")

# Local static assets (no CDN)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Session middleware for server-side flash messages and small ephemeral data.
# Use WEBUI_SECRET_KEY or SESSION_SECRET from the environment; fall back to a
# development secret (make sure to override in production).
_session_secret = os.getenv("WEBUI_SECRET_KEY") or os.getenv("SESSION_SECRET") or "dev-secret-change-me"
app.add_middleware(SessionMiddleware, secret_key=_session_secret, same_site="lax")

# Database setup — reuse repo DATABASE_URL when available
DATABASE_URL = os.getenv("DATABASE_URL")
PG_DSN = None
if DATABASE_URL:
    PG_DSN = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")


def db_exec(query: str, params: tuple = ()):  # simple helper
    if not PG_DSN:
        raise RuntimeError("DATABASE_URL not set for admin_app")
    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            try:
                return cur.fetchall()
            except psycopg.ProgrammingError:
                return None


def db_init():
    # Table creation moved to SQL files under backend/initdb/ which are mounted
    # into Postgres' /docker-entrypoint-initdb.d so the DB is initialized on
    # first start. Keep this function as a no-op at runtime.
    # If you need programmatic migrations later, replace this with a proper
    # migration check or integrate a migration tool (alembic, etc.).
    return


def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# Flash message helpers (store ephemeral messages in the signed session cookie)
def flash(request: Request, message: str, category: str = "success") -> None:
    sess = request.session
    flashes = sess.get("_flashes", [])
    flashes.append({"message": message, "category": category})
    sess["_flashes"] = flashes


def get_flashed_messages(request: Request) -> list:
    sess = request.session
    flashes = sess.pop("_flashes", [])
    return flashes


# expose helper to Jinja templates
templates.env.globals["get_flashed_messages"] = get_flashed_messages


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


@app.get("/admin/api/health")
def api_health():
    return {"ok": True}


@app.get("/admin", response_class=HTMLResponse)
def home(request: Request):
    # get counts from DB when available
    urls_count = mcps_count = rags_count = llms_count = 0
    if PG_DSN:
        r = db_exec("SELECT count(*) FROM urls") or [(0,)]
        urls_count = r[0][0]
        r = db_exec("SELECT count(*) FROM mcps") or [(0,)]
        mcps_count = r[0][0]
        # migrations define rag_groups as the logical collection table
        r = db_exec("SELECT count(*) FROM rag_groups") or [(0,)]
        rags_count = r[0][0]
        r = db_exec("SELECT count(*) FROM llms") or [(0,)]
        llms_count = r[0][0]
    ctx = {
        "request": request,
        "user": user_from_headers(request),
        "counts": {"urls": urls_count, "mcps": mcps_count, "rags": rags_count, "llms": llms_count},
    }
    return templates.TemplateResponse("home.html", ctx)


@app.get("/admin/urls", response_class=HTMLResponse)
def urls_list(
    request: Request, q: str | None = None, scope: str | None = None, status: str | None = None, tag: str | None = None
):
    # Fetch from source_urls table
    params: list = []
    where: list = []
    sql = "SELECT id,url,scope,tags,is_parent,discovery_status,discovered_count,created_at FROM source_urls"
    if q:
        where.append("url ILIKE %s")
        params.append(f"%{q}%")
    if scope and scope != "all":
        where.append("scope = %s")
        params.append(scope)
    if status and status != "all":
        # Map status filter to discovery_status
        where.append("discovery_status = %s")
        params.append(status)
    if tag:
        where.append("tags::text ILIKE %s")
        params.append(f"%{tag}%")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC"
    rows = db_exec(sql, tuple(params)) or []
    items = []
    for r in rows:
        _id, url, _scope, tags, is_parent, discovery_status, discovered_count, created_at = r
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        items.append({
            "id": _id,
            "url": url,
            "scope": _scope,
            "tags": tags or [],
            "status": discovery_status or "pending",  # Use discovery_status as status
            "is_parent": is_parent or False,
            "discovery_status": discovery_status or "pending",
            "discovered_count": discovered_count or 0,
            "created_at": created_at.isoformat() + "Z" if hasattr(created_at, "isoformat") else str(created_at),
        })
    ctx = {
        "request": request,
        "user": user_from_headers(request),
        "items": items,
        "filters": {"q": q or "", "scope": scope or "all", "status": status or "all", "tag": tag or ""},
    }
    return templates.TemplateResponse("urls_list.html", ctx)


@app.post("/admin/urls/bulk")
def urls_bulk(request: Request, selected: list[str] | None = Form(None), action: str = Form(...)):
    """Handle bulk actions for selected URLs: delete or refresh.

    - delete: remove URLs and related rag_group_urls and rag_documents linked to the url
    - refresh: mark the urls with status='refresh'
    """
    require_admin(request)
    if not selected:
        return RedirectResponse(url="/admin/urls", status_code=303)

    # Normalize selected ids
    ids = [s for s in selected if s]
    if not ids:
        return RedirectResponse(url="/admin/urls", status_code=303)

    # Build placeholders
    placeholders = ",".join(["%s"] * len(ids))

    try:
        if action == "delete":
            # Remove association rows and documents, then the url itself
            db_exec(f"DELETE FROM rag_group_urls WHERE url_id IN ({placeholders})", tuple(ids))
            db_exec(f"DELETE FROM rag_documents WHERE url_id IN ({placeholders})", tuple(ids))
            db_exec(f"DELETE FROM urls WHERE id IN ({placeholders})", tuple(ids))
        elif action == "refresh":
            db_exec(f"UPDATE urls SET status=%s WHERE id IN ({placeholders})", tuple(["refresh"] + ids))
        else:
            # Unknown action — ignore
            pass
    except Exception as e:
        flash(request, f"Bulk action failed: {e}", "error")
        return RedirectResponse(url="/admin/urls", status_code=303)

    # record audit and show success flash
    try:
        actor = user_from_headers(request).get("email") or user_from_headers(request).get("name")
        details = {"ids": ids}
        db_exec(
            "INSERT INTO admin_actions (id,action,actor,details) VALUES (%s,%s,%s,%s)",
            (str(uuid.uuid4()), f"urls_bulk_{action}", actor, json.dumps(details)),
        )
    except Exception:
        # don't block on audit failures — just continue
        pass

    flash(request, f"Bulk action '{action}' applied to {len(ids)} URL(s)", "success")
    return RedirectResponse(url="/admin/urls", status_code=303)


@app.get("/admin/urls/add", response_class=HTMLResponse)
def urls_add_form(request: Request):
    return templates.TemplateResponse("urls_add.html", {"request": request, "user": user_from_headers(request)})


@app.post("/admin/urls/add")
def urls_add(request: Request, url: str = Form(...), scope: str = Form("global"), tags: str = Form(""), is_parent: str = Form(None)):
    if not is_admin(request) and scope == "global":
        raise HTTPException(status_code=403, detail="admin group required to add global urls")
    
    # Convert is_parent checkbox value
    is_parent_bool = is_parent == "true" if is_parent else False
    
    item = {
        "id": str(uuid.uuid4())[:8],
        "url": url.strip(),
        "scope": scope,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "is_parent": is_parent_bool,
        "created_at": now_iso(),
    }
    # persist to source_urls table
    if PG_DSN:
        db_exec(
            "INSERT INTO source_urls (id,url,scope,tags,is_parent,created_at,created_by) VALUES (%s,%s,%s,%s,%s,now(),%s)",
            (item["id"], item["url"], item["scope"], json.dumps(item["tags"]), item["is_parent"], user_from_headers(request).get("email")),
        )
    return RedirectResponse(url="/admin/urls", status_code=303)


@app.get("/admin/urls/{source_url_id}/discovered", response_class=HTMLResponse)
def urls_discovered_view(request: Request, source_url_id: str):
    """View discovered URLs for a source URL (read-only)."""
    require_admin(request)
    
    # Fetch source URL info
    rows = db_exec("""
        SELECT id, url, scope, is_parent, discovery_status, discovered_count 
        FROM source_urls 
        WHERE id = %s
    """, (source_url_id,))
    
    if not rows:
        raise HTTPException(status_code=404, detail="Source URL not found")
    
    source = {
        "id": rows[0][0],
        "url": rows[0][1],
        "scope": rows[0][2],
        "is_parent": rows[0][3],
        "discovery_status": rows[0][4],
        "discovered_count": rows[0][5] or 0
    }
    
    # Fetch discovered URLs for this source
    discovered_rows = db_exec("""
        SELECT id, url, title, depth, status, chunks_count, last_fetched_at
        FROM discovered_urls
        WHERE source_url_id = %s
        ORDER BY depth, url
    """, (source_url_id,))
    
    discovered_urls = []
    for r in discovered_rows:
        discovered_urls.append({
            "id": r[0],
            "url": r[1],
            "title": r[2],
            "depth": r[3],
            "status": r[4],
            "chunks_count": r[5] or 0,
            "last_fetched_at": r[6].isoformat() + "Z" if r[6] and hasattr(r[6], "isoformat") else None
        })
    
    ctx = {
        "request": request,
        "user": user_from_headers(request),
        "source": source,
        "discovered_urls": discovered_urls
    }
    return templates.TemplateResponse("urls_discovered.html", ctx)


@app.get("/admin/mcps", response_class=HTMLResponse)
def mcps_list(request: Request, q: str | None = None, status: str | None = None, tag: str | None = None):
    params: list = []
    where: list = []
    # include description and resource columns (added via migration)
    sql = "SELECT id,name,endpoint,kind,description,resource,tags,status,created_at FROM mcps"
    if q:
        where.append("(name ILIKE %s OR endpoint ILIKE %s)")
        params.extend([f"%{q}%", f"%{q}%"])
    if status and status != "all":
        where.append("status = %s")
        params.append(status)
    if tag:
        where.append("tags::text ILIKE %s")
        params.append(f"%{tag}%")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC"
    rows = db_exec(sql, tuple(params)) or []
    items = []
    for r in rows:
        _id, name, endpoint, kind, description, resource, tags, _status, created_at = r
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        items.append({
            "id": _id,
            "name": name,
            "endpoint": endpoint,
            "kind": kind,
            "description": description,
            "resource": resource,
            "tags": tags or [],
            "status": _status,
            "created_at": created_at.isoformat() + "Z" if hasattr(created_at, "isoformat") else str(created_at),
        })
    ctx = {
        "request": request,
        "user": user_from_headers(request),
        "items": items,
        "filters": {"q": q or "", "status": status or "all", "tag": tag or ""},
    }
    return templates.TemplateResponse("mcps_list.html", ctx)


@app.post("/admin/mcps/bulk")
def mcps_bulk(request: Request, selected: list[str] | None = Form(None), action: str = Form(...)):
    """Handle bulk actions for selected MCPs: delete or refresh.

    - delete: remove tools attached to the MCP, then remove the MCP
    - refresh: mark tools related to these MCPs with a metadata flag {"refresh": true}
    """
    require_admin(request)
    if not selected:
        return RedirectResponse(url="/admin/mcps", status_code=303)

    ids = [s for s in selected if s]
    if not ids:
        return RedirectResponse(url="/admin/mcps", status_code=303)

    placeholders = ",".join(["%s"] * len(ids))
    try:
        if action == "delete":
            # delete tools that reference these mcps, then delete the mcps
            db_exec(f"DELETE FROM tools WHERE mcp_id IN ({placeholders})", tuple(ids))
            db_exec(f"DELETE FROM mcps WHERE id IN ({placeholders})", tuple(ids))
        elif action == "refresh":
            # add a refresh flag into tools.metadata JSONB
            # metadata = metadata || '{"refresh": true}'
            # Use a simple update for all matched tools
            db_exec(
                f"UPDATE tools SET metadata = COALESCE(metadata, '{{}}'::jsonb) || %s WHERE mcp_id IN ({placeholders})",
                tuple([json.dumps({"refresh": True})] + ids),
            )
        else:
            pass
    except Exception as e:
        flash(request, f"Bulk MCP action failed: {e}", "error")
        return RedirectResponse(url="/admin/mcps", status_code=303)

    # audit and flash success
    try:
        actor = user_from_headers(request).get("email") or user_from_headers(request).get("name")
        details = {"ids": ids}
        db_exec(
            "INSERT INTO admin_actions (id,action,actor,details) VALUES (%s,%s,%s,%s)",
            (str(uuid.uuid4()), f"mcps_bulk_{action}", actor, json.dumps(details)),
        )
    except Exception:
        pass

    flash(request, f"Bulk action '{action}' applied to {len(ids)} MCP(s)", "success")
    return RedirectResponse(url="/admin/mcps", status_code=303)


@app.get("/admin/mcps/add", response_class=HTMLResponse)
def mcps_add_form(request: Request):
    return templates.TemplateResponse("mcps_add.html", {"request": request, "user": user_from_headers(request)})


@app.post("/admin/mcps/discover")
def mcps_discover(
    request: Request,
    endpoint: str = Form(...),
    auth_type: str = Form(""),
    auth_token: str = Form(""),
    auth_headers: str = Form(""),
):
    """Discover MCP metadata by calling the provided endpoint using supplied auth.

    Attempts GET requests against the provided endpoint and a few common
    discovery paths. Returns discovered fields as JSON for the frontend to
    auto-fill the add form.
    """
    if not endpoint:
        return JSONResponse({"error": "endpoint required"}, status_code=400)

    headers = {}
    if auth_type and auth_token:
        t = auth_type.lower()
        if t == "bearer":
            headers["Authorization"] = f"Bearer {auth_token}"
        elif t == "basic":
            headers["Authorization"] = f"Basic {auth_token}"
    if auth_headers:
        try:
            extra = json.loads(auth_headers)
            if isinstance(extra, dict):
                headers.update(extra)
        except Exception:
            for line in auth_headers.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip()] = v.strip()

    candidates = [endpoint.rstrip("/"), endpoint.rstrip("/") + "/.well-known/mcp", endpoint.rstrip("/") + "/mcp", endpoint.rstrip("/") + "/tools"]
    for url in candidates:
        try:
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code != 200:
                continue
            try:
                data = r.json()
            except Exception:
                continue
            result = {}
            if isinstance(data, dict):
                for k in ("name", "title", "service"):
                    if k in data and data[k]:
                        result["name"] = data[k]
                        break
                if "description" in data:
                    result["description"] = data.get("description")
                if "tags" in data and isinstance(data["tags"], (list, tuple)):
                    result["tags"] = data["tags"]
                if "tools" in data and isinstance(data["tools"], list) and len(data["tools"]) > 0:
                    t0 = data["tools"][0]
                    if isinstance(t0, dict) and "name" in t0:
                        result.setdefault("name", t0.get("name"))
                if "resource" in data:
                    result["resource"] = data.get("resource")
                result.setdefault("raw", data)
                return JSONResponse({"ok": True, "data": result})
        except Exception:
            continue

    return JSONResponse({"ok": False, "error": "could not discover MCP metadata"}, status_code=502)


@app.post("/admin/mcps/add")
def mcps_add(
    request: Request,
    name: str = Form(...),
    endpoint: str = Form(...),
    kind: str = Form("http"),
    tags: str = Form(""),
    status: str = Form("enabled"),
    description: str = Form(""),
    resource: str = Form(""),
    context: str = Form(""),
    auth_type: str = Form(""),
    auth_token: str = Form(""),
    auth_headers: str = Form(""),
):
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="admin group required to add mcps")
    item = {
        "id": str(uuid.uuid4())[:8],
        "name": name.strip(),
        "endpoint": endpoint.strip(),
        "kind": kind,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "status": status,
        "description": description,
        "resource": resource,
        "context": context,
        "created_at": now_iso(),
    }
    # Build an auth JSON object if any auth form values were provided
    auth_obj = None
    if auth_type or auth_token or auth_headers:
        auth_obj = {}
        if auth_type:
            auth_obj["type"] = auth_type
        if auth_token:
            auth_obj["token"] = auth_token
        if auth_headers:
            try:
                auth_obj["headers"] = json.loads(auth_headers)
            except Exception:
                # Accept free-form header lines as fallback: KEY: value per-line
                hdrs = {}
                for line in auth_headers.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        hdrs[k.strip()] = v.strip()
                if hdrs:
                    auth_obj["headers"] = hdrs

    if PG_DSN:
        if auth_obj is not None:
            db_exec(
                "INSERT INTO mcps (id,name,endpoint,kind,description,resource,context,tags,status,auth,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())",
                (
                    item["id"],
                    item["name"],
                    item["endpoint"],
                    item["kind"],
                    item["description"],
                    item["resource"],
                    item["context"],
                    json.dumps(item["tags"]),
                    item["status"],
                    json.dumps(auth_obj),
                ),
            )
        else:
            db_exec(
                "INSERT INTO mcps (id,name,endpoint,kind,description,resource,context,tags,status,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,now())",
                (
                    item["id"],
                    item["name"],
                    item["endpoint"],
                    item["kind"],
                    item["description"],
                    item["resource"],
                    item["context"],
                    json.dumps(item["tags"]),
                    item["status"],
                ),
            )
    return RedirectResponse(url="/admin/mcps", status_code=303)


@app.get("/admin/rags", response_class=HTMLResponse)
def rags_list(
    request: Request, q: str | None = None, scope: str | None = None, owner: str | None = None, embed: str | None = None
):
    params: list = []
    where: list = []
    # use rag_groups (defined in alembic 0001_initial) which stores RAG collections
    sql = "SELECT id,name,scope,owner,doc_count,embed_model,updated_at FROM rag_groups"
    if q:
        where.append("name ILIKE %s")
        params.append(f"%{q}%")
    if scope and scope != "all":
        where.append("scope = %s")
        params.append(scope)
    if owner and owner != "all":
        where.append("owner ILIKE %s")
        params.append(f"%{owner}%")
    if embed and embed != "all":
        where.append("embed_model = %s")
        params.append(embed)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at DESC"
    rows = db_exec(sql, tuple(params)) or []
    items = []
    embed_models_set = set()
    for r in rows:
        _id, name, _scope, owner, doc_count, embed_model, updated_at = r
        if embed_model:
            embed_models_set.add(embed_model)
        items.append(
            {
                "id": _id,
                "name": name,
                "scope": _scope,
                "owner": owner,
                "doc_count": doc_count,
                "embed_model": embed_model,
                "updated_at": updated_at.isoformat() + "Z" if hasattr(updated_at, "isoformat") else str(updated_at),
            }
        )
    embed_models = sorted(embed_models_set)
    ctx = {
        "request": request,
        "user": user_from_headers(request),
        "items": items,
        "filters": {"q": q or "", "scope": scope or "all", "owner": owner or "all", "embed": embed or "all"},
        "embed_models": embed_models,
    }
    return templates.TemplateResponse("rags_list.html", ctx)


@app.post("/admin/rags/bulk")
def rags_bulk(request: Request, selected: list[str] | None = Form(None), action: str = Form(...)):
    """Handle bulk actions for rag_groups: delete or refresh.

    - delete: remove rag_group_urls, rag_documents for the selected rag_group ids, then delete rag_groups
    - refresh: set updated_at = now() for chosen rag_groups (mark them for re-indexing/refresh)
    """
    require_admin(request)
    if not selected:
        return RedirectResponse(url="/admin/rags", status_code=303)

    ids = [s for s in selected if s]
    if not ids:
        return RedirectResponse(url="/admin/rags", status_code=303)

    placeholders = ",".join(["%s"] * len(ids))
    try:
        if action == "delete":
            # delete documents and associations, then rag_groups
            db_exec(f"DELETE FROM rag_group_urls WHERE rag_group_id IN ({placeholders})", tuple(ids))
            db_exec(f"DELETE FROM rag_documents WHERE rag_group_id IN ({placeholders})", tuple(ids))
            db_exec(f"DELETE FROM rag_groups WHERE id IN ({placeholders})", tuple(ids))
        elif action == "refresh":
            db_exec(f"UPDATE rag_groups SET updated_at = now() WHERE id IN ({placeholders})", tuple(ids))
        else:
            pass
    except Exception as e:
        flash(request, f"Bulk RAG action failed: {e}", "error")
        return RedirectResponse(url="/admin/rags", status_code=303)

    # audit and flash success
    try:
        actor = user_from_headers(request).get("email") or user_from_headers(request).get("name")
        details = {"ids": ids}
        db_exec(
            "INSERT INTO admin_actions (id,action,actor,details) VALUES (%s,%s,%s,%s)",
            (str(uuid.uuid4()), f"rags_bulk_{action}", actor, json.dumps(details)),
        )
    except Exception:
        pass

    flash(request, f"Bulk action '{action}' applied to {len(ids)} RAG group(s)", "success")
    return RedirectResponse(url="/admin/rags", status_code=303)


@app.on_event("startup")
def seed():
    # Schema creation and demo data are now managed by Alembic migrations.
    # No runtime seeding is performed here to avoid race conditions.
    return


# =============================================================================
# LLM Management Routes
# =============================================================================

@app.get("/admin/llms", response_class=HTMLResponse)
def llms_list(request: Request, q: str | None = None, provider: str | None = None, enabled: str | None = None):
    """List all LLMs with filtering"""
    params: list = []
    where: list = []
    sql = "SELECT id, name, provider, model_name, description, enabled, is_default, priority, created_at FROM llms"
    
    if q:
        where.append("(name ILIKE %s OR model_name ILIKE %s OR description ILIKE %s)")
        params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
    if provider and provider != "all":
        where.append("provider = %s")
        params.append(provider)
    if enabled and enabled != "all":
        where.append("enabled = %s")
        params.append(enabled == "true")
    
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY priority ASC, name ASC"
    
    rows = db_exec(sql, tuple(params)) or []
    items = []
    for row in rows:
        items.append({
            "id": row[0],
            "name": row[1],
            "provider": row[2],
            "model_name": row[3],
            "description": row[4],
            "enabled": row[5],
            "is_default": row[6],
            "priority": row[7],
            "created_at": row[8].isoformat() if row[8] else None
        })
    
    # Get available providers
    providers = db_exec("SELECT DISTINCT provider FROM llms ORDER BY provider") or []
    provider_list = [p[0] for p in providers]
    
    ctx = {
        "request": request,
        "user": user_from_headers(request),
        "items": items,
        "providers": provider_list,
        "filters": {"q": q or "", "provider": provider or "all", "enabled": enabled or "all"},
        "flash": get_flashed_messages(request)
    }
    return templates.TemplateResponse("llms_list.html", ctx)


@app.get("/admin/llms/add", response_class=HTMLResponse)
def llms_add_form(request: Request):
    """Show form to add a new LLM"""
    ctx = {
        "request": request,
        "user": user_from_headers(request),
        "flash": get_flashed_messages(request)
    }
    return templates.TemplateResponse("llms_add.html", ctx)


@app.post("/admin/llms/add")
def llms_add(
    request: Request,
    name: str = Form(...),
    provider: str = Form(...),
    model_name: str = Form(...),
    description: str = Form(""),
    auth_meta: str = Form("{}"),
    config: str = Form("{}"),
    enabled: bool = Form(False),
    is_default: bool = Form(False),
    priority: int = Form(100)
):
    """Add a new LLM to the database"""
    try:
        # Validate JSON fields
        try:
            auth_meta_json = json.loads(auth_meta) if auth_meta else {}
            config_json = json.loads(config) if config else {}
        except json.JSONDecodeError as e:
            flash(request, f"Invalid JSON: {e}", "error")
            return RedirectResponse(url="/admin/llms/add", status_code=303)
        
        llm_id = str(uuid.uuid4())
        
        # If setting as default, unset other defaults first
        if is_default:
            db_exec("UPDATE llms SET is_default = false WHERE scope = 'global'")
        
        db_exec(
            """INSERT INTO llms (id, name, provider, model_name, description, auth_meta, config, 
               enabled, is_default, priority, scope) 
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (llm_id, name, provider, model_name, description, json.dumps(auth_meta_json),
             json.dumps(config_json), enabled, is_default, priority, 'global')
        )
        
        # Audit log
        try:
            actor = user_from_headers(request).get("email") or user_from_headers(request).get("name")
            db_exec(
                "INSERT INTO admin_actions (id, action, actor, details) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), "llm_add", actor, json.dumps({"llm_id": llm_id, "name": name}))
            )
        except Exception:
            pass
        
        flash(request, f"LLM '{name}' added successfully", "success")
        return RedirectResponse(url="/admin/llms", status_code=303)
    
    except Exception as e:
        flash(request, f"Error adding LLM: {e}", "error")
        return RedirectResponse(url="/admin/llms/add", status_code=303)


@app.get("/admin/llms/{llm_id}/edit", response_class=HTMLResponse)
def llms_edit_form(request: Request, llm_id: str):
    """Show form to edit an existing LLM"""
    row = db_exec("SELECT id, name, provider, model_name, description, auth_meta, config, enabled, is_default, priority FROM llms WHERE id = %s", (llm_id,))
    if not row:
        flash(request, "LLM not found", "error")
        return RedirectResponse(url="/admin/llms", status_code=303)
    
    llm = {
        "id": row[0][0],
        "name": row[0][1],
        "provider": row[0][2],
        "model_name": row[0][3],
        "description": row[0][4],
        "auth_meta": json.dumps(row[0][5], indent=2) if row[0][5] else "{}",
        "config": json.dumps(row[0][6], indent=2) if row[0][6] else "{}",
        "enabled": row[0][7],
        "is_default": row[0][8],
        "priority": row[0][9]
    }
    
    ctx = {
        "request": request,
        "user": user_from_headers(request),
        "llm": llm,
        "flash": get_flashed_messages(request)
    }
    return templates.TemplateResponse("llms_edit.html", ctx)


@app.post("/admin/llms/{llm_id}/edit")
def llms_edit(
    request: Request,
    llm_id: str,
    name: str = Form(...),
    provider: str = Form(...),
    model_name: str = Form(...),
    description: str = Form(""),
    auth_meta: str = Form("{}"),
    config: str = Form("{}"),
    enabled: bool = Form(False),
    is_default: bool = Form(False),
    priority: int = Form(100)
):
    """Update an existing LLM"""
    try:
        # Validate JSON fields
        try:
            auth_meta_json = json.loads(auth_meta) if auth_meta else {}
            config_json = json.loads(config) if config else {}
        except json.JSONDecodeError as e:
            flash(request, f"Invalid JSON: {e}", "error")
            return RedirectResponse(url=f"/admin/llms/{llm_id}/edit", status_code=303)
        
        # If setting as default, unset other defaults first
        if is_default:
            db_exec("UPDATE llms SET is_default = false WHERE scope = 'global' AND id != %s", (llm_id,))
        
        db_exec(
            """UPDATE llms SET name = %s, provider = %s, model_name = %s, description = %s,
               auth_meta = %s, config = %s, enabled = %s, is_default = %s, priority = %s,
               updated_at = now() WHERE id = %s""",
            (name, provider, model_name, description, json.dumps(auth_meta_json),
             json.dumps(config_json), enabled, is_default, priority, llm_id)
        )
        
        # Audit log
        try:
            actor = user_from_headers(request).get("email") or user_from_headers(request).get("name")
            db_exec(
                "INSERT INTO admin_actions (id, action, actor, details) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), "llm_edit", actor, json.dumps({"llm_id": llm_id, "name": name}))
            )
        except Exception:
            pass
        
        flash(request, f"LLM '{name}' updated successfully", "success")
        return RedirectResponse(url="/admin/llms", status_code=303)
    
    except Exception as e:
        flash(request, f"Error updating LLM: {e}", "error")
        return RedirectResponse(url=f"/admin/llms/{llm_id}/edit", status_code=303)


@app.post("/admin/llms/{llm_id}/delete")
def llms_delete(request: Request, llm_id: str):
    """Delete an LLM"""
    try:
        # Get LLM name for audit log
        row = db_exec("SELECT name FROM llms WHERE id = %s", (llm_id,))
        llm_name = row[0][0] if row else "Unknown"
        
        db_exec("DELETE FROM llms WHERE id = %s", (llm_id,))
        
        # Audit log
        try:
            actor = user_from_headers(request).get("email") or user_from_headers(request).get("name")
            db_exec(
                "INSERT INTO admin_actions (id, action, actor, details) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), "llm_delete", actor, json.dumps({"llm_id": llm_id, "name": llm_name}))
            )
        except Exception:
            pass
        
        flash(request, f"LLM '{llm_name}' deleted successfully", "success")
    except Exception as e:
        flash(request, f"Error deleting LLM: {e}", "error")
    
    return RedirectResponse(url="/admin/llms", status_code=303)


@app.post("/admin/llms/{llm_id}/toggle")
def llms_toggle(request: Request, llm_id: str):
    """Toggle LLM enabled/disabled status"""
    try:
        row = db_exec("SELECT name, enabled FROM llms WHERE id = %s", (llm_id,))
        if not row:
            flash(request, "LLM not found", "error")
            return RedirectResponse(url="/admin/llms", status_code=303)
        
        llm_name = row[0][0]
        current_status = row[0][1]
        new_status = not current_status
        
        db_exec("UPDATE llms SET enabled = %s, updated_at = now() WHERE id = %s", (new_status, llm_id))
        
        # Audit log
        try:
            actor = user_from_headers(request).get("email") or user_from_headers(request).get("name")
            db_exec(
                "INSERT INTO admin_actions (id, action, actor, details) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), "llm_toggle", actor, 
                 json.dumps({"llm_id": llm_id, "name": llm_name, "enabled": new_status}))
            )
        except Exception:
            pass
        
        status_text = "enabled" if new_status else "disabled"
        flash(request, f"LLM '{llm_name}' {status_text}", "success")
    except Exception as e:
        flash(request, f"Error toggling LLM: {e}", "error")
    
    return RedirectResponse(url="/admin/llms", status_code=303)


@app.post("/admin/llms/{llm_id}/set-default")
def llms_set_default(request: Request, llm_id: str):
    """Set an LLM as the default"""
    try:
        row = db_exec("SELECT name FROM llms WHERE id = %s", (llm_id,))
        if not row:
            flash(request, "LLM not found", "error")
            return RedirectResponse(url="/admin/llms", status_code=303)
        
        llm_name = row[0][0]
        
        # Unset all defaults, then set this one
        db_exec("UPDATE llms SET is_default = false WHERE scope = 'global'")
        db_exec("UPDATE llms SET is_default = true, updated_at = now() WHERE id = %s", (llm_id,))
        
        # Audit log
        try:
            actor = user_from_headers(request).get("email") or user_from_headers(request).get("name")
            db_exec(
                "INSERT INTO admin_actions (id, action, actor, details) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), "llm_set_default", actor, json.dumps({"llm_id": llm_id, "name": llm_name}))
            )
        except Exception:
            pass
        
        flash(request, f"LLM '{llm_name}' set as default", "success")
    except Exception as e:
        flash(request, f"Error setting default LLM: {e}", "error")
    
    return RedirectResponse(url="/admin/llms", status_code=303)


@app.post("/admin/llms/bulk")
def llms_bulk_action(request: Request, action: str = Form(...), ids: str = Form(...)):
    """Bulk actions for LLMs: enable, disable, delete"""
    try:
        id_list = [i.strip() for i in ids.split(",") if i.strip()]
        if not id_list:
            flash(request, "No LLMs selected", "warning")
            return RedirectResponse(url="/admin/llms", status_code=303)
        
        placeholders = ",".join(["%s"] * len(id_list))
        
        if action == "enable":
            db_exec(f"UPDATE llms SET enabled = true, updated_at = now() WHERE id IN ({placeholders})", tuple(id_list))
        elif action == "disable":
            db_exec(f"UPDATE llms SET enabled = false, updated_at = now() WHERE id IN ({placeholders})", tuple(id_list))
        elif action == "delete":
            db_exec(f"DELETE FROM llms WHERE id IN ({placeholders})", tuple(id_list))
        else:
            flash(request, f"Unknown action: {action}", "error")
            return RedirectResponse(url="/admin/llms", status_code=303)
        
        # Audit log
        try:
            actor = user_from_headers(request).get("email") or user_from_headers(request).get("name")
            db_exec(
                "INSERT INTO admin_actions (id, action, actor, details) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), f"llms_bulk_{action}", actor, json.dumps({"ids": id_list}))
            )
        except Exception:
            pass
        
        flash(request, f"Bulk action '{action}' applied to {len(id_list)} LLM(s)", "success")
    except Exception as e:
        flash(request, f"Bulk action failed: {e}", "error")
    
    return RedirectResponse(url="/admin/llms", status_code=303)


# =============================================================================
# End of LLM Management Routes
# =============================================================================


@app.get("/admin/api/users")
def users(request: Request):
    """Placeholder user-management endpoint protected by admin group."""
    require_admin(request)
    return {"ok": True, "message": "user management endpoint (placeholder)"}

