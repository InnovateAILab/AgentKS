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
    urls_count = mcps_count = rags_count = 0
    if PG_DSN:
        r = db_exec("SELECT count(*) FROM urls") or [(0,)]
        urls_count = r[0][0]
        r = db_exec("SELECT count(*) FROM mcps") or [(0,)]
        mcps_count = r[0][0]
        # migrations define rag_groups as the logical collection table
        r = db_exec("SELECT count(*) FROM rag_groups") or [(0,)]
        rags_count = r[0][0]
    ctx = {
        "request": request,
        "user": user_from_headers(request),
        "counts": {"urls": urls_count, "mcps": mcps_count, "rags": rags_count},
    }
    return templates.TemplateResponse("home.html", ctx)


@app.get("/admin/urls", response_class=HTMLResponse)
def urls_list(
    request: Request, q: str | None = None, scope: str | None = None, status: str | None = None, tag: str | None = None
):
    # fetch from DB
    params: list = []
    where: list = []
    sql = "SELECT id,url,scope,tags,status,created_at FROM urls"
    if q:
        where.append("url ILIKE %s")
        params.append(f"%{q}%")
    if scope and scope != "all":
        where.append("scope = %s")
        params.append(scope)
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
        _id, url, _scope, tags, _status, created_at = r
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
            "status": _status,
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
def urls_add(request: Request, url: str = Form(...), scope: str = Form("global"), tags: str = Form("")):
    if not is_admin(request) and scope == "global":
        raise HTTPException(status_code=403, detail="admin group required to add global urls")
    item = {
        "id": str(uuid.uuid4())[:8],
        "url": url.strip(),
        "scope": scope,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "status": "queued",
        "created_at": now_iso(),
    }
    # persist
    if PG_DSN:
        db_exec(
            "INSERT INTO urls (id,url,scope,tags,status,created_at) VALUES (%s,%s,%s,%s,%s,now())",
            (item["id"], item["url"], item["scope"], json.dumps(item["tags"]), item["status"]),
        )
    return RedirectResponse(url="/admin/urls", status_code=303)


@app.get("/admin/mcps", response_class=HTMLResponse)
def mcps_list(request: Request, q: str | None = None, status: str | None = None, tag: str | None = None):
    params: list = []
    where: list = []
    sql = "SELECT id,name,endpoint,kind,tags,status,created_at FROM mcps"
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
        _id, name, endpoint, kind, tags, _status, created_at = r
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


@app.post("/admin/mcps/add")
def mcps_add(
    request: Request,
    name: str = Form(...),
    endpoint: str = Form(...),
    kind: str = Form("http"),
    tags: str = Form(""),
    status: str = Form("enabled"),
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
        "created_at": now_iso(),
    }
    if PG_DSN:
        db_exec(
            "INSERT INTO mcps (id,name,endpoint,kind,tags,status,created_at) VALUES (%s,%s,%s,%s,%s,%s,now())",
            (item["id"], item["name"], item["endpoint"], item["kind"], json.dumps(item["tags"]), item["status"]),
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


@app.get("/admin/api/users")
def users(request: Request):
    """Placeholder user-management endpoint protected by admin group."""
    require_admin(request)
    return {"ok": True, "message": "user management endpoint (placeholder)"}
