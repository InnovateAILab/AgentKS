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

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Admin UI (Flowbite-style)", version="0.1.0")

# Local static assets (no CDN)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

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
        r = db_exec("SELECT count(*) FROM rags") or [(0,)]
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
    sql = "SELECT id,name,scope,owner,doc_count,embed_model,updated_at FROM rags"
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


@app.on_event("startup")
def seed():
    # initialize DB schema and seed minimal demo data if tables are empty
    db_init()
    if not PG_DSN:
        return
    # seed urls
    r = db_exec("SELECT count(*) FROM urls") or [(0,)]
    if r[0][0] == 0:
        db_exec(
            "INSERT INTO urls (id,url,scope,tags,status,created_at) VALUES (%s,%s,%s,%s,%s,now())",
            ("u1", "https://example.com/doc1", "global", json.dumps(["physics", "note"]), "ingested"),
        )
        db_exec(
            "INSERT INTO urls (id,url,scope,tags,status,created_at) VALUES (%s,%s,%s,%s,%s,now())",
            ("u2", "https://example.com/doc2", "private", json.dumps(["personal"]), "queued"),
        )
    # seed mcps
    r = db_exec("SELECT count(*) FROM mcps") or [(0,)]
    if r[0][0] == 0:
        db_exec(
            "INSERT INTO mcps (id,name,endpoint,kind,tags,status,created_at) VALUES (%s,%s,%s,%s,%s,%s,now())",
            ("m1", "mcp-search", "http://mcp:8080", "http", json.dumps(["tools"]), "enabled"),
        )
    # seed rags
    r = db_exec("SELECT count(*) FROM rags") or [(0,)]
    if r[0][0] == 0:
        db_exec(
            "INSERT INTO rags (id,name,scope,owner,doc_count,embed_model,updated_at) VALUES (%s,%s,%s,%s,%s,%s,now())",
            ("r1", "global", "global", "", 1234, "nomic-embed-text"),
        )
        db_exec(
            "INSERT INTO rags (id,name,scope,owner,doc_count,embed_model,updated_at) VALUES (%s,%s,%s,%s,%s,%s,now())",
            ("r2", "private:user123", "private", "user123", 87, "nomic-embed-text"),
        )


@app.get("/admin/api/users")
def users(request: Request):
    """Placeholder user-management endpoint protected by admin group."""
    require_admin(request)
    return {"ok": True, "message": "user management endpoint (placeholder)"}
