from __future__ import annotations

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uuid
import datetime

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Admin UI (Flowbite-style)", version="0.1.0")

# Local static assets (no CDN)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

URLS = []
MCPS = []
RAGS = []


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
    ctx = {
        "request": request,
        "user": user_from_headers(request),
        "counts": {"urls": len(URLS), "mcps": len(MCPS), "rags": len(RAGS)},
    }
    return templates.TemplateResponse("home.html", ctx)


@app.get("/admin/urls", response_class=HTMLResponse)
def urls_list(
    request: Request, q: str | None = None, scope: str | None = None, status: str | None = None, tag: str | None = None
):
    items = URLS[:]
    if q:
        items = [x for x in items if q.lower() in x["url"].lower()]
    if scope and scope != "all":
        items = [x for x in items if x["scope"] == scope]
    if status and status != "all":
        items = [x for x in items if x["status"] == status]
    if tag:
        items = [x for x in items if tag.lower() in ",".join(x["tags"]).lower()]
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
    URLS.insert(0, item)
    return RedirectResponse(url="/admin/urls", status_code=303)


@app.get("/admin/mcps", response_class=HTMLResponse)
def mcps_list(request: Request, q: str | None = None, status: str | None = None, tag: str | None = None):
    items = MCPS[:]
    if q:
        ql = q.lower()
        items = [x for x in items if ql in x["name"].lower() or ql in x["endpoint"].lower()]
    if status and status != "all":
        items = [x for x in items if x["status"] == status]
    if tag:
        items = [x for x in items if tag.lower() in ",".join(x["tags"]).lower()]
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
    MCPS.insert(0, item)
    return RedirectResponse(url="/admin/mcps", status_code=303)


@app.get("/admin/rags", response_class=HTMLResponse)
def rags_list(
    request: Request, q: str | None = None, scope: str | None = None, owner: str | None = None, embed: str | None = None
):
    items = RAGS[:]
    if q:
        items = [x for x in items if q.lower() in x["name"].lower()]
    if scope and scope != "all":
        items = [x for x in items if x["scope"] == scope]
    if owner and owner != "all":
        items = [x for x in items if owner.lower() in (x.get("owner") or "").lower()]
    if embed and embed != "all":
        items = [x for x in items if x.get("embed_model") == embed]
    embed_models = sorted({x.get("embed_model", "") for x in RAGS if x.get("embed_model")})
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
    if URLS or MCPS or RAGS:
        return
    URLS.extend(
        [
            {
                "id": "u1",
                "url": "https://example.com/doc1",
                "scope": "global",
                "tags": ["physics", "note"],
                "status": "ingested",
                "created_at": now_iso(),
            },
            {
                "id": "u2",
                "url": "https://example.com/doc2",
                "scope": "private",
                "tags": ["personal"],
                "status": "queued",
                "created_at": now_iso(),
            },
        ]
    )
    MCPS.append(
        {
            "id": "m1",
            "name": "mcp-search",
            "endpoint": "http://mcp:8080",
            "kind": "http",
            "tags": ["tools"],
            "status": "enabled",
            "created_at": now_iso(),
        }
    )
    RAGS.extend(
        [
            {
                "id": "r1",
                "name": "global",
                "scope": "global",
                "owner": "",
                "doc_count": 1234,
                "embed_model": "nomic-embed-text",
                "updated_at": now_iso(),
            },
            {
                "id": "r2",
                "name": "private:user123",
                "scope": "private",
                "owner": "user123",
                "doc_count": 87,
                "embed_model": "nomic-embed-text",
                "updated_at": now_iso(),
            },
        ]
    )


@app.get("/admin/api/users")
def users(request: Request):
    """Placeholder user-management endpoint protected by admin group."""
    require_admin(request)
    return {"ok": True, "message": "user management endpoint (placeholder)"}
