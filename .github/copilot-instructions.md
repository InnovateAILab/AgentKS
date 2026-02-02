## Quick context for AI code agents

This repository is an agentic RAG knowledge-stack composed of multiple services wired by Docker Compose and Caddy. Key pieces:

- Top-level orchestration: `docker-compose.yml` (services: `caddy`, `authentik_*`, `openwebui`, `admin_api`, `rag-backend`).
- HTTP ingress & auth: `Caddyfile` — Caddy uses forward_auth (Authentik) and injects headers `X-Authentik-Email`, `X-Authentik-Name`, `X-Authentik-Groups` into proxied requests.
- Admin API (FastAPI): `admin_app/` — builds as the `admin_api` service (exposes port 4000 in compose).
- Admin UI (FastAPI): `admin_app/app/` — templates and static assets are under `admin_app/app/templates` and `admin_app/app/static`.

Read these files first for accurate context: `README.md`, `docker-compose.yml`, `Caddyfile`, `admin_app/app/main.py`, and `backend/backend_app/app/main.py`.

## Big-picture architecture to keep in mind

- Caddy is the public router. Requests to `/admin*` are forwarded to `admin_api` only after forward_auth succeeds. Authentication metadata is delivered via headers — treat those as the canonical identity source for backend and UI code.
- The admin UI stores demo data in-memory; there is no persistent DB used by the admin UI code. Changes via the UI/API do not persist across restarts unless you add a persistence layer.
- `admin_app` is a simple FastAPI example showing header-based admin gating (`require_admin` reads `X-Authentik-Groups`).

## Developer workflows and useful commands

- Bring up the full stack for integration testing: run `docker compose up --build` from the repo root (see `docker-compose.yml`). This launches Caddy, Authentik, OpenWebUI, the admin_api service and the rag-backend.
- Run the admin UI locally (fast check) — build and run the `admin_app` Docker image:

```bash
docker build -t agentks-admin ./admin_app
docker run --rm -p 4000:4000 agentks-admin
```

then open: http://localhost:4000/admin

- Health endpoint: `GET /admin/api/health` returns `{"ok": true}`. Example (simulate trusted headers):

```bash
curl -H "X-Authentik-Groups: admin" http://localhost:4000/admin/api/health
```

## Important code patterns and conventions (concrete examples)

- Header-based identity: code expects Authentik to inject headers — see `admin_app/app/main.py`.
  - Use `request.headers.get("X-Authentik-Groups")` to check admin privileges. Example: `if "admin" not in groups.lower(): raise HTTPException(403)`

- In-memory collections (no persistence): demo resources are stored in Python lists in the admin UI handlers; when editing or extending endpoints, remember the lack of persistence and either add persistence intentionally or document ephemeral behavior.

- Template rendering: The UI uses Jinja2 templates in `admin_app/app/templates` and static assets in `admin_app/app/static`. Keep UI changes there.

## Integration points and external dependencies

- Authentik: required for forward_auth flow. Caddy forwards auth to `authentik_proxy`; identity info is shared via request headers as above.
- OpenWebUI: reverse proxied at `/webui` and configured to trust the same header names.
- Docker images and tags are configured in `docker-compose.yml` via environment vars — respect those when changing images.

## Safety and quick diagnostics

- To simulate an authenticated admin request locally, set the `X-Authentik-*` headers in curl or your browser proxy.
- Remember that running the full stack requires several env vars used by `docker-compose.yml` (e.g. DOMAIN, AUTH_DOMAIN, AUTHENTIK_* and AK_* variables). If missing, the stack may fail to start.

## Where to make changes

- API logic: `admin_app/app/main.py` (small, focused FastAPI app).
- UI pages and behavior: `admin_app/app/main.py`, `admin_app/app/templates/` and `admin_app/app/static/`.
- Backend agent & API: `backend/backend_app/app/main.py`.
- Orchestration and routing: `docker-compose.yml` and `Caddyfile`.

If any of the above assumptions are incomplete or you want me to expand the instructions (add run scripts, tests, or CI hints), tell me which area to flesh out and I will iterate.
