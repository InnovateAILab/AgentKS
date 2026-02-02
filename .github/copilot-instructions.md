## Quick context for AI code agents

This repository is an agentic RAG knowledge-stack composed of multiple services wired by Docker Compose and Caddy. Key pieces:

- Top-level orchestration: `docker-compose.yml` (services: `caddy`, `authentik_*`, `openwebui`, `admin_api`).
- HTTP ingress & auth: `Caddyfile` — Caddy uses forward_auth (Authentik) and injects headers `X-Authentik-Email`, `X-Authentik-Name`, `X-Authentik-Groups` into proxied requests.
- Admin API (FastAPI): `admin_app/` — builds as the `admin_api` service (exposes port 4000 in compose).
- Admin UI (static Flowbite-like FastAPI): `admin_flowbite_local_static/app/` — serves templates and static assets from `/static` and `/templates`.

Read these files first for accurate context: `README.md`, `docker-compose.yml`, `Caddyfile`, `admin_app/app.py`, `admin_flowbite_local_static/app/main.py`, and `admin_flowbite_local_static/README.md`.

## Big-picture architecture to keep in mind

- Caddy is the public router. Requests to `/admin*` are forwarded to `admin_api` only after forward_auth succeeds. Authentication metadata is delivered via headers — treat those as the canonical identity source for backend and UI code.
- There is no persistent DB used by the admin UI code in this repo: `admin_flowbite_local_static/app/main.py` stores `URLS`, `MCPS`, and `RAGS` in memory. Changes via the UI/API do not persist across restarts.
- `admin_app` is a simple FastAPI example showing header-based admin gating (`require_admin` reads `X-Authentik-Groups`).

## Developer workflows and useful commands

- Bring up the full stack for integration testing: run `docker-compose up --build` from the repo root (see `docker-compose.yml`). This launches Caddy, Authentik, OpenWebUI and the `admin_api` service.
- Run the local admin UI container (flowbite static) for quick UI checks:

  docker build -t admin-flowbite-local ./admin_flowbite_local_static
  docker run --rm -p 4000:4000 admin-flowbite-local

  then open: http://localhost:4000/admin

- Health endpoints: `GET /admin/api/health` on whichever admin service is running (e.g. port 4000) returns `{"ok": true}`. Example (simulate trusted headers):

  curl -H "X-Authentik-Groups: admin" http://localhost:4000/admin/api/health

## Important code patterns and conventions (concrete examples)

- Header-based identity: code expects Authentik to inject headers — see `admin_app/app.py` and `admin_flowbite_local_static/app/main.py`.
  - Use `request.headers.get("X-Authentik-Groups")` to check admin privileges. Example: `if "admin" not in groups.lower(): raise HTTPException(403)`

- In-memory collections (no persistence): `URLS`, `MCPS`, `RAGS` are Python lists mutated in handlers in `admin_flowbite_local_static/app/main.py`. When editing or extending endpoints, remember the lack of persistence and either add persistence intentionally or document ephemeral behavior.

- Template rendering: The UI uses Jinja2 templates in `admin_flowbite_local_static/app/templates` and static assets in `admin_flowbite_local_static/app/static`. Keep UI changes there.

## Integration points and external dependencies

- Authentik: required for forward_auth flow. Caddy forwards auth to `authentik_proxy`; identity info is shared via request headers as above.
- OpenWebUI: reverse proxied at `/webui` and configured to trust the same header names.
- Docker images and tags are configured in `docker-compose.yml` via environment vars — respect those when changing images.

## Safety and quick diagnostics

- To simulate an authenticated admin request locally, set the `X-Authentik-*` headers in curl or your browser proxy.
- Remember that running the full stack requires several env vars used by `docker-compose.yml` (e.g. DOMAIN, AUTH_DOMAIN, AUTHENTIK_* and AK_* variables). If missing, the stack may fail to start.

## Where to make changes

- API logic: `admin_app/app.py` (small, focused FastAPI app).
- UI pages and behavior: `admin_flowbite_local_static/app/main.py`, `templates/` and `static/` under the same folder.
- Orchestration and routing: `docker-compose.yml` and `Caddyfile`.

If any of the above assumptions are incomplete or you want me to expand the instructions (add run scripts, tests, or CI hints), tell me which area to flesh out and I will iterate.
