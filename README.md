# AgentKS — agentic RAG knowledge stack

AgentKS wires together several services (via Docker Compose + Caddy) to provide an agentic RAG stack and a small admin UI/API for managing demo resources.

Key components
- Caddy (HTTP ingress + forward_auth)
- Authentik (authentication provider)
---
# AgentKS — agentic RAG knowledge stack

AgentKS wires together several services (via Docker Compose + Caddy) to provide an agentic RAG stack and a small admin UI/API for managing demo resources.

Key components
- Caddy (HTTP ingress + forward_auth)
- Authentik (authentication provider)
- OpenWebUI (web UI for models)
- admin UI (Flowbite-style FastAPI app in `admin_app/`)
- rag backend (FastAPI app in `backend_app/`) with multiple services:
  - web (8000) - Main web API
  - app (4000) - Admin UI
  - rag_mcp (4001) - RAG MCP retrieval service
  - rag_injector (4002) - RAG injection REST service
  - url_watcher - Background daemon
- basic_tools_mcp (5000) - General utility MCP service

Architecture highlights
- Caddy is the public router. It protects `/webui`, `/admin*` and `/api` with Authentik via forward_auth and copies identity headers into proxied requests (see `Caddyfile`).
- Auth information is delivered via headers: `X-Authentik-Email`, `X-Authentik-Name`, `X-Authentik-Groups`. Services treat these as the canonical identity source.
- The admin UI is implemented in `admin_app/app/main.py` and serves the admin pages under `/admin`.
- The backend API lives in `backend/backend_app/app/main.py` and exposes `/api` (and an OpenAI-compatible `/v1/*` endpoint).

Run locally (quick)
1. Create a `.env` with required variables used by `docker-compose.yml` (examples below).
2. From the repo root run:

```bash
docker compose up --build
```

Run backend-only (local dev)
--------------------------------

If you only want to run the backend and its local dependencies (OLLAMA, SearxNG) for development, use the backend-local compose file:

```bash
docker compose -f backend/docker-compose.local.yml up --build
```

Note: the repository root `docker-compose.yml` is the recommended entrypoint for running the full stack (Caddy + Authentik + OpenWebUI + backend).

Run the admin UI locally (fast check)

The admin UI is a small FastAPI app in `admin_app/` and can be run using Docker:

```bash
docker build -t agentks-admin ./admin_app
docker run --rm -p 4000:4000 agentks-admin
# then open: http://localhost:4000/admin
```

Health & quick checks
- Admin API health: `GET /admin/api/health` -> `{"ok": true}`. If the admin UI/API are proxied behind Caddy, simulate headers when calling directly:

```bash
curl -H "X-Authentik-Groups: admin" http://localhost:4000/admin/api/health
```

Important environment variables (examples — check `docker-compose.yml` for full list)
- DOMAIN, AUTH_DOMAIN  # used by Caddy and Authentik host config
- AUTHENTIK_TAG, AUTHENTIK_SECRET_KEY, AUTHENTIK_OUTPOST_TOKEN
- AK_POSTGRES_DB, AK_POSTGRES_USER, AK_POSTGRES_PASSWORD, AK_REDIS_PASSWORD
- WEBUI_AUTH (used by OpenWebUI)

Where to look / change code
- API logic: `admin_app/app/main.py` (FastAPI example, header-based gating in `require_admin`).
- UI templates & static: `admin_app/app/templates` and `admin_app/app/static` (Jinja2 + simple static assets).
- Admin UI behavior: `admin_app/app/main.py` (in-memory collections and example `seed()` data).
- Backend API & agent: `backend/backend_app/app/main.py` (RAG, tools, agent orchestration).
- Orchestration & routing: `docker-compose.yml`, `Caddyfile`.

Debugging tips
- Simulate authenticated requests by setting `X-Authentik-*` headers in curl or API clients.
- View logs:

```bash
docker compose logs -f caddy admin_api openwebui rag-backend
```

- If you change Python code in `admin_app/` or `backend/backend_app/`, rebuild the relevant image(s) or run them locally with your interpreter for faster iteration.

Compose service -> directory mapping
- `admin_api` (service) -> `./admin_app` (folder with Dockerfile)
- `rag-backend` (service) -> `./backend_app` (folder with Dockerfile)

When using `docker compose logs` refer to the service names defined in `docker-compose.yml` (for example: `admin_api`, `rag-backend`).

Notes / constraints
- The admin UI uses ephemeral, in-memory collections for demo purposes — changes are lost on restart unless you add a persistence layer.
- The project assumes Authentik is the identity provider; Caddy forwards Authentik headers to downstream services.

Want edits?
If you'd like I can:
- Add a minimal `.env.example` listing all variables used by `docker-compose.yml`.
- Add a short "How to debug locally" section with concrete curl examples and a tiny pytest for the admin API.



