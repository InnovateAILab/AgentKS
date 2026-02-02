# AgentKS — agentic RAG knowledge stack

AgentKS wires together several services (via Docker Compose + Caddy) to provide an agentic RAG stack and a small admin UI/API for managing demo resources.

Key components
- Caddy (HTTP ingress + forward_auth)
- Authentik (authentication provider)
- OpenWebUI (web UI for models)
- admin_api (FastAPI app in `admin_app/`)
- admin UI (Flowbite-style FastAPI app in `admin_app/`)

Architecture highlights
- Caddy is the public router. It protects `/webui` and `/admin*` with Authentik via forward_auth and copies identity headers into proxied requests (see `Caddyfile`).
- Auth information is delivered via headers: `X-Authentik-Email`, `X-Authentik-Name`, `X-Authentik-Groups`. Services treat these as the canonical identity source.
- The admin UI (`admin_app/app/`) uses in-memory lists (`URLS`, `MCPS`, `RAGS`) — there is no persistence for those demo resources. The only persistent services in the compose are Authentik's Postgres/Redis.

Run locally (quick)
1. Create a `.env` with required variables used by `docker-compose.yml` (examples below).
2. From the repo root run:

```bash
docker-compose up --build
```

This will build and start Caddy, Authentik services, OpenWebUI, and the `admin_api` service (built from `admin_app/`).

Run just the static admin UI (fast check)

```bash
docker build -t admin-flowbite-local ./admin_flowbite_local_static
docker run --rm -p 4000:4000 admin-flowbite-local
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
- API logic: `admin_app/app.py` (FastAPI example, header-based gating in `require_admin`).
- UI templates & static: `admin_flowbite_local_static/app/templates` and `admin_flowbite_local_static/app/static` (Jinja2 + Flowbite-like assets).
- Admin UI behavior: `admin_flowbite_local_static/app/main.py` (in-memory collections and example `seed()` data).
- Orchestration & routing: `docker-compose.yml`, `Caddyfile`.

Debugging tips
- Simulate authenticated requests by setting `X-Authentik-*` headers in curl or API clients.
- View logs:

```bash
docker-compose logs -f caddy admin_api openwebui
```

- If you change Python code in `admin_app/` or `admin_flowbite_local_static/`, rebuild the relevant image(s) or run them locally with your interpreter for faster iteration.

Notes / constraints
- The admin UI uses ephemeral, in-memory collections for demo purposes — changes are lost on restart unless you add a persistence layer.
- The project assumes Authentik is the identity provider; Caddy forwards Authentik headers to downstream services.

Want edits?
If you'd like I can:
- Add a minimal `.env.example` listing all variables used by `docker-compose.yml`.
- Add a short "How to debug locally" section with concrete curl examples and a tiny pytest for the admin API.



