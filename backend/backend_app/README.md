# AgentKS — Backend

This folder contains the backend pieces for AgentKS: two FastAPI apps (the admin web UI and the RAG backend), Alembic migrations, and runtime tooling.

Overview
- admin web UI: `web/` — Flowbite-style FastAPI app that serves `/admin` (container port 8000).
- RAG backend: `app/` — RAG logic, tools, and OpenAI-compatible endpoints (container port 4000).
- Migrations: Alembic-managed migrations live under `migrations/versions/`.
- Startup: the backend image runs migrations at startup and then uses `supervisord` to run both FastAPI apps.

Quick start — full stack
1. Create a `.env` at the repo root with values referenced by `docker-compose.yml` (see suggested variables below).
2. From the repo root:

```bash
docker compose up --build
```

Run only the backend image (dev)

To build and run just the backend image (it will run migrations then start both apps via supervisord):

```bash
docker build -t agentks-backend ./backend/backend_app
docker run --rm -p 8000:8000 -p 4000:4000 --env-file .env agentks-backend
# admin UI: http://localhost:8000/admin
# backend API: http://localhost:4000/v1/models
```

Run the standalone admin app (fast check)

There is a legacy standalone admin app in `admin_app/` useful for quick checks:

```bash
docker build -t agentks-admin ./admin_app
docker run --rm -p 4000:4000 agentks-admin
# then open http://localhost:4000/admin
```

What happens on container start
- `startup.sh` runs `alembic -c /app/alembic.ini upgrade head` (with retries) to ensure schema + seed migrations are applied.
- On success, `supervisord` (configured in `supervisord.conf`) launches:
  - `uvicorn web.main:app --host 0.0.0.0 --port 8000` (admin UI)
  - `uvicorn app.main:app --host 0.0.0.0 --port 4000` (RAG backend)

Health & quick checks
- Admin health: GET http://localhost:8000/admin/api/health -> {"ok": true}
- Backend models: GET http://localhost:4000/v1/models

If your compose deployment proxies these services behind Caddy, test direct endpoints with Authentik headers when needed e.g.:

```bash
curl -H "X-Authentik-Groups: admin" http://localhost:8000/admin/api/health
```

Database & migrations
- Migrations: `migrations/versions/` (Alembic). The image runs `alembic upgrade head` at startup.
- Manually run migrations inside a running backend container:

```bash
docker compose exec rag-backend alembic -c /app/alembic.ini upgrade head
```

Or for a built image:

```bash
docker run --rm --env-file .env -it agentks-backend alembic -c /app/alembic.ini upgrade head
```

Important env vars (non-exhaustive)
- DATABASE_URL — SQLAlchemy DSN (postgresql+psycopg://user:pass@postgres:5432/dbname)
- OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL
- SEARXNG_URL, CDS_BASE_URL, ARXIV_API_URL, INSPIRE_BASE_URL
- COLLECTION_DOCS, COLLECTION_TOOLS (PGVector collection names)

See `docker-compose.yml` for the full set of variables used by the stack.

Where to look in the code
- Admin UI (templates, static, handlers): `backend/backend_app/web/`
- RAG backend & API: `backend/backend_app/app/`
- Migrations: `backend/backend_app/migrations/`
- Supervisord config: `backend/backend_app/supervisord.conf`
- Startup script: `backend/backend_app/startup.sh`

Development & debugging
- Tail logs for the whole stack:

```bash
docker compose logs -f
```

- Tail backend logs only:

```bash
docker compose logs -f rag-backend
```

- If you change Python code, rebuild the backend image or run locally for faster iteration.

Notes
- The `admin_app/` folder contains a small standalone admin UI useful for quick testing, but the backend image serves the admin UI alongside the RAG backend in production mode.
- Alembic migrations are the canonical source of schema and seed data; the app avoids runtime CREATE TABLE calls.

Next steps (optional)
- Add a `.env.example` listing common env vars referenced in `docker-compose.yml` (I can add this file on request).
- Consider converting the admin app to an `APIRouter` and including it in the RAG backend process if you prefer a single FastAPI process and unified OpenAPI docs.


Included files
- `.env.example` — example environment variables for `backend/backend_app` (copy to `.env` and edit locally).

If you'd like, I can also add a short troubleshooting checklist with exact psql/alembic commands to verify migrations.



