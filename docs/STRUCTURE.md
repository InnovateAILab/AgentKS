# AgentKS — Full Repository Structure

Complete directory structure and component organization for the AgentKS agentic RAG knowledge stack.

## Repository Tree

```
AgentKS/
├── docs/                           # Documentation
│   ├── AGENT_FLOW.md              # Multi-skill agent architecture
│   └── STRUCTURE.md               # This file - full structure diagram
│
├── backend/                        # Backend services
│   ├── backend_app/               # Main backend application
│   │   ├── app/                   # Agent backend (port 4000)
│   │   │   ├── main.py           # FastAPI app, OpenAI-compatible API
│   │   │   ├── agent_skill.py    # LangGraph orchestrator
│   │   │   ├── rag_skill.py      # RAG retrieval skill
│   │   │   ├── tools_skill.py    # Tool discovery skill
│   │   │   ├── llms.py           # LLM configuration
│   │   │   ├── RAG_SKILL_GUIDE.md
│   │   │   ├── TOOLS_SKILL_GUIDE.md
│   │   │   └── backup/           # Archived code
│   │   │       └── langgraph_adapter.py.bak  # Old adapter (replaced by agent_skill.py)
│   │   │
│   │   ├── web/                  # Admin UI (port 8000)
│   │   │   ├── main.py           # FastAPI admin interface
│   │   │   ├── templates/        # Jinja2 templates
│   │   │   │   ├── base.html
│   │   │   │   ├── home.html
│   │   │   │   ├── mcps_list.html
│   │   │   │   ├── rags_list.html
│   │   │   │   └── urls_list.html
│   │   │   └── static/           # CSS, JS assets
│   │   │       ├── app.js
│   │   │       └── styles.css
│   │   │
│   │   ├── rag/                  # RAG services
│   │   │   ├── rag_mcp/          # MCP server (port 4001)
│   │   │   │   └── main.py       # RAG retrieval service
│   │   │   ├── rag_injector/     # REST API (port 4002)
│   │   │   │   └── main.py       # Document injection
│   │   │   ├── rag_common.py     # Shared utilities
│   │   │   └── daemons/          # Background processing
│   │   │
│   │   ├── tools/                # MCP tool integration
│   │   │   ├── __init__.py
│   │   │   ├── client.py         # MCP client
│   │   │   ├── tool_discovery.py # Semantic/hybrid search
│   │   │   ├── watcher.py        # mcp_watcher daemon
│   │   │   ├── models.py         # Data models
│   │   │   ├── discovery.py      # Discovery algorithms
│   │   │   ├── README.md
│   │   │   ├── TOOL_DISCOVERY_GUIDE.md
│   │   │   ├── MCP_WATCHER_ARCHITECTURE.md
│   │   │   ├── MCP_WATCHER_MIGRATION.md
│   │   │   ├── MCP_WATCHER_QUICKREF.md
│   │   │   └── MCP_WATCHER_SUMMARY.md
│   │   │
│   │   ├── migrations/           # Alembic database migrations
│   │   │   ├── versions/         # Migration scripts
│   │   │   └── env.py
│   │   │
│   │   ├── daemons/              # (Empty - moved to rag/daemons)
│   │   │
│   │   ├── supervisord.conf      # Multi-service orchestration
│   │   ├── startup.sh            # Container init script
│   │   ├── Dockerfile            # Multi-stage build
│   │   ├── requirements.txt      # Python dependencies
│   │   ├── alembic.ini           # Alembic config
│   │   ├── README.md             # Backend documentation
│   │   ├── LLM_MANAGEMENT.md
│   │   ├── USING_LLMS.md
│   │   ├── URL_HIERARCHY_IMPLEMENTATION.md
│   │   ├── database.md
│   │   └── .env.example
│   │
│   ├── basic_tools_mcp_service/  # MCP tool service (port 5000)
│   │   ├── main.py               # Search tools server
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── searxng/                  # SearXNG configuration
│   │   └── settings.yml
│   │
│   ├── docker-compose.local.yml  # Backend-only compose
│   └── alembic.ini               # Alembic config (backend root)
│
├── .github/                       # GitHub specific files
│   └── copilot-instructions.md   # AI agent context
│
├── docker-compose.yml             # Full stack orchestration
├── Caddyfile                      # HTTP ingress & routing
├── README.md                      # This project overview
├── .env.example                   # Environment template
├── Makefile                       # Build & run shortcuts
├── pyproject.toml                 # Python project config
├── .flake8                        # Linting config
├── run                            # Start script
├── stop                           # Stop script
└── status                         # Status script
```

## Component Overview

### Infrastructure Layer

| Component | Technology | Port(s) | Purpose |
|-----------|-----------|---------|---------|
| **Caddy** | Caddy 2.8 | 80, 443 | Reverse proxy, TLS, forward_auth |
| **Postgres** | pgvector/pg16 | 5432 | Database with vector extension |
| **Redis** | Redis 7 | 6379 | Cache for Authentik |
| **Authentik** | Server + Worker + Proxy | 9000 | Authentication & SSO |

### Application Layer

| Service | Location | Port(s) | Description |
|---------|----------|---------|-------------|
| **OpenWebUI** | Docker image | 8080 | Web UI for chat interface (uses PostgreSQL) |
| **Admin UI** | `backend_app/web/` | 8000 | Admin dashboard (FastAPI) |
| **Agent Backend** | `backend_app/app/` | 4000 | Main API with LangGraph agent |
| **RAG MCP** | `backend_app/rag/rag_mcp/` | 4001 | Document retrieval MCP server |
| **RAG Injector** | `backend_app/rag/rag_injector/` | 4002 | Document ingestion REST API |
| **Tools MCP** | `basic_tools_mcp_service/` | 5000 | Search tools MCP server |
| **Ollama** | Docker image | 11434 | LLM inference engine |
| **SearXNG** | Docker image | 8081 | Meta search engine |

### Supporting Services

| Service | Type | Purpose |
|---------|------|---------|
| **mcp_watcher** | Daemon | Auto-discovers MCP tools, updates DB |
| **supervisord** | Process manager | Orchestrates backend services |

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CADDY (80/443)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Forward Auth (Authentik Proxy)             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────┬───────────────┬───────────────┬──────────────────┘
          │               │               │
          │ /webui        │ /admin        │ /web
          ↓               ↓               ↓
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │OpenWebUI │    │Admin UI  │    │Web API   │
    │  :8080   │    │  :8000   │    │  :8000   │
    └─────┬────┘    └────┬─────┘    └────┬─────┘
          │              │               │
          │              └───────┬───────┘
          │                      │
          │            ┌─────────▼─────────┐
          │            │  Agent Backend    │
          │            │    (port 4000)    │
          │            │  ┌──────────────┐ │
          │            │  │ agent_skill  │ │ ← LangGraph orchestrator
          │            │  └──────┬───────┘ │
          │            │         │         │
          │            │   ┌─────┴─────┐   │
          │            │   │           │   │
          │            │   ↓           ↓   │
          │            │ rag_skill  tools_skill
          │            └───┬──────────┬────┘
          │                │          │
          └────────────────┼──────────┘
                          │          │
                ┌─────────▼──┐   ┌───▼──────────┐
                │ RAG MCP    │   │ Tools MCP    │
                │  :4001     │   │  :5000       │
                │            │   │ ┌──────────┐ │
                │ ┌────────┐ │   │ │ arXiv    │ │
                │ │PGVector│ │   │ │ CDS      │ │
                │ │Search  │ │   │ │ INSPIRE  │ │
                │ └────────┘ │   │ │ SearXNG  │ │
                └────────────┘   │ └──────────┘ │
                                 └──────────────┘
                    ↓                    ↓
              ┌───────────┐        ┌──────────┐
              │ Postgres  │        │ SearXNG  │
              │ (PGVector)│        │  :8081   │
              └───────────┘        └──────────┘
```

## Service Communication Matrix

| From Service | To Service | Protocol | Purpose |
|--------------|-----------|----------|---------|
| Caddy | Authentik Proxy | HTTP | Forward auth verification |
| Caddy | OpenWebUI | HTTP | Proxy /webui requests |
| Caddy | Admin UI | HTTP | Proxy /admin requests |
| Caddy | Web API | HTTP | Proxy /web requests |
| Agent Backend | RAG MCP | SSE/HTTP | Document retrieval |
| Agent Backend | Tools MCP | SSE/HTTP | Tool discovery & execution |
| Agent Backend | Ollama | HTTP | LLM inference |
| Agent Backend | Postgres | PostgreSQL | Data persistence |
| Tools MCP | SearXNG | HTTP | Web search |
| Tools MCP | arXiv API | HTTP | Paper search |
| Tools MCP | CDS API | HTTP | CERN docs search |
| Tools MCP | INSPIRE API | HTTP | HEP literature search |
| RAG MCP | Postgres | PostgreSQL | Vector search |
| RAG Injector | Postgres | PostgreSQL | Document insertion |
| mcp_watcher | Tools MCP | SSE/HTTP | Tool discovery |
| mcp_watcher | Postgres | PostgreSQL | Tool registration |

## Network Architecture

### External Access (via Caddy)

```
https://your-domain.com/
├── /webui/*           → OpenWebUI (port 8080)
├── /admin/*           → Admin UI (backend port 8000)
├── /web/*             → Web API (backend port 8000)
└── /rag/*             → RAG Injector (backend port 4002) - prefix stripped

https://auth.your-domain.com/
└── /*                 → Authentik Server (port 9000)
```

**Path Rewriting with `handle_path`:**
- Caddy uses `handle_path` directive to strip path prefixes
- Client: `GET /rag/health` → Backend: `GET /health`
- Client: `POST /rag/upload` → Backend: `POST /upload`
- This allows backend services to expose root paths internally

### Internal Docker Network

All services communicate via Docker's internal DNS:

- `postgres:5432` - Database
- `redis:6379` - Cache
- `authentik_server:9000` - Auth server
- `authentik_proxy:9000` - Auth proxy
- `openwebui:8080` - Web UI
- `backend:8000` - Admin UI
- `backend:4000` - Agent API
- `backend:4001` - RAG MCP
- `backend:4002` - RAG Injector
- `basic_tools_mcp_service:5000` - Tools MCP
- `ollama:11434` - LLM
- `searxng:8080` - Search

## Persistent Data

Docker volumes for data persistence:

| Volume | Purpose | Backed By |
|--------|---------|-----------|
| `caddy_data` | TLS certificates | Caddy |
| `caddy_config` | Caddy configuration | Caddy |
| `ak_pgdata` | Postgres data (Authentik, Backend, OpenWebUI) | Postgres |
| `openwebui_data` | OpenWebUI files/uploads | OpenWebUI |
| `ollama` | Ollama models | Ollama |

## Configuration Files

### Top-Level Configuration

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Full stack orchestration |
| `Caddyfile` | HTTP routing & reverse proxy |
| `.env` | Environment variables (create from `.env.example`) |
| `Makefile` | Build & run shortcuts |
| `pyproject.toml` | Python project metadata |

### Backend Configuration

| File | Purpose |
|------|---------|
| `backend_app/supervisord.conf` | Multi-service process management |
| `backend_app/startup.sh` | Container initialization |
| `backend_app/alembic.ini` | Database migration config |
| `backend_app/requirements.txt` | Python dependencies |
| `backend_app/.env.example` | Backend env template |

## LangGraph Agent Architecture

The agent system uses LangGraph for orchestration:

```
User Query
    ↓
┌───────────────────┐
│ Analyze (LLM)     │ ← Determine execution strategy
└────────┬──────────┘
         │
    ┌────┴────┬────────┬─────────┬─────────┐
    │         │        │         │         │
    ↓         ↓        ↓         ↓         ↓
Calculator  RAG    Tools    RAG+Tools   Direct
    │         │        │         │         │
    │         │        │         ↓         │
    │         │        │    Synthesize    │
    │         │        │         │         │
    └─────────┴────────┴─────────┴─────────┘
                      ↓
                 Final Answer
```

**Execution Patterns:**

1. **Calculator**: Direct math evaluation (fast path)
2. **RAG Only**: Knowledge base search
3. **Tools Only**: External API calls
4. **RAG + Tools**: Multi-skill with synthesis ⭐
5. **Direct**: Simple conversational response

See `docs/AGENT_FLOW.md` for detailed flow documentation.

## Port Reference

| Port | Service | Access | Protocol |
|------|---------|--------|----------|
| 80 | Caddy | Public | HTTP |
| 443 | Caddy | Public | HTTPS |
| 5432 | Postgres | Internal | PostgreSQL |
| 6379 | Redis | Internal | Redis |
| 8000 | Admin UI | Via Caddy or Internal | HTTP |
| 4000 | Agent API | Via Caddy or Internal | HTTP |
| 4001 | RAG MCP | Internal | SSE/HTTP |
| 4002 | RAG Injector | Internal | HTTP |
| 5000 | Tools MCP | Internal | SSE/HTTP |
| 8080 | OpenWebUI | Via Caddy | HTTP |
| 8081 | SearXNG | Internal | HTTP |
| 9000 | Authentik | Internal | HTTP |
| 11434 | Ollama | Internal | HTTP |

## Authentication Flow

```
1. User → Caddy → Forward Auth → Authentik Proxy
                       ↓
              Is user authenticated?
                       ↓
         ┌─────────────┴─────────────┐
         │ YES                       │ NO
         ↓                           ↓
   Add headers:                Redirect to
   X-Authentik-Email           Authentik login
   X-Authentik-Name               ↓
   X-Authentik-Groups        User logs in
         ↓                           ↓
   Forward to                  Redirect back
   backend service             to original URL
         ↓
   Backend reads
   headers for
   user identity
```

## Development Workflows

### Full Stack Development

```bash
# Start everything
docker compose up --build

# View logs
docker compose logs -f

# Stop everything
docker compose down
```

### Backend-Only Development

```bash
# Start backend + dependencies
cd backend
docker compose -f docker-compose.local.yml up --build

# Or use shortcuts from repo root
./run      # Start services
./status   # Check status
./stop     # Stop services
```

### Local Python Development

```bash
cd backend/backend_app

# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Run individual services
uvicorn app.main:app --reload --port 4000
uvicorn web.main:app --reload --port 8000
python rag/rag_mcp/main.py
python tools/watcher.py
```

## Key Technologies

| Category | Technology |
|----------|-----------|
| **Orchestration** | Docker Compose, Supervisord |
| **Web Framework** | FastAPI |
| **Agent Framework** | LangGraph |
| **LLM** | Ollama (llama2, nomic-embed-text) |
| **Vector DB** | PostgreSQL with pgvector |
| **Authentication** | Authentik |
| **Reverse Proxy** | Caddy |
| **MCP Protocol** | SSE/HTTP transport |
| **Search** | SearXNG, arXiv, CDS, INSPIRE-HEP |
| **Migrations** | Alembic |

## Environment Variables Reference

### Infrastructure

```bash
DOMAIN=your-domain.com
AUTH_DOMAIN=auth.your-domain.com
```

### Authentik

```bash
AUTHENTIK_TAG=2024.2.0
AUTHENTIK_SECRET_KEY=<generate-random-key>
AUTHENTIK_OUTPOST_TOKEN=<from-authentik-admin>
```

### Database

```bash
AK_POSTGRES_DB=authentik
AK_POSTGRES_USER=authentik
AK_POSTGRES_PASSWORD=<secure-password>
DATABASE_URL=postgresql+psycopg://${AK_POSTGRES_USER}:${AK_POSTGRES_PASSWORD}@postgres:5432/${AK_POSTGRES_DB}
```

### OpenWebUI

```bash
WEBUI_AUTH=true  # Enable Authentik-based authentication
OPENWEBUI_POSTGRES_DB=openwebui  # Optional, defaults to 'openwebui'

# OpenWebUI uses Authentik trusted headers for authentication:
# - WEBUI_AUTH_TRUSTED_EMAIL_HEADER: X-Authentik-Email
# - WEBUI_AUTH_TRUSTED_NAME_HEADER: X-Authentik-Name
# - WEBUI_AUTH_TRUSTED_GROUPS_HEADER: X-Authentik-Groups
```

### Backend Services

```bash
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_CHAT_MODEL=llama2:7b
OLLAMA_EMBED_MODEL=nomic-embed-text
COLLECTION_DOCS=kb_docs
COLLECTION_TOOLS=tool_catalog
RAG_MCP_URL=http://localhost:4002/mcp
TOOL_SELECT_TOPK=6
```

### Search Services

```bash
SEARXNG_URL=http://searxng:8080
CDS_BASE_URL=https://cds.cern.ch
INSPIRE_BASE_URL=https://inspirehep.net
ARXIV_API_URL=http://export.arxiv.org/api/query
```

## Documentation Index

- **Repository Root**: `/README.md` - Project overview
- **Backend**: `/backend/backend_app/README.md` - Backend services
- **Agent Flow**: `/docs/AGENT_FLOW.md` - Multi-skill architecture
- **Structure**: `/docs/STRUCTURE.md` - This file
- **OpenWebUI Auth**: `/docs/OPENWEBUI_AUTHENTIK.md` - Authentication integration
- **Caddy Routing**: `/docs/CADDY_ROUTING.md` - Path rewriting and routing
- **LLM Management**: `/backend/backend_app/LLM_MANAGEMENT.md`
- **RAG Skill**: `/backend/backend_app/app/RAG_SKILL_GUIDE.md`
- **Tools Skill**: `/backend/backend_app/app/TOOLS_SKILL_GUIDE.md`
- **MCP Watcher**: `/backend/backend_app/tools/MCP_WATCHER_*.md`
- **Tool Discovery**: `/backend/backend_app/tools/TOOL_DISCOVERY_GUIDE.md`

## Quick Reference

### Start Services

```bash
docker compose up -d
```

### View Logs

```bash
docker compose logs -f backend
```

### Run Migrations

```bash
docker compose exec backend alembic upgrade head
```

### Check Service Health

```bash
# Admin UI
curl http://localhost:8000/admin/api/health

# Agent API
curl http://localhost:4000/v1/models

# RAG Injector
curl http://localhost:4002/health

# Tools MCP
curl http://localhost:5000/health
```

### Access Services

- OpenWebUI: http://localhost/webui
- Admin UI: http://localhost/admin
- Authentik: http://localhost:9000 (or https://auth.your-domain.com)

### OpenWebUI Database Migration

If you're migrating from SQLite to PostgreSQL:

```bash
# 1. Stop services
docker compose down

# 2. Ensure OPENWEBUI_POSTGRES_DB is set in .env
echo "OPENWEBUI_POSTGRES_DB=openwebui" >> .env

# 3. Start with new configuration
docker compose up -d

# OpenWebUI will automatically create the PostgreSQL database on first run
# Note: Existing SQLite data in openwebui_data volume is not automatically migrated
# You'll need to export/import data manually if you need to preserve it
```

## Support & Resources

- GitHub: https://github.com/InnovateAILab/AgentKS
- Issues: Use GitHub Issues for bug reports
- Copilot Instructions: `.github/copilot-instructions.md`
