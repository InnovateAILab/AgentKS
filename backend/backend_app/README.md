# AgentKS — Backend

This folder contains the backend services for AgentKS: a multi-service FastAPI application orchestrated with LangGraph, providing admin UI, RAG capabilities, MCP services, and agentic workflows.

## Directory Structure

```
backend_app/
├── app/                    # Main agent backend (port 4000)
│   ├── main.py            # FastAPI app with OpenAI-compatible endpoints
│   ├── agent_skill.py     # LangGraph agent orchestrator (multi-skill execution)
│   ├── rag_skill.py       # RAG retrieval skill (LangGraph)
│   ├── tools_skill.py     # Dynamic tool discovery skill (LangGraph)
│   ├── llms.py            # LLM configuration and database fallback
│   └── backup/            # Archived code (langgraph_adapter.py.bak)
│
├── web/                   # Admin web UI (port 8000)
│   ├── main.py           # FastAPI admin interface
│   ├── templates/        # Jinja2 templates
│   └── static/           # CSS, JS assets
│
├── rag/                  # RAG services
│   ├── rag_mcp/          # MCP server for RAG retrieval (port 4001)
│   ├── rag_injector/     # REST API for document injection (port 4002)
│   ├── rag_common.py     # Shared embeddings and DB utilities
│   └── daemons/          # Background processing (URL watcher, etc.)
│
├── tools/                # MCP tool integration
│   ├── client.py         # MCP client for tool execution
│   ├── tool_discovery.py # Semantic/hybrid tool search
│   ├── watcher.py        # Auto-discovery daemon (mcp_watcher)
│   └── models.py         # Tool data models
│
├── migrations/           # Alembic database migrations
│   └── versions/         # Migration scripts
│
├── supervisord.conf      # Multi-service orchestration
├── startup.sh           # Container initialization script
├── Dockerfile           # Multi-stage build
└── requirements.txt     # Python dependencies
```

## Architecture Overview

**LangGraph Agent System:**
- **agent_skill.py**: Main orchestrator with smart routing (calculator/RAG/tools/direct)
- **rag_skill.py**: Knowledge base search via RAG MCP service
- **tools_skill.py**: Dynamic tool discovery and execution
- **Multi-skill execution**: Can combine RAG + Tools with intelligent synthesis

**Service Ports:**
- 8000: Admin Web UI
- 4000: Agent Backend API (OpenAI-compatible)
- 4001: RAG MCP Server
- 4002: RAG Injection REST API

**Key Features:**
- All tools managed via MCP services (basic_tools_mcp_service)
- Automatic tool discovery with semantic/hybrid search
- Document management fully delegated to RAG MCP
- No local vector stores (PGVector via MCP)

## Quick Start

### Full Stack (Recommended)

From the repo root with `.env` configured:

```bash
docker compose up --build
```

**Exposed Services:**
- http://localhost (Caddy reverse proxy with Authentik)
- http://localhost/webui (OpenWebUI)
- http://localhost/admin (Admin UI - requires auth)
- http://localhost/web (Web API - requires auth)

### Backend Only (Development)

Build and run just the backend container (runs migrations then starts all services):

```bash
docker build -t agentks-backend ./backend/backend_app
docker run --rm \
  -p 8000:8000 \
  -p 4000:4000 \
  -p 4001:4001 \
  -p 4002:4002 \
  --env-file .env \
  agentks-backend
```

**Direct Access:**
- http://localhost:8000/admin (Admin UI)
- http://localhost:4000/v1/models (Agent API)
- http://localhost:4001 (RAG MCP - MCP client only)
- http://localhost:4002/health (RAG Injection API)

## Container Startup Process

The `startup.sh` script orchestrates initialization:

1. **Database Migrations** (with retries for postgres readiness)
   ```bash
   alembic -c /app/alembic.ini upgrade head
   ```

2. **Supervisord Launch** - Starts all services in parallel:
   - `uvicorn web.main:app --host 0.0.0.0 --port 8000` (Admin UI)
   - `uvicorn app.main:app --host 0.0.0.0 --port 4000` (Agent Backend)
   - `python -u rag/rag_mcp/main.py` (RAG MCP Server, port 4001)
   - `uvicorn rag.rag_injector.main:app --host 0.0.0.0 --port 4002` (RAG Injector)
   - `python -u tools/watcher.py` (mcp_watcher daemon - tool discovery)

See `supervisord.conf` for full configuration.

## Health Checks & API Endpoints

### Health Endpoints

```bash
# Admin UI health
curl http://localhost:8000/admin/api/health
# Response: {"ok": true}

# Agent API models
curl http://localhost:4000/v1/models
# Response: {"object": "list", "data": [...]}

# RAG Injection health
curl http://localhost:4002/health
# Response: {"status": "healthy"}

# RAG MCP (requires MCP client)
# Connect via SSE/HTTP on port 4001
```

### Main API Endpoints (port 4000)

**OpenAI-Compatible:**
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions (streaming/non-streaming)

**RAG Skill:**
- `POST /api/rag-skill/run` - Run RAG skill with LLM generation
- `POST /api/rag-skill/retrieve` - Retrieve documents only (no LLM)

**Tools Skill:**
- `POST /api/tools-skill/run` - Run tools skill with dynamic discovery
- `POST /api/tools-skill/discover` - Discover tools without execution

**Admin (port 8000):**
- `GET /admin/api/health` - Health check
- `GET /admin` - Admin dashboard (requires auth in production)

### Authentication (Production)

When behind Caddy + Authentik, include headers:

```bash
curl -H "X-Authentik-Groups: admin" \
     -H "X-Authentik-Email: user@example.com" \
     http://localhost:8000/admin/api/health
```

## Database & Migrations

**Schema Management:** Alembic migrations in `migrations/versions/`

### Automatic Migration (Container Startup)

Migrations run automatically via `startup.sh` with retry logic:

```bash
alembic -c /app/alembic.ini upgrade head
```

### Manual Migration Commands

Inside running container:

```bash
docker compose exec backend alembic -c /app/alembic.ini upgrade head
```

For standalone image:

```bash
docker run --rm --env-file .env -it agentks-backend \
  alembic -c /app/alembic.ini upgrade head
```

### Create New Migration

```bash
# Inside container or with local environment
alembic -c alembic.ini revision --autogenerate -m "description"
```

**Important:** Migrations are the canonical source for schema. Avoid runtime `CREATE TABLE` statements.

## Environment Variables

### Required Variables

```bash
# Database
DATABASE_URL=postgresql+psycopg://user:pass@postgres:5432/dbname

# Ollama LLM
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_CHAT_MODEL=llama2:7b
OLLAMA_EMBED_MODEL=nomic-embed-text

# PGVector Collections
COLLECTION_DOCS=kb_docs
COLLECTION_TOOLS=tool_catalog

# RAG MCP Service
RAG_MCP_URL=http://localhost:4002/mcp

# Tool Discovery
TOOL_SELECT_TOPK=6
```

### Optional Variables

```bash
# Search Services (provided by basic_tools_mcp_service)
SEARXNG_URL=http://searxng:8080
CDS_BASE_URL=https://cds.cern.ch
INSPIRE_BASE_URL=https://inspirehep.net
ARXIV_API_URL=http://export.arxiv.org/api/query

# Performance Tuning
FETCH_TIMEOUT_SECONDS=25
MAX_CHARS_PER_DOC=250000
MAX_URLS_PER_REQUEST=30
```

**Note:** All search tools (arXiv, CDS, INSPIRE-HEP, SearXNG) are now provided by `basic_tools_mcp_service` and discovered automatically via the mcp_watcher daemon.

See `.env.example` for a complete template.

## Code Organization

### Main Application (`app/`)

```
app/
├── main.py              # FastAPI app, OpenAI-compatible endpoints
│                        # - /v1/chat/completions (OpenAI API)
│                        # - RAG skill endpoints
│                        # - Tools skill endpoints
│
├── agent_skill.py       # LangGraph agent orchestrator
│                        # - Multi-skill execution (RAG + Tools)
│                        # - Smart routing (calculator/rag/tools/direct)
│                        # - Result synthesis
│
├── rag_skill.py        # RAG retrieval skill (LangGraph)
│                        # - Document retrieval via RAG MCP
│                        # - LLM generation with context
│                        # - Source citation
│
├── tools_skill.py      # Dynamic tool discovery (LangGraph)
│                        # - Semantic/hybrid tool search
│                        # - Automatic tool binding
│                        # - MCP tool execution
│
├── llms.py             # LLM configuration
│                        # - Database-backed LLM settings
│                        # - Fallback to environment config
│
└── backup/             # Archived code
    └── langgraph_adapter.py.bak  # Old fallback-based adapter (replaced by agent_skill.py)
```

### Tools Module (`tools/`)

```
tools/
├── client.py           # MCP client for tool execution
├── tool_discovery.py   # Semantic/hybrid tool search
├── watcher.py          # mcp_watcher daemon (auto-discovery)
├── models.py           # Tool data models
└── discovery.py        # Core discovery algorithms
```

### RAG Services (`rag/`)

```
rag/
├── rag_mcp/           # MCP server (port 4001)
│   └── main.py        # RAG retrieval MCP service
│
├── rag_injector/      # REST API (port 4002)
│   └── main.py        # Document injection endpoints
│
├── rag_common.py      # Shared utilities
│                       # - Embeddings (Ollama)
│                       # - Database operations
│
└── daemons/           # Background processing
```

### Admin UI (`web/`)

```
web/
├── main.py            # FastAPI admin interface
├── templates/         # Jinja2 templates
│   ├── base.html
│   ├── home.html
│   ├── mcps_list.html
│   ├── rags_list.html
│   └── urls_list.html
└── static/            # CSS, JavaScript
    ├── app.js
    └── styles.css
```

### Documentation

- `AGENT_FLOW.md` - Multi-skill execution architecture (in `/docs`)
- `RAG_SKILL_GUIDE.md` - RAG skill usage and configuration
- `TOOLS_SKILL_GUIDE.md` - Tools skill usage and configuration
- `LLM_MANAGEMENT.md` - LLM configuration guide
- `tools/MCP_WATCHER_*.md` - MCP watcher documentation
- `tools/TOOL_DISCOVERY_GUIDE.md` - Tool discovery system

## Development & Debugging

### View Logs

Full stack logs:
```bash
docker compose logs -f
```

Backend container only:
```bash
docker compose logs -f backend
```

Specific service within backend:
```bash
docker compose exec backend supervisorctl tail -f app
docker compose exec backend supervisorctl tail -f web
docker compose exec backend supervisorctl tail -f rag_mcp
docker compose exec backend supervisorctl tail -f mcp_watcher
```

### Supervisord Management

Check service status:
```bash
docker compose exec backend supervisorctl status
```

Restart a specific service:
```bash
docker compose exec backend supervisorctl restart app
docker compose exec backend supervisorctl restart mcp_watcher
```

### Local Development

For faster iteration, run services locally without Docker:

```bash
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

### Testing Agent Skills

Test RAG skill:
```bash
curl -X POST http://localhost:4000/api/rag-skill/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is quantum computing?", "k": 5}'
```

Test Tools skill:
```bash
curl -X POST http://localhost:4000/api/tools-skill/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search arXiv for transformer papers",
    "user_id": "test@example.com",
    "role": "user"
  }'
```

Test chat completions (OpenAI-compatible):
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-agent",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Agent Flow Architecture

The backend uses **LangGraph** for agent orchestration with multi-skill execution:

### Execution Patterns

1. **Calculator** - Direct math evaluation (fast path)
2. **RAG Only** - Knowledge base search
3. **Tools Only** - External API/search
4. **RAG + Tools** - Multi-skill with synthesis ⭐
5. **Direct** - Simple conversational responses

### Flow Example: RAG + Tools

```
Query → Analyze → RAG Skill → Tools Skill → Synthesize → Answer
```

**Process:**
1. Query analyzed by LLM to determine execution plan
2. RAG skill retrieves relevant documents from knowledge base
3. Tools skill discovers and executes external tools (with RAG context)
4. Synthesis node combines results intelligently
5. Final answer with citations from both sources

See `/docs/AGENT_FLOW.md` for detailed architecture documentation.

## MCP Services Integration

All tools and RAG operations go through MCP services:

**External Services:**
- `basic_tools_mcp_service` (port 5000) - Search tools (arXiv, CDS, INSPIRE-HEP, SearXNG)

**Internal Services:**
- RAG MCP (port 4001) - Document retrieval
- RAG Injector (port 4002) - Document ingestion

**Auto-Discovery:**
- `mcp_watcher` daemon automatically discovers MCP tools
- Semantic/hybrid search for tool selection
- Dynamic tool binding to LLM

## Key Features

✅ **No local vector stores** - All via MCP services  
✅ **Dynamic tool discovery** - Semantic search for tools  
✅ **Multi-skill execution** - RAG + Tools with synthesis  
✅ **Smart routing** - LLM-based query analysis  
✅ **OpenAI-compatible** - Drop-in replacement for OpenAI API  
✅ **Streaming support** - Real-time response streaming  
✅ **Source citations** - Automatic reference tracking  

## Troubleshooting

### Services won't start

Check postgres is ready:
```bash
docker compose exec postgres pg_isready
```

Check migrations:
```bash
docker compose exec backend alembic current
docker compose exec backend alembic upgrade head
```

### MCP tools not discovered

Check mcp_watcher logs:
```bash
docker compose exec backend supervisorctl tail -f mcp_watcher
```

Verify basic_tools_mcp_service is running:
```bash
curl http://localhost:5000/health
```

### Agent not using tools

1. Check tool discovery:
```bash
curl -X POST http://localhost:4000/api/tools-skill/discover \
  -H "Content-Type: application/json" \
  -d '{"query": "search for papers", "user_id": "test", "role": "user"}'
```

2. Verify database has tools (seeded by mcp_watcher)

3. Check TOOL_SELECT_TOPK environment variable

### RAG returns no results

Check RAG MCP service:
```bash
curl http://localhost:4001/sse/tools/list
```

Verify documents are ingested:
```bash
curl http://localhost:4002/health
```

## Production Deployment

When deploying behind Caddy + Authentik (full stack):

1. **Configure `.env`** with all required variables
2. **Run full stack**: `docker compose up -d`
3. **Access via Caddy**:
   - https://your-domain.com/webui (OpenWebUI)
   - https://your-domain.com/admin (Admin UI)
   - https://your-domain.com/web (Web API)

4. **Authentication** handled by Authentik
5. **Identity headers** forwarded by Caddy

## Additional Resources

- **Agent Flow**: `/docs/AGENT_FLOW.md` - Multi-skill execution details
- **RAG Skill**: `app/RAG_SKILL_GUIDE.md` - RAG configuration
- **Tools Skill**: `app/TOOLS_SKILL_GUIDE.md` - Tool discovery
- **LLM Config**: `LLM_MANAGEMENT.md` - LLM setup
- **MCP Watcher**: `tools/MCP_WATCHER_*.md` - Tool auto-discovery
- **Root README**: `/README.md` - Full stack overview



