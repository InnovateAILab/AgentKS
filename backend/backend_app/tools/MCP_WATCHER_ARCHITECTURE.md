# MCP Watcher Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AgentKS Backend                              │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Admin UI   │  │  Backend API │  │  RAG MCP     │              │
│  │   (Port 8000)│  │  (Port 4000) │  │  (Port 4001) │              │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘              │
│         │                 │                                          │
│         │                 │                                          │
│  ┌──────▼─────────────────▼──────────────────────────────────┐     │
│  │              PostgreSQL Database (PGVector)                │     │
│  │                                                             │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐            │     │
│  │  │   mcps   │  │  tools   │  │rag_documents │            │     │
│  │  └──────────┘  └──────────┘  └──────────────┘            │     │
│  │                                                             │     │
│  │  ┌─────────────────────────────────────────┐              │     │
│  │  │  PGVector Collections                    │              │     │
│  │  │  - tool_embeddings (tool search)        │              │     │
│  │  │  - document_embeddings (RAG search)     │              │     │
│  │  └─────────────────────────────────────────┘              │     │
│  └─────▲──────────────────▲───────────────────────────────────┘     │
│        │                  │                                          │
│        │                  │                                          │
│  ┌─────┴──────────┐  ┌────┴─────────────┐                          │
│  │  MCP Watcher   │  │   URL Watcher    │  ← Background Daemons    │
│  │   (Daemon)     │  │    (Daemon)      │                          │
│  └────────────────┘  └──────────────────┘                          │
│         │                                                            │
│         │ Discovers tools                                           │
│         ▼                                                            │
│  ┌──────────────────────────────────────────┐                      │
│  │  External MCP Servers                     │                      │
│  │  - basic_tools_mcp (port 5000)           │                      │
│  │  - hep_tools_mcp                          │                      │
│  │  - custom MCPs                            │                      │
│  └──────────────────────────────────────────┘                      │
└───────────────────────────────────────────────────────────────────┘
```

## MCP Watcher Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MCP Tool Discovery Flow                          │
└─────────────────────────────────────────────────────────────────────┘

1. Admin adds MCP server
   ┌──────────────┐
   │  Admin UI    │  POST /admin/mcps/add
   └──────┬───────┘
          │
          ▼
   ┌─────────────────────────────────┐
   │  INSERT INTO mcps               │
   │  - status = 'pending'           │
   │  - endpoint = 'http://mcp:5000' │
   │  - description, context, etc    │
   └─────────────────────────────────┘

2. MCP Watcher claims MCP
   ┌──────────────┐
   │ mcp_watcher  │  Every 60 seconds
   └──────┬───────┘
          │
          ▼
   ┌─────────────────────────────────┐
   │  UPDATE mcps                    │
   │  SET status = 'processing'      │
   │  WHERE status = 'pending'       │
   │  RETURNING *                    │
   │  FOR UPDATE SKIP LOCKED         │
   └─────────────────────────────────┘

3. Connect to MCP and discover tools
   ┌──────────────┐
   │ mcp_watcher  │
   └──────┬───────┘
          │ langchain_mcp_adapters
          ▼
   ┌─────────────────────────────────┐
   │  External MCP Server            │
   │  GET /mcp/tools                 │
   │                                 │
   │  Returns:                       │
   │  [                              │
   │    {                            │
   │      name: "arxiv_search",     │
   │      description: "Search...",  │
   │      inputSchema: {...}         │
   │    },                           │
   │    ...                          │
   │  ]                              │
   └─────────────────────────────────┘

4. Register tools in database
   ┌──────────────┐
   │ mcp_watcher  │  For each tool
   └──────┬───────┘
          │
          ▼
   ┌─────────────────────────────────┐
   │  INSERT INTO tools              │
   │  - id = 'mcp123_arxiv_search'   │
   │  - name = 'arxiv_search'        │
   │  - kind = 'mcp_tool'            │
   │  - mcp_id = 'mcp123'            │
   │  - metadata = {...}             │
   │  ON CONFLICT (id) DO UPDATE     │
   └─────────────────────────────────┘

5. Index tools for semantic search
   ┌──────────────┐
   │ mcp_watcher  │  For each tool
   └──────┬───────┘
          │ index_tool_with_mcp_context()
          ▼
   ┌─────────────────────────────────────────┐
   │  Generate embeddings                    │
   │  Combined description:                  │
   │  "Tool desc | Provider: MCP desc |      │
   │   Context: context | Resources: res"    │
   └─────────┬───────────────────────────────┘
             │
             ▼
   ┌─────────────────────────────────┐
   │  INSERT INTO PGVector           │
   │  collection = 'tool_embeddings' │
   │  - embedding vector             │
   │  - metadata (tool_id, enabled)  │
   └─────────────────────────────────┘

6. Mark MCP as processed
   ┌──────────────┐
   │ mcp_watcher  │
   └──────┬───────┘
          │
          ▼
   ┌─────────────────────────────────┐
   │  UPDATE mcps                    │
   │  SET status = 'enabled'         │
   │      last_checked_at = now()    │
   │      metadata = {tool_count: 3} │
   └─────────────────────────────────┘

7. Tools available for discovery
   ┌──────────────┐
   │  User Query  │  "search for physics papers"
   └──────┬───────┘
          │
          ▼
   ┌─────────────────────────────────┐
   │  tool_discovery.discover_tools  │
   │  - Semantic search in PGVector  │
   │  - Filters: enabled=true        │
   │  - Ranks by relevance           │
   └─────────┬───────────────────────┘
             │
             ▼
   ┌─────────────────────────────────┐
   │  Returns matched tools:         │
   │  [                              │
   │    {                            │
   │      id: "mcp123_arxiv_search", │
   │      name: "arxiv_search",      │
   │      score: 0.91,               │
   │      metadata: {...}            │
   │    }                            │
   │  ]                              │
   └─────────────────────────────────┘
```

## Component Interaction Matrix

| Component | Reads From | Writes To | Purpose |
|-----------|-----------|-----------|---------|
| **Admin UI** | mcps, tools | mcps | Add/edit MCP servers |
| **MCP Watcher** | mcps (pending) | mcps (status), tools, PGVector | Discover and register tools |
| **Tool Discovery** | tools, PGVector | - | Semantic tool search |
| **Backend API** | tools, mcps | tool_runs | Execute tools, log runs |
| **URL Watcher** | urls | rag_documents, PGVector | Process URLs for RAG |

## Supervisord Process Tree

```
supervisord (PID 1)
├── web (uvicorn, port 8000)
│   └── Admin UI + web endpoints
├── app (uvicorn, port 4000)
│   └── Backend API + chat completions
├── rag_mcp (python, port 4001)
│   └── RAG MCP service (tools to query RAG)
├── rag_injector (uvicorn, port 4002)
│   └── RAG document injection API
├── url_watcher (python)
│   └── Monitor urls table, fetch & ingest
└── mcp_watcher (python) ← NEW!
    └── Monitor mcps table, discover tools
```

## Database Schema Relationships

```
┌──────────────────┐
│      mcps        │
│                  │
│  id (PK)         │
│  name            │
│  endpoint        │
│  status          │◄────────┐
│  description     │         │
│  context         │         │ References
│  resource        │         │
│  auth            │         │
│  last_checked_at │         │
└──────────────────┘         │
                             │
                             │
┌──────────────────┐         │
│     tools        │         │
│                  │         │
│  id (PK)         │         │
│  name            │         │
│  kind            │         │
│  mcp_id (FK) ────┼─────────┘
│  metadata        │
│  tags            │
└────────┬─────────┘
         │
         │ Indexed in
         │
         ▼
┌─────────────────────────────┐
│  PGVector: tool_embeddings  │
│                             │
│  - embedding (vector)       │
│  - metadata.tool_id         │
│  - metadata.enabled         │
│  - metadata.scope           │
└─────────────────────────────┘
```

## MCP Watcher State Machine

```
         ┌──────────┐
    ┌───►│ pending  │◄─── Initial state (new MCP)
    │    └────┬─────┘
    │         │
    │         │ Claimed by daemon
    │         ▼
    │    ┌──────────┐
    │    │processing│
    │    └────┬─────┘
    │         │
    │         ├─── Success ───────┐
    │         │                   │
    │         │                   ▼
    │         │              ┌──────────┐
    │         │              │ enabled  │
    │         │              └────┬─────┘
    │         │                   │
    │         │                   │ After 24 hours
    │         │                   │
    │         └───────────────────┘
    │         
    │         Error
    │         │
    │         ▼
    │    ┌──────────┐
    └────│  error   │
         └──────────┘
              │
              │ Manual reset
              │ (change status back to 'pending')
              └─────────────────────►
```

## Error Handling Flow

```
┌─────────────────────────────────────────────────────────────┐
│              MCP Watcher Error Handling                      │
└─────────────────────────────────────────────────────────────┘

Try:
  ┌─────────────────────┐
  │ Connect to MCP      │
  └──────┬──────────────┘
         │
         │ Success
         ▼
  ┌─────────────────────┐
  │ Discover tools      │
  └──────┬──────────────┘
         │
         │ Success
         ▼
  ┌─────────────────────┐
  │ Register tools      │
  └──────┬──────────────┘
         │
         │ Success
         ▼
  ┌─────────────────────┐
  │ Mark as 'enabled'   │
  └─────────────────────┘

Except:
  Connection Error ─────┐
  Authentication Error  │
  Timeout              ├───┐
  Invalid Response     │   │
  Tool Registration    │   │
  Indexing Error ──────┘   │
                           │
                           ▼
                    ┌──────────────────┐
                    │ Log error        │
                    │ Mark as 'error'  │
                    │ Save last_error  │
                    └──────────────────┘
                           │
                           │ Admin reviews
                           │ Fixes config
                           │ Resets status
                           ▼
                    ┌──────────────────┐
                    │ Retry on next    │
                    │ daemon cycle     │
                    └──────────────────┘
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Check interval** | 60s | Configurable via MCP_CHECK_INTERVAL |
| **MCPs per cycle** | 5 | Configurable via MCP_CLAIM_LIMIT |
| **Tool discovery time** | 1-5s per MCP | Depends on MCP response time |
| **Tool indexing time** | 0.1-0.5s per tool | Depends on Ollama embedding speed |
| **Refresh interval** | 24h | Periodic re-discovery |
| **Database locks** | Yes | FOR UPDATE SKIP LOCKED prevents conflicts |
| **Memory usage** | ~50MB | Plus Ollama embedding overhead |

## Monitoring Queries

```sql
-- Check MCP processing status
SELECT 
  status, 
  COUNT(*) as count 
FROM mcps 
GROUP BY status;

-- Recent MCP activity
SELECT 
  id, 
  name, 
  status, 
  last_checked_at, 
  last_error 
FROM mcps 
ORDER BY last_checked_at DESC 
LIMIT 10;

-- Tools per MCP
SELECT 
  m.id,
  m.name as mcp_name,
  COUNT(t.id) as tool_count,
  m.status,
  m.last_checked_at
FROM mcps m
LEFT JOIN tools t ON t.mcp_id = m.id
GROUP BY m.id, m.name, m.status, m.last_checked_at
ORDER BY tool_count DESC;

-- Recent tool registrations
SELECT 
  t.name as tool_name,
  m.name as mcp_name,
  t.created_at
FROM tools t
JOIN mcps m ON t.mcp_id = m.id
WHERE t.kind = 'mcp_tool'
ORDER BY t.created_at DESC
LIMIT 10;

-- Tool indexing status
SELECT 
  c.name as collection,
  COUNT(*) as embedding_count
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'tool_embeddings'
GROUP BY c.name;
```

---

This architecture enables fully automated tool discovery and registration, making it easy for admins to add MCP servers and immediately make their tools available for semantic search and LLM binding.
