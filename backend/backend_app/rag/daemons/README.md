# RAG Daemons

Background services for RAG (Retrieval-Augmented Generation) system automation.

## Daemons

1. **URL Watcher**: Monitors `urls` table and automatically processes URLs

**Note:** The MCP Watcher daemon has been moved to the `mcp` module (`mcp.watcher`). See `backend_app/mcp/README.md` for details.

---

## URL Watcher Daemon

Automated URL monitoring and document injection service.

### Overview

The MCP watcher daemon monitors the `mcps` table for new or updated MCP servers and automatically:

1. **Discovers Tools**: Connects to MCP servers and retrieves available tools
2. **Registers Tools**: Creates tool entries in the `tools` table
3. **Indexes Tools**: Indexes tools with semantic search (including MCP context)
4. **Keeps Tools Updated**: Periodically refreshes tool listings

### Workflow

```
Admin adds MCP server via Admin UI
         ↓
    [mcps table]
  status='pending'
         ↓
   MCP Watcher Daemon
         ↓
    ┌─────────────────────┐
    │ Connect to MCP      │
    │ (via langchain_mcp) │
    └─────────────────────┘
         ↓
    ┌─────────────────────┐
    │ Get Available Tools │
    │ - name              │
    │ - description       │
    │ - inputSchema       │
    └─────────────────────┘
         ↓
    ┌─────────────────────┐
    │ Register in DB      │
    │ - tools table       │
    │ - tool_id           │
    │ - kind='mcp_tool'   │
    └─────────────────────┘
         ↓
    ┌─────────────────────┐
    │ Index for Search    │
    │ - Combine tool desc │
    │ - Add MCP context   │
    │ - Create embeddings │
    │ - Store in PGVector │
    └─────────────────────┘
         ↓
   Update status='enabled'
   Set last_checked_at
         ↓
   Periodic refresh (daily)
```

### Features

- ✅ **Automatic Discovery**: No manual tool registration needed
- ✅ **MCP Context Integration**: Combines tool + MCP descriptions for better search
- ✅ **Deterministic Tool IDs**: `{mcp_id}_{tool_name}` prevents duplicates
- ✅ **Smart Indexing**: Uses `index_tool_with_mcp_context()` for rich semantic search
- ✅ **Error Handling**: Marks failed MCPs with error messages
- ✅ **Periodic Refresh**: Daily re-discovery to catch new tools
- ✅ **Concurrent Safe**: Uses database locks (`FOR UPDATE SKIP LOCKED`)

### Configuration

Environment variables:

```bash
# Polling interval (seconds)
MCP_CHECK_INTERVAL=60

# MCPs to process per batch
MCP_CLAIM_LIMIT=5

# Database connection
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname
```

### MCP Status Flow

```
pending → processing → enabled → [periodic checks] → processing → enabled
   ↓                       ↓                             ↓
 error                   error                         error
```

**Status meanings:**
- `pending`: Newly added, awaiting tool discovery
- `processing`: Currently being processed by daemon
- `enabled`: Successfully processed, tools registered
- `error`: Failed to discover tools (see `last_error`)

### Tool Registration

For each discovered tool, the daemon creates:

**Database Record (`tools` table):**
```json
{
  "id": "mcp123_arxiv_search",
  "name": "arxiv_search",
  "kind": "mcp_tool",
  "mcp_id": "mcp123",
  "metadata": {
    "description": "Search arXiv for papers",
    "enabled": true,
    "scope": "global",
    "config": {},
    "inputSchema": {...}
  },
  "tags": ["physics", "academic"]
}
```

**Vector Index:**
```
Combined description for semantic search:
"Search arXiv for papers | Provider: High Energy Physics research tools | 
Context: Tools for academic research in physics | Resources: arXiv, INSPIRE-HEP"
```

### Tool ID Format

Tool IDs are deterministic: `{mcp_id}_{tool_name}`

**Benefits:**
- Prevents duplicate tool registration
- Easy to identify tool source
- Supports tool updates (upsert on conflict)
- Clean deletion when MCP is removed

### Integration with Tool Discovery

Tools registered by MCP watcher are automatically available for:

1. **Semantic Search**: Via `tool_discovery.discover_tools(query)`
2. **LLM Binding**: Via `bind_discovered_tools_to_llm(llm, tools)`
3. **Admin UI**: Listed in `/admin/tools` with MCP provider info
4. **API**: Available via `/api/tools` endpoint

### Functions Moved from main.py

The following functions were moved from `app/main.py` to `mcp_watcher.py`:

1. **`tool_upsert()`**: Insert/update tools in database
2. **Tool indexing**: Now uses `index_tool_with_mcp_context()` from `tool_discovery.py`

This centralizes tool management logic in the daemon where it belongs.

### Monitoring

**Logs:**
```bash
# View mcp_watcher logs
docker-compose logs -f backend | grep mcp_watcher

# Check processing status
docker-compose exec postgres psql -U user -d dbname -c \
  "SELECT id, name, status, last_checked_at, last_error FROM mcps;"
```

**Metrics in database:**
```sql
-- Tool count per MCP
SELECT 
  m.id, 
  m.name, 
  COUNT(t.id) as tool_count,
  m.last_checked_at
FROM mcps m
LEFT JOIN tools t ON t.mcp_id = m.id
GROUP BY m.id, m.name, m.last_checked_at;
```

### Troubleshooting

**MCP stuck in 'processing':**
- Daemon may have crashed during processing
- Manually reset: `UPDATE mcps SET status='pending' WHERE id='...'`

**No tools discovered:**
- Check MCP endpoint is accessible
- Verify auth credentials in `mcps.auth` column
- Check daemon logs for connection errors

**Tools not appearing in search:**
- Verify indexing succeeded (check daemon logs)
- Re-index manually: `reindex_all_tools()` in Python shell
- Check PGVector collection: `SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id IN (SELECT uuid FROM langchain_pg_collection WHERE name='tool_embeddings');`

**Authentication errors:**
- Update `mcps.auth` column with correct credentials
- Format: `{"type": "bearer", "token": "...", "headers": {...}}`
- Reset status to trigger re-processing

**Note:** The MCP Watcher daemon has been moved to the `mcp` module (`mcp.watcher`). See `backend_app/mcp/README.md` for details.

---

## URL Watcher Daemon

Automated URL monitoring and document injection service.

### Overview

The URL watcher daemon monitors the `urls` table and automatically processes URLs added by users through the web/admin UI. It handles:

1. **QueuedURLs (`status='queued'`)**: Fetch content and inject into RAG
2. **Refresh URLs (`status='refresh'`)**: Delete old documents and re-fetch
3. **Periodic Checks**: Monitor ingested URLs for content changes

### Workflow

```
User adds URL via Admin UI
         ↓
    [urls table]
   status='queued'
         ↓
   URL Watcher Daemon
         ↓
    ┌─────────────────┐
    │ Fetch Content   │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │ Check Hash      │ ← Compare with existing
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │ Split Chunks    │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │ Generate        │
    │ Embeddings      │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │ Store in RAG    │
    │ - rag_documents │
    │ - PGVector      │
    └─────────────────┘
         ↓
   Update status='ingested'
         ↓
   Periodic checks for updates
```

### Features

- ✅ **Automatic Processing**: Processes queued URLs without manual intervention
- ✅ **Change Detection**: Uses content hashing to detect updates
- ✅ **Smart Refresh**: Only re-processes when content changes
- ✅ **Periodic Monitoring**: Checks ingested URLs for updates
- ✅ **RAG Integration**: Uses `rag_common` for embeddings and storage
- ✅ **Error Handling**: Marks failed URLs with error messages
- ✅ **Concurrent Safe**: Uses database locks to prevent duplicate processing

### Configuration

Environment variables:

```bash
# Polling interval (seconds)
SLEEP_SECONDS=5

# URLs to process per batch
BATCH_SIZE=10

# How often to check ingested URLs (seconds)
CHECK_INTERVAL_SECONDS=3600  # 1 hour

# Consider URL stale after this time (seconds)
STALE_AFTER_SECONDS=21600  # 6 hours

# Default RAG group for URL documents
DEFAULT_RAG_GROUP=web_content

# Default embedding model
DEFAULT_EMBED_MODEL=nomic-embed-text

# Chunking configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### URL Status Flow

```
queued → ingested → [periodic checks] → refresh → ingested
   ↓                                        ↓
 failed                                  failed
```

**Status Values:**
- `queued`: Waiting to be processed
- `ingested`: Successfully processed and stored in RAG
- `refresh`: Needs to be re-processed (detected change or manual trigger)
- `failed`: Processing failed (check `last_error` field)

### Database Schema

**urls table:**
```sql
CREATE TABLE urls (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    scope TEXT DEFAULT 'global',
    tags JSONB DEFAULT '[]'::jsonb,
    status TEXT DEFAULT 'queued',
    last_fetched_at TIMESTAMP,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT now()
);
```

### Usage Examples

#### 1. Add URL via Admin UI

Users add URLs through the web interface, which inserts into the `urls` table with `status='queued'`.

#### 2. Manual URL Addition (SQL)

```sql
INSERT INTO urls (id, url, scope, status)
VALUES (
    gen_random_uuid()::text,
    'https://example.com/document',
    'global',
    'queued'
);
```

#### 3. Force Refresh

```sql
UPDATE urls 
SET status = 'refresh' 
WHERE url = 'https://example.com/document';
```

#### 4. Check Status

```sql
SELECT id, url, status, last_fetched_at, last_error
FROM urls
WHERE url = 'https://example.com/document';
```

### Running the Daemon

The daemon is automatically started by supervisord (see `supervisord.conf`):

```ini
[program:url_watcher]
command=python -u -m backend_app.daemons.url_watcher
directory=/app
autostart=true
autorestart=true
```

**Manual run (for testing):**
```bash
# From backend_app directory
python -m rag.daemons.url_watcher

# Or with custom config
SLEEP_SECONDS=10 BATCH_SIZE=5 python -m rag.daemons.url_watcher
```

### Monitoring

**Check daemon status:**
```bash
# Inside container
supervisorctl status url_watcher

# View logs
docker logs agentks-backend-1 | grep url_watcher
```

**Database queries:**
```sql
-- Count by status
SELECT status, COUNT(*) 
FROM urls 
GROUP BY status;

-- Recently failed
SELECT url, last_error, last_fetched_at
FROM urls
WHERE status = 'failed'
ORDER BY last_fetched_at DESC
LIMIT 10;

-- Stale URLs (not checked recently)
SELECT url, last_fetched_at
FROM urls
WHERE status = 'ingested'
  AND last_fetched_at < now() - INTERVAL '6 hours'
ORDER BY last_fetched_at;
```

### Integration with RAG Services

The daemon works seamlessly with other RAG services:

1. **rag_common**: Uses shared embeddings and database utilities
2. **rag_injector** (port 4002): Could also be used for manual injection
3. **rag_mcp** (port 4001): Users query the injected documents

### Content Processing

**Steps:**
1. Fetch URL content via HTTP
2. Compute SHA256 hash for change detection
3. Split content into chunks (RecursiveCharacterTextSplitter)
4. Generate embeddings using group's embed_model
5. Store in `rag_documents` table
6. Store vectors in PGVector collection
7. Update `rag_groups.doc_count`

### Error Handling

**Common errors and solutions:**

| Error | Cause | Solution |
|-------|-------|----------|
| `Failed to fetch URL` | Network error, invalid URL | Check URL accessibility, network config |
| `Failed to get vector store` | Missing embedding model | Ensure Ollama is running with required model |
| `Duplicate key violation` | URL already exists | Normal, handled gracefully |
| `Database connection error` | PostgreSQL unavailable | Check database status |

### Performance Considerations

- **Batch Size**: Larger batches = more throughput, but longer processing time
- **Chunk Size**: Smaller chunks = more precise search, but more storage
- **Check Interval**: Shorter intervals = fresher content, but more load
- **Concurrent Workers**: Multiple instances can run safely with database locks

### Future Enhancements

- [ ] Support for different content types (PDF, DOCX, etc.)
- [ ] Sitemap crawling for bulk URL discovery
- [ ] Robots.txt compliance
- [ ] Rate limiting per domain
- [ ] Priority queue for important URLs
- [ ] Webhook notifications on completion
- [ ] Advanced content extraction (main content only)
- [ ] Link extraction and recursive crawling

### Troubleshooting

**Daemon not processing:**
```bash
# Check if running
supervisorctl status url_watcher

# Restart
supervisorctl restart url_watcher

# Check for queued URLs
psql $DATABASE_URL -c "SELECT COUNT(*) FROM urls WHERE status = 'queued';"
```

**URLs stuck in 'queued':**
- Check daemon logs for errors
- Verify network connectivity
- Test URL manually: `curl -I <url>`

**High failure rate:**
- Check `last_error` field in database
- Verify Ollama service is running
- Check embedding model availability

## Related Documentation

- [RAG Common Module](../rag_common.py)
- [RAG Injection Service](../rag_injector/README.md)
- [RAG MCP Service](../rag_mcp/README.md)
