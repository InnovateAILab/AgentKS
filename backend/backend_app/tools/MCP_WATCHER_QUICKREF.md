# MCP Watcher Quick Reference

## TL;DR

The MCP watcher daemon automatically discovers and registers tools from MCP servers. Just add an MCP server via the Admin UI, and tools appear automatically within 60 seconds.

## Quick Start

### Add an MCP Server

**Via Admin UI:**
1. Navigate to `/admin/mcps/add`
2. Fill in the form:
   ```
   Name: My MCP Service
   Endpoint: http://my-mcp:5000/mcp
   Kind: http
   Description: My custom tools
   Status: pending
   ```
3. Submit
4. Wait 60 seconds
5. Tools appear in `/admin/tools`

**Via SQL:**
```sql
INSERT INTO mcps (id, name, endpoint, kind, status, description, created_at)
VALUES (
  gen_random_uuid()::text,
  'My MCP Service',
  'http://my-mcp:5000/mcp',
  'http',
  'pending',
  'My custom tools',
  now()
);
```

### Check Status

```bash
# View daemon logs
docker-compose logs -f backend | grep mcp_watcher

# Check MCP status in database
docker-compose exec postgres psql -U user -d dbname -c \
  "SELECT id, name, status, last_checked_at, last_error FROM mcps;"

# Count tools per MCP
docker-compose exec postgres psql -U user -d dbname -c \
  "SELECT m.name, COUNT(t.id) FROM mcps m 
   LEFT JOIN tools t ON t.mcp_id = m.id 
   GROUP BY m.name;"
```

## Configuration

**Environment Variables (in docker-compose.yml):**
```yaml
environment:
  DATABASE_URL: postgresql+psycopg://user:pass@postgres:5432/db
  MCP_CHECK_INTERVAL: "60"  # seconds between checks
  MCP_CLAIM_LIMIT: "5"      # max MCPs per cycle
```

## Common Tasks

### Force Re-discovery

```sql
-- Reset MCP to trigger re-discovery
UPDATE mcps SET status = 'pending', last_checked_at = NULL WHERE id = 'mcp_id';
```

### Disable Auto-discovery

```sql
-- Set status to 'disabled' to skip
UPDATE mcps SET status = 'disabled' WHERE id = 'mcp_id';
```

### Manual Tool Registration

```python
from rag.daemons.mcp_watcher import tool_upsert, index_tool_with_mcp_context

# Register tool
tool_upsert(
    tool_id="custom_tool_1",
    name="my_tool",
    kind="mcp_tool",
    mcp_id="mcp123",
    metadata={
        "description": "My custom tool",
        "enabled": True,
        "scope": "global",
        "config": {}
    },
    tags=["custom"]
)

# Index for search
index_tool_with_mcp_context(
    tool_id="custom_tool_1",
    name="my_tool",
    description="My custom tool",
    enabled=True,
    scope="global",
    mcp_description="MCP provider description",
    mcp_context="Usage context",
    mcp_resource="Available resources"
)
```

## MCP Authentication

**Bearer Token:**
```sql
UPDATE mcps 
SET auth = '{"type": "bearer", "token": "my-secret-token"}'::jsonb
WHERE id = 'mcp_id';
```

**Custom Headers:**
```sql
UPDATE mcps 
SET auth = '{"type": "token", "token": "my-token", "headers": {"X-API-Key": "key123"}}'::jsonb
WHERE id = 'mcp_id';
```

## Troubleshooting

### MCP stuck in 'processing'
**Cause:** Daemon crashed during processing
**Fix:**
```sql
UPDATE mcps SET status = 'pending' WHERE status = 'processing';
```

### No tools discovered
**Possible causes:**
- MCP endpoint not accessible
- Invalid auth credentials
- MCP server error

**Debug:**
```bash
# Check daemon logs
docker-compose logs backend | grep "mcp_watcher"

# Test MCP endpoint manually
curl -H "Authorization: Bearer token" http://mcp-endpoint:5000/mcp/tools

# Check last error
docker-compose exec postgres psql -U user -d dbname -c \
  "SELECT name, last_error FROM mcps WHERE status = 'error';"
```

### Tools not appearing in search
**Debug:**
```bash
# Check if tools were registered
docker-compose exec postgres psql -U user -d dbname -c \
  "SELECT COUNT(*) FROM tools WHERE mcp_id = 'mcp_id';"

# Check if tools were indexed
docker-compose exec postgres psql -U user -d dbname -c \
  "SELECT COUNT(*) FROM langchain_pg_embedding e
   JOIN langchain_pg_collection c ON e.collection_id = c.uuid
   WHERE c.name = 'tool_embeddings';"

# Re-index all tools
docker-compose exec backend python -c "
from app.tool_discovery import reindex_all_tools
reindex_all_tools()
"
```

## Status Codes

| Status | Meaning | Action |
|--------|---------|--------|
| `pending` | Awaiting discovery | Daemon will process |
| `processing` | Currently being processed | Wait or reset if stuck |
| `enabled` | Successfully processed | Tools available |
| `error` | Discovery failed | Check last_error, fix config |
| `disabled` | Manually disabled | Enable to resume |

## Tool ID Format

Tools are assigned deterministic IDs:
```
{mcp_id}_{tool_name}

Examples:
- mcp123_arxiv_search
- hep-mcp_inspirehep_search
- custom-mcp_my_tool
```

**Benefits:**
- No duplicates (upsert on conflict)
- Easy to identify tool source
- Clean deletion when MCP removed

## API Endpoints

### Check MCP Status
```bash
GET /admin/mcps
# Returns list of MCPs with status
```

### List Tools
```bash
GET /admin/tools
# Returns all registered tools including MCP tools
```

### Force Refresh
```bash
POST /admin/mcps/{id}/refresh
# Manually trigger tool discovery for specific MCP
# (TODO: implement if needed)
```

## Database Schema

```sql
-- MCPs table
CREATE TABLE mcps (
    id TEXT PRIMARY KEY,
    name TEXT,
    endpoint TEXT,
    kind TEXT,
    status TEXT DEFAULT 'enabled',
    description TEXT,
    resource TEXT,
    context TEXT,
    auth JSONB,
    last_checked_at TIMESTAMP,
    last_error TEXT,
    metadata JSONB,
    tags JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Tools table (tools reference MCPs)
CREATE TABLE tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT,
    mcp_id TEXT REFERENCES mcps(id) ON DELETE SET NULL,
    metadata JSONB,
    tags JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

## Performance Tips

### Optimize Discovery
- Set `MCP_CHECK_INTERVAL` higher (e.g., 300s) if MCPs don't change often
- Set `MCP_CLAIM_LIMIT` lower if discovery is slow
- Use caching in MCP servers for faster responses

### Monitor Resource Usage
```bash
# Check daemon memory usage
docker stats backend

# Check database connections
docker-compose exec postgres psql -U user -d dbname -c \
  "SELECT COUNT(*) FROM pg_stat_activity WHERE application_name LIKE '%mcp_watcher%';"
```

### Batch Operations
The daemon already uses batch claiming via `MCP_CLAIM_LIMIT`. Increase this value to process more MCPs per cycle (but be careful of timeout issues).

## Integration Examples

### Use Discovered Tools in Code

```python
from app.tool_discovery import discover_tools, bind_discovered_tools_to_llm
from app.llms import get_llm

# Discover tools relevant to query
tools = discover_tools(
    query="search for physics papers",
    user_scope="global",
    top_k=5,
    enabled_only=True
)

# Bind to LLM
llm = get_llm()
llm_with_tools = bind_discovered_tools_to_llm(llm, tools)

# Use in agent
response = llm_with_tools.invoke("Find papers about quantum computing")
```

### Create Custom MCP Server

```python
# Your MCP server must implement:
# GET /mcp/tools
# Returns: [{"name": "tool_name", "description": "...", "inputSchema": {...}}]

from fastapi import FastAPI

app = FastAPI()

@app.get("/mcp/tools")
def list_tools():
    return [
        {
            "name": "my_custom_tool",
            "description": "Does something useful",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ]

@app.post("/mcp/tools/{tool_name}")
def execute_tool(tool_name: str, payload: dict):
    # Execute tool logic
    return {"result": "success"}
```

## Best Practices

1. **Use descriptive MCP names**: Makes tools easier to find
2. **Include rich descriptions**: Improves semantic search
3. **Tag MCPs appropriately**: Enables filtering
4. **Set auth properly**: Prevents discovery failures
5. **Monitor logs**: Catch issues early
6. **Periodic review**: Remove unused MCPs
7. **Test endpoints**: Verify accessibility before adding
8. **Use HTTPS in production**: Secure MCP connections
9. **Version your MCPs**: Track tool changes over time
10. **Document tools**: Include usage examples in descriptions

## Security Considerations

- Store sensitive auth tokens in environment variables
- Use encrypted connections (HTTPS) for MCP endpoints
- Validate tool inputs before execution
- Implement rate limiting on MCP servers
- Audit tool usage via `tool_runs` table
- Restrict admin-only tools via `scope='admin_only'`

## Further Reading

- `MCP_WATCHER_SUMMARY.md` - Detailed implementation guide
- `MCP_WATCHER_ARCHITECTURE.md` - System architecture diagrams
- `README.md` - General daemon documentation
- `TOOL_DISCOVERY_GUIDE.md` - Tool discovery system guide
- `TOOL_MCP_COMBINATION_GUIDE.md` - Why/how to combine descriptions

---

**Questions?** Check the logs, database, or contact the dev team!
