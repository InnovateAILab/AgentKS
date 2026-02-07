# Tools Module (MCP Integration)

This module provides utilities and functions for working with MCP servers and tool discovery in the AgentKS system.

## Structure

```
tools/
├── __init__.py                    # Module exports
├── client.py                      # MCP client utilities
├── discovery.py                   # Tool discovery from MCP servers
├── models.py                      # Pydantic models for MCP data
├── tool_discovery.py              # Tool semantic search and indexing
├── watcher.py                     # MCP watcher daemon (monitors & auto-discovers tools)
├── README.md                      # This file
├── TOOL_DISCOVERY_GUIDE.md        # Comprehensive tool discovery guide
├── TOOL_MCP_COMBINATION_GUIDE.md  # Why combine tool and MCP descriptions
├── MCP_WATCHER_SUMMARY.md         # Watcher implementation guide
├── MCP_WATCHER_ARCHITECTURE.md    # System architecture diagrams
├── MCP_WATCHER_QUICKREF.md        # Quick reference guide
└── MCP_WATCHER_MIGRATION.md       # Migration guide
```

## Components

### client.py

MCP client utilities for connecting to and invoking MCP tools.

**Functions:**
- `run_mcp_tool_async(mcp_url, headers, tool_name, payload)`: Connect to an MCP server and invoke a specific tool

**Example:**
```python
from tools import run_mcp_tool_async

result = await run_mcp_tool_async(
    mcp_url="http://mcp-server:5000/mcp",
    headers={"Authorization": "Bearer token"},
    tool_name="arxiv_search",
    payload={"query": "quantum computing"}
)
```

### discovery.py

Functions for discovering tools from MCP servers.

**Functions:**
- `discover_mcp_tools_async(mcp)`: Async function to discover tools from an MCP server
- `discover_mcp_tools(mcp)`: Synchronous wrapper for `discover_mcp_tools_async`

**Example:**
```python
from tools import discover_mcp_tools

mcp_config = {
    "endpoint": "http://mcp-server:5000/mcp",
    "auth": {
        "type": "bearer",
        "token": "secret-token"
    }
}

tools = discover_mcp_tools(mcp_config)
# Returns: [{"name": "tool1", "description": "...", "inputSchema": {...}}, ...]
```

### models.py

Pydantic models for MCP-related data structures.

**Models:**
- `MCPSyncRequest`: Request model for syncing tools from an MCP server
- `MCPServerConfig`: Configuration for an MCP server

**Example:**
```python
from tools.models import MCPServerConfig

mcp = MCPServerConfig(
    id="mcp-001",
    name="Research Tools",
    endpoint="http://mcp:5000/mcp",
    description="Tools for academic research",
    context="Physics and mathematics",
    resource="arXiv, INSPIRE-HEP",
    status="enabled"
)
```

### tool_discovery.py

Tool discovery and semantic search utilities for finding and ranking tools based on queries.

**Key Functions:**
- `discover_tools(query, user_scope, top_k, enabled_only, tags, min_score)`: Semantic search for tools
- `discover_tools_hybrid(query, semantic_weight, keyword_weight)`: Combined semantic + keyword scoring
- `bind_discovered_tools_to_llm(llm, discovered_tools)`: Bind tools to LangChain LLM
- `index_tool_with_mcp_context(...)`: Index tool with MCP context for richer search
- `index_tool_simple(...)`: Simple tool indexing without MCP context
- `reindex_all_tools()`: Re-index all tools with MCP context

**Example:**
```python
from tools import discover_tools, bind_discovered_tools_to_llm
from app.llms import get_llm

# Discover relevant tools
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

See `TOOL_DISCOVERY_GUIDE.md` and `TOOL_MCP_COMBINATION_GUIDE.md` for comprehensive documentation.

### watcher.py

Background daemon that monitors the `mcps` table and automatically discovers and registers tools from MCP servers.

**Key Functions:**
- `claim_pending_mcps(limit)`: Claim MCPs needing tool discovery
- `discover_mcp_tools(mcp)`: Discover tools from MCP server (imported from discovery.py)
- `register_tools_for_mcp(mcp, tools)`: Register discovered tools in database
- `process_mcp(mcp)`: Process a single MCP server
- `main_loop()`: Main daemon loop

**Usage:**
```bash
# Run as daemon via supervisord
# Configured in supervisord.conf as [program:mcp_watcher]

# Or run manually for testing
cd backend/backend_app
python -m tools.watcher
```

**Configuration:**
```bash
MCP_CHECK_INTERVAL=60  # Check interval in seconds
MCP_CLAIM_LIMIT=5      # Max MCPs to process per cycle
DATABASE_URL=postgresql+psycopg://...
```

See the comprehensive documentation files in this directory for more details:
- `MCP_WATCHER_SUMMARY.md` - Implementation guide and features
- `MCP_WATCHER_ARCHITECTURE.md` - System architecture and data flows
- `MCP_WATCHER_QUICKREF.md` - Quick reference and troubleshooting
- `MCP_WATCHER_MIGRATION.md` - Migration guide for existing tools

## Usage

### Basic Import

```python
# Import everything
from tools import (
    run_mcp_tool_async,
    discover_mcp_tools,
    discover_tools,
    discover_tools_hybrid,
    bind_discovered_tools_to_llm,
    MCPSyncRequest,
    MCPServerConfig
)

# Or import specific components
from mcp.client import run_mcp_tool_async
from mcp.discovery import discover_mcp_tools
from mcp.tool_discovery import discover_tools, bind_discovered_tools_to_llm
from tools.models import MCPServerConfig
```

### Discovering Tools from MCP

```python
import asyncio
from tools import discover_mcp_tools_async

async def main():
    mcp_config = {
        "name": "Basic Tools MCP",
        "endpoint": "http://basic_tools_mcp_service:5000/mcp",
        "auth": None  # No auth required
    }
    
    tools = await discover_mcp_tools_async(mcp_config)
    
    for tool in tools:
        print(f"Tool: {tool['name']}")
        print(f"Description: {tool['description']}")
        print(f"Schema: {tool['inputSchema']}")
        print()

asyncio.run(main())
```

### Invoking MCP Tools

```python
import asyncio
from tools import run_mcp_tool_async

async def search_papers():
    result = await run_mcp_tool_async(
        mcp_url="http://basic_tools_mcp_service:5000/mcp",
        headers={},
        tool_name="arxiv_search",
        payload={
            "query": "quantum computing",
            "max_results": 5
        }
    )
    return result

result = asyncio.run(search_papers())
print(result)
```

### Semantic Tool Search

```python
from tools import discover_tools, bind_discovered_tools_to_llm
from app.llms import get_llm

# Find tools relevant to a query using semantic search
tools = discover_tools(
    query="I need to search for physics papers",
    user_scope="global",
    top_k=5,
    enabled_only=True,
    tags=["physics", "research"],  # optional filtering
    min_score=0.5
)

print(f"Found {len(tools)} relevant tools:")
for tool in tools:
    print(f"- {tool['name']}: {tool['description']} (score: {tool['score']:.2f})")

# Bind discovered tools to LLM for agent use
llm = get_llm()
llm_with_tools = bind_discovered_tools_to_llm(llm, tools)

# Now the LLM can use these tools
response = llm_with_tools.invoke("Find recent papers about quantum computing")
```

### Hybrid Search (Semantic + Keyword)

```python
from tools import discover_tools_hybrid

# Combine semantic understanding with keyword matching
tools = discover_tools_hybrid(
    query="arxiv physics search",
    semantic_weight=0.7,  # 70% semantic relevance
    keyword_weight=0.3,   # 30% keyword matching
    top_k=5
)
```

## Integration Points

### With MCP Watcher Daemon

The MCP watcher daemon (`mcp.watcher`) uses this module to:
1. Discover tools from MCP servers (`discover_mcp_tools`)
2. Register discovered tools in the database
3. Index tools for semantic search

```python
from mcp.discovery import discover_mcp_tools

# In watcher.py
tools = discover_mcp_tools(mcp_config)
for tool in tools:
    register_tool_in_db(tool)
    index_tool_for_search(tool)
```

### With Backend API

The main backend API (`app/main.py`) uses this module to:
1. Execute MCP tools on demand (`run_mcp_tool_async`)
2. Sync tools from MCP servers (via `MCPSyncRequest`)

```python
from tools import run_mcp_tool_async
from tools.models import MCPSyncRequest

# Execute MCP tool
result = await run_mcp_tool_async(
    mcp_url=tool_config["endpoint"],
    headers=auth_headers,
    tool_name=tool_name,
    payload=tool_input
)
```

### With Tool Discovery

The tool discovery system (`app/tool_discovery.py`) indexes tools discovered by this module for semantic search.

## Authentication

MCP servers can be configured with various authentication methods:

### Bearer Token

```python
mcp_config = {
    "endpoint": "http://mcp:5000",
    "auth": {
        "type": "bearer",
        "token": "my-secret-token"
    }
}
```

### Custom Headers

```python
mcp_config = {
    "endpoint": "http://mcp:5000",
    "auth": {
        "type": "token",
        "token": "my-token",
        "headers": {
            "X-API-Key": "api-key-123",
            "X-Client-ID": "client-456"
        }
    }
}
```

### No Authentication

```python
mcp_config = {
    "endpoint": "http://mcp:5000",
    "auth": None
}
```

## Error Handling

All functions in this module raise exceptions on errors:

```python
from tools import discover_mcp_tools

try:
    tools = discover_mcp_tools(mcp_config)
except RuntimeError as e:
    # langchain_mcp_adapters not available
    print(f"MCP client not available: {e}")
except ValueError as e:
    # Tool not found
    print(f"Tool error: {e}")
except Exception as e:
    # Connection errors, timeouts, etc.
    print(f"Discovery failed: {e}")
```

## Dependencies

This module requires:
- `langchain_mcp_adapters`: For MCP client functionality
- `pydantic`: For data models
- `asyncio`: For async operations

Install with:
```bash
pip install langchain-mcp-adapters pydantic
```

## Testing

### Unit Tests

```python
import pytest
from tools import discover_mcp_tools
from tools.models import MCPServerConfig

def test_mcp_server_config():
    config = MCPServerConfig(
        id="test",
        name="Test MCP",
        endpoint="http://test:5000"
    )
    assert config.id == "test"
    assert config.status == "pending"  # default

@pytest.mark.asyncio
async def test_discover_tools():
    # Mock MCP server
    mcp = {"endpoint": "http://mock:5000", "auth": None}
    # Would need actual mock here
    tools = await discover_mcp_tools_async(mcp)
    assert isinstance(tools, list)
```

### Integration Tests

```bash
# Start test MCP server
docker-compose up -d basic_tools_mcp_service

# Run tests
pytest tests/test_mcp.py -v
```

## Troubleshooting

### ImportError: langchain_mcp_adapters not found

**Solution:** Install the package:
```bash
pip install langchain-mcp-adapters
```

### RuntimeError: MCP client not available

**Solution:** Ensure `langchain_mcp_adapters` is installed and importable.

### Connection refused / Timeout

**Solution:** 
1. Check MCP server is running: `curl http://mcp-server:5000/health`
2. Verify endpoint URL is correct
3. Check firewall/network settings
4. Review MCP server logs

### Tool not found

**Solution:**
1. List available tools: `discover_mcp_tools(mcp_config)`
2. Verify tool name matches exactly (case-sensitive)
3. Check MCP server has the tool registered

## Related Documentation

**In this directory:**
- `TOOL_DISCOVERY_GUIDE.md` - Comprehensive tool discovery system documentation
- `TOOL_MCP_COMBINATION_GUIDE.md` - Why and how to combine tool and MCP descriptions
- `MCP_WATCHER_SUMMARY.md` - Complete implementation guide
- `MCP_WATCHER_ARCHITECTURE.md` - System architecture diagrams
- `MCP_WATCHER_QUICKREF.md` - Quick reference guide
- `MCP_WATCHER_MIGRATION.md` - Migration guide for existing tools

**Other modules:**
- `../rag/daemons/README.md` - RAG daemons overview

## Contributing

When adding new MCP-related functionality:

1. Add core logic to appropriate module (`client.py`, `discovery.py`, `models.py`)
2. Export new functions/classes in `__init__.py`
3. Update this README with usage examples
4. Add tests for new functionality
5. Update related documentation

## Future Enhancements

- [ ] Add caching layer for discovered tools
- [ ] Support WebSocket transport for MCP
- [ ] Add health check utilities
- [ ] Implement tool versioning
- [ ] Add MCP server discovery (auto-find MCP servers on network)
- [ ] Support batch tool invocation
- [ ] Add metrics/monitoring hooks
