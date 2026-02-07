# RAG MCP Server

A Model Context Protocol (MCP) server for RAG (Retrieval-Augmented Generation) operations.
Provides tools to search and retrieve documents from the knowledge base using vector similarity
and database queries.

## ⚠️ Important: Embedding Model Compatibility

**Different embedding models are NOT compatible with each other!**

- Each RAG group tracks which embedding model was used to create its embeddings
- You MUST use the same model for both document ingestion and query search
- Changing models requires re-embedding ALL documents in that group
- **Always specify `rag_group` in searches** to ensure model consistency

See [EMBEDDING_MODELS.md](./EMBEDDING_MODELS.md) for detailed information about managing embedding models.

## Features

- **Protocol Compliant**: Uses FastMCP for proper MCP JSON-RPC 2.0 protocol
- **HTTP/SSE Transport**: Accessible via HTTP with streaming support (port 4001)
- **Vector Similarity Search**: Semantic search using embeddings and pgvector
- **Embedding Model Tracking**: Each RAG group uses a specific embedding model
- **Database Queries**: Direct SQL queries for precise filtering
- **Group Management**: Organize documents into named collections (RAG groups)
- **Metadata Resources**: Expose knowledge base statistics and group information

## Tools

### 1. `rag_search` - Vector Similarity Search

Performs semantic search across the knowledge base using embeddings.

**⚠️ IMPORTANT:** Always specify `rag_group` to ensure the correct embedding model is used.
Searching without a group may mix results from incompatible embedding models.

**Parameters:**
- `query` (string, required): Search query or question
- `k` (int, optional): Number of results to return (default: 5, max: 20)
- `rag_group` (string, **RECOMMENDED**): Filter by RAG group name - ensures model consistency
- `score_threshold` (float, optional): Minimum similarity score 0.0-1.0

**Example:**
```json
{
  "query": "What is machine learning?",
  "k": 5,
  "rag_group": "ml-docs"
}
```

**Response includes:**
- Search results with content and metadata
- `embedding_model`: Which model was used (confirms compatibility)
- `similarity_score`: Relevance score for each result

### 2. `rag_query` - Database Query

Structured queries on rag_documents table with pattern matching.

**Parameters:**
- `rag_group` (string, optional): Filter by RAG group name
- `title_pattern` (string, optional): SQL LIKE pattern for title
- `content_pattern` (string, optional): SQL LIKE pattern for content
- `limit` (int, optional): Maximum results (default: 10, max: 50)

**Example:**
```json
{
  "rag_group": "python-docs",
  "title_pattern": "%async%",
  "limit": 10
}
```

### 3. `rag_get_document` - Get Document by ID

Retrieves full document content and metadata.

**Parameters:**
- `document_id` (string, required): Document ID

**Example:**
```json
{
  "document_id": "d1"
}
```

### 4. `rag_list_groups` - List RAG Groups

Lists all RAG groups (document collections) with statistics.

**Parameters:**
- `scope` (string, optional): Filter by scope (default: "global")
- `owner` (string, optional): Filter by owner

**Example:**
```json
{
  "scope": "global"
}
```

### 5. `rag_get_group_documents` - Get Documents in Group

Retrieves all documents in a specific RAG group.

**Parameters:**
- `rag_group_name` (string, required): RAG group name
- `limit` (int, optional): Maximum documents (default: 20, max: 100)

**Example:**
```json
{
  "rag_group_name": "api-reference",
  "limit": 50
}
```

## Resources

### `rag://metadata`

Returns metadata about the RAG knowledge base including:
- Total groups and documents
- Embedding model configuration
- Top groups by document count

### `rag://groups`

Returns complete list of all RAG groups with details.

## Running the Server

### Via Supervisord (Integrated with Backend)

The RAG MCP service runs as a daemon managed by supervisord within the backend container.

**Deployment:**
```bash
# Build and start the backend (includes RAG MCP)
docker compose up -d backend

# Check logs for all services
docker compose logs -f backend

# Check RAG MCP specifically
docker compose exec backend supervisorctl status rag_mcp
docker compose exec backend supervisorctl tail -f rag_mcp

# Server will be available at http://localhost:4001
```

**Supervisord Configuration:**

The service is configured in `/app/supervisord.conf`:
```ini
[program:rag_mcp]
command=python -u rag_mcp/main.py
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stderr_logfile=/dev/stderr
startretries=3
environment=PYTHONUNBUFFERED="1"
```

**Manage the service:**
```bash
# Stop RAG MCP
docker compose exec backend supervisorctl stop rag_mcp

# Start RAG MCP
docker compose exec backend supervisorctl start rag_mcp

# Restart RAG MCP
docker compose exec backend supervisorctl restart rag_mcp

# View status of all services
docker compose exec backend supervisorctl status
```

### Local Development

```bash
cd backend/backend_app/rag_mcp

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/dbname"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_EMBED_MODEL="nomic-embed-text"
export COLLECTION_DOCS="document_embeddings"

# Run server
python main.py
```

Server runs at http://localhost:4001

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | ✅ Yes | - | PostgreSQL connection string with psycopg |
| `OLLAMA_BASE_URL` | No | `http://ollama:11434` | Ollama service URL |
| `OLLAMA_EMBED_MODEL` | No | `nomic-embed-text` | Embedding model name |
| `COLLECTION_DOCS` | No | `document_embeddings` | PGVector collection name |

### Database Schema

The service uses these tables:

**rag_groups** - Document collections
- `id`: Primary key
- `name`: Group name (unique per scope)
- `scope`: Scope for multi-tenancy (default: "global")
- `owner`: Optional owner identifier
- `description`: Group description
- `embed_model`: Embedding model used
- `doc_count`: Number of documents

**rag_documents** - Individual documents
- `id`: Primary key
- `rag_group_id`: Foreign key to rag_groups
- `url_id`: Optional foreign key to source URL
- `title`: Document title
- `content`: Full text content
- `content_hash`: Hash for deduplication
- `metadata`: JSONB for flexible metadata
- `created_at`, `updated_at`: Timestamps

**PGVector collection** - Vector embeddings
- Stores document embeddings for similarity search
- Indexed with HNSW for fast retrieval

## Usage with MCP Clients

### Python Client Example

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async with stdio_client(
    StdioServerParameters(
        command="python",
        args=["main.py"],
        env={
            "DATABASE_URL": "postgresql+psycopg://...",
            "OLLAMA_BASE_URL": "http://localhost:11434"
        }
    )
) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # Vector search
        result = await session.call_tool("rag_search", {
            "query": "machine learning concepts",
            "k": 5
        })
        print(result)
        
        # Get group documents
        result = await session.call_tool("rag_get_group_documents", {
            "rag_group_name": "ml-docs",
            "limit": 10
        })
        print(result)
```

### HTTP/SSE Client Example

```python
import httpx
import json

async with httpx.AsyncClient() as client:
    # Call tool via HTTP
    response = await client.post(
        "http://localhost:4001/sse",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "rag_search",
                "arguments": {
                    "query": "Python async programming",
                    "k": 5
                }
            },
            "id": 1
        }
    )
    result = response.json()
    print(result)
```

### OpenWebUI Integration

In OpenWebUI admin settings:

1. Navigate to Admin → Settings → Tools
2. Click "Add MCP Server"
3. Configure:
   - Name: `RAG Knowledge Base`
   - Type: `HTTP/SSE`
   - Endpoint: `http://rag_mcp_service:4001/sse` (or `http://localhost:4001/sse`)
4. Save and test connection

The RAG tools will appear in the chat interface for knowledge retrieval.

## Discovery Endpoint

The server exposes metadata at `/.well-known/mcp`:

```bash
curl http://localhost:4001/.well-known/mcp
```

Returns:
```json
{
  "name": "rag-mcp",
  "version": "1.0.0",
  "description": "RAG knowledge base retrieval service",
  "capabilities": {
    "tools": [...],
    "resources": [...],
    "prompts": [...]
  },
  "configuration": {...},
  "environment": {...}
}
```

## Architecture

### Vector Search Flow

1. User query → Embedding generation (Ollama)
2. Vector similarity search in PGVector
3. Results ranked by cosine similarity
4. Return top-k documents with scores

### Database Query Flow

1. Build SQL query with filters
2. Execute on rag_documents table
3. Join with rag_groups for metadata
4. Return structured results

### Group Organization

Documents are organized into RAG groups (collections):
- Each group has a unique name per scope
- Groups track document counts and embed model
- Supports multi-tenancy via scope field
- Optional owner association

## Performance

- **Vector Search**: O(log n) with HNSW index
- **Database Queries**: Indexed on rag_group_id
- **Concurrent Requests**: Async/await support
- **Connection Pooling**: psycopg built-in

## Monitoring

Check server health:
```bash
curl http://localhost:4001/
```

View knowledge base stats:
```bash
curl http://localhost:4001/mcp/resources/rag://metadata
```

## Troubleshooting

### "Vector search not available"

- Ensure LangChain dependencies are installed
- Check Ollama service is running and accessible
- Verify DATABASE_URL has pgvector extension enabled

### "Document not found"

- Check document ID exists in rag_documents table
- Verify RAG group scope matches query

### Connection errors

- Verify DATABASE_URL format: `postgresql+psycopg://user:pass@host:5432/db`
- Check PostgreSQL is running and accessible
- Ensure database has required tables (run migrations)

## Development

### Adding New Tools

```python
@mcp.tool()
def my_new_tool(param: str) -> str:
    """Tool description."""
    # Implementation
    return json.dumps(result)
```

### Adding Resources

```python
@mcp.resource("rag://custom")
def custom_resource():
    """Resource description."""
    return json.dumps(data)
```

## Related Services

The backend container runs multiple services via supervisord:

- **web** (port 8000): Admin web UI for managing URLs, RAG groups, MCPs, LLMs
- **app** (port 4000): Main FastAPI application with RAG ingestion and chat API
- **rag_mcp** (port 4001): RAG retrieval MCP service (this service)
- **url_watcher**: Background daemon for monitoring and fetching URLs

External services:
- **basic_tools_mcp_service** (port 5000): General search and utility tools
- **postgres**: Database with pgvector extension
- **ollama**: LLM and embedding generation service

## Architecture

```
┌─────────────────────────────────────────────────────┐
│          Backend Container (supervisord)            │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │   web    │  │   app    │  │    rag_mcp       │ │
│  │ :8000    │  │ :4000    │  │    :4001         │ │
│  │ Admin UI │  │ Chat API │  │ RAG Retrieval    │ │
│  └──────────┘  └──────────┘  └──────────────────┘ │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │         url_watcher (background)               │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
      ┌─────────────────────────────────────┐
      │  PostgreSQL + pgvector              │
      │  • rag_groups, rag_documents        │
      │  • document_embeddings collection   │
      └─────────────────────────────────────┘
                         │
                         ▼
      ┌─────────────────────────────────────┐
      │  Ollama                             │
      │  • nomic-embed-text (embeddings)    │
      │  • llama2:7b (chat)                 │
      └─────────────────────────────────────┘
```

## License

Part of the AgentKS project.
