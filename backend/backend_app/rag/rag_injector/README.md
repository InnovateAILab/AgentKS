# RAG Injection Service

A REST API service for injecting documents into the RAG (Retrieval-Augmented Generation) knowledge base. This service runs on **port 4002** and is managed by supervisord in the backend container.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Backend Container                        │
├─────────────────────────────────────────────────────────────┤
│  supervisord                                                 │
│  ├── web (8000)           - Main web API                     │
│  ├── app (4000)           - Admin UI                         │
│  ├── rag_mcp (4001)       - RAG MCP retrieval service       │
│  └── rag_injector (4002)  - RAG injection REST service      │
└─────────────────────────────────────────────────────────────┘
          │                          │
          ▼                          ▼
    ┌──────────┐              ┌──────────┐
    │ Postgres │              │ PGVector │
    │  + pgvector extension   │          │
    └──────────┘              └──────────┘
```

### Workflow

1. **Injection (port 4002)**: Users submit documents via REST API
2. **Storage**: Documents stored in `rag_documents` table
3. **Embedding**: Content chunked and embedded using group's embedding model
4. **Vector Storage**: Embeddings stored in PGVector for similarity search
5. **Retrieval (port 4001)**: RAG MCP service queries using same embedding model

## Key Features

- ✅ **REST API**: Standard HTTP endpoints for easy integration
- ✅ **Embedding Model Tracking**: Each RAG group uses a specific embedding model
- ✅ **Automatic Chunking**: RecursiveCharacterTextSplitter for optimal chunk sizes
- ✅ **Deduplication**: Content hash prevents duplicate documents
- ✅ **Batch Operations**: Inject multiple documents efficiently
- ✅ **Multi-tenant**: Scope and owner-based isolation
- ✅ **Metadata Support**: Custom metadata for each document

## Quick Start

### Option 1: Quick Inject (Easiest - One Call)

The simplest way to inject documents - creates the group automatically if it doesn't exist:

```bash
curl -X POST http://localhost:4002/quick-inject \
  -H "Content-Type: application/json" \
  -d '{
    "group_name": "research_papers",
    "title": "Quantum Entanglement Study",
    "content": "This paper presents a comprehensive study of quantum entanglement...",
    "metadata": {
      "author": "Dr. Smith",
      "year": 2024,
      "tags": ["quantum", "physics"]
    }
  }'
```

**Features:**
- Automatically creates RAG group if it doesn't exist
- Defaults scope to "global" if not provided
- Uses "nomic-embed-text" as default embedding model
- Handles deduplication automatically

**Response:**
```json
{
  "status": "success",
  "group_created": true,
  "rag_group": "research_papers",
  "rag_group_id": "550e8400-e29b-41d4-a716-446655440000",
  "scope": "global",
  "embed_model": "nomic-embed-text",
  "document": {
    "id": "doc-12345",
    "title": "Quantum Entanglement Study",
    "content_hash": "a1b2c3d4...",
    "chunks_created": 5,
    "created_at": "2024-01-15T10:35:00"
  }
}
```

### Option 2: Manual Group Creation (More Control)

#### 1. Create a RAG Group

Before injecting documents, create a RAG group with its embedding model:

```bash
curl -X POST http://localhost:4002/groups \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_papers",
    "scope": "global",
    "description": "Physics research papers",
    "embed_model": "nomic-embed-text"
  }'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "research_papers",
  "scope": "global",
  "owner": null,
  "description": "Physics research papers",
  "embed_model": "nomic-embed-text",
  "doc_count": 0,
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:30:00"
}
```

#### 2. Inject a Document

```bash
curl -X POST http://localhost:4002/inject/research_papers \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Quantum Entanglement Study",
    "content": "This paper presents a comprehensive study of quantum entanglement...",
    "metadata": {
      "author": "Dr. Smith",
      "year": 2024,
      "tags": ["quantum", "physics"]
    },
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

Response:
```json
{
  "id": "doc-12345",
  "rag_group_id": "550e8400-e29b-41d4-a716-446655440000",
  "url_id": null,
  "title": "Quantum Entanglement Study",
  "content_hash": "a1b2c3d4...",
  "metadata": {
    "author": "Dr. Smith",
    "year": 2024,
    "tags": ["quantum", "physics"]
  },
  "created_at": "2024-01-15T10:35:00",
  "chunks_created": 5
}
```

#### 3. List Documents

```bash
curl "http://localhost:4002/documents/research_papers?limit=10&offset=0"
```

#### 4. Query via RAG MCP

Once documents are injected, use the RAG MCP service (port 4001) to search:

```bash
curl -X POST http://localhost:4001/rag_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quantum entanglement",
    "k": 5,
    "rag_group": "research_papers"
  }'
```

## API Reference

### Quick Inject (Convenience Endpoint)

#### `POST /quick-inject`
**🚀 Recommended for simple workflows**

Create RAG group (if not exists) and inject document in a single call.

**Request Body:**
```json
{
  // Group configuration
  "group_name": "string",              // Required: RAG group name
  "scope": "string",                   // Optional: defaults to "global" if null
  "owner": "string",                   // Optional: owner identifier
  "group_description": "string",       // Optional: group description
  "embed_model": "string",             // Optional: defaults to "nomic-embed-text"
  
  // Document content
  "title": "string",                   // Required: document title
  "content": "string",                 // Required: document content
  "url_id": "string",                  // Optional: source URL ID
  "metadata": {},                      // Optional: custom metadata
  "chunk_size": 1000,                  // Optional: chunk size (default: 1000)
  "chunk_overlap": 200                 // Optional: chunk overlap (default: 200)
}
```

**Behavior:**
1. If `scope` is `null` or omitted → defaults to `"global"`
2. If RAG group doesn't exist → creates it automatically with provided `embed_model`
3. If RAG group exists → uses existing group's `embed_model` (ignores request's `embed_model`)
4. Checks for duplicate content via hash → skips if duplicate found
5. Splits content into chunks and generates embeddings
6. Stores in database and vector store

**Response (Success):**
```json
{
  "status": "success",
  "group_created": true,               // Whether group was auto-created
  "rag_group": "research_papers",
  "rag_group_id": "550e8400-...",
  "scope": "global",
  "embed_model": "nomic-embed-text",
  "document": {
    "id": "doc-12345",
    "title": "Quantum Entanglement Study",
    "content_hash": "a1b2c3d4...",
    "chunks_created": 5,
    "created_at": "2024-01-15T10:35:00"
  }
}
```

**Response (Duplicate):**
```json
{
  "status": "skipped",
  "reason": "Document with same content already exists in this group",
  "group_created": false,
  "rag_group": "research_papers",
  "rag_group_id": "550e8400-...",
  "existing_document_id": "doc-67890"
}
```

**Example Usage:**

```bash
# Minimal request (uses defaults)
curl -X POST http://localhost:4002/quick-inject \
  -H "Content-Type: application/json" \
  -d '{
    "group_name": "my_docs",
    "title": "Meeting Notes",
    "content": "Today we discussed quantum computing applications..."
  }'

# Full request with all options
curl -X POST http://localhost:4002/quick-inject \
  -H "Content-Type: application/json" \
  -d '{
    "group_name": "user_documents",
    "scope": "user",
    "owner": "alice@example.com",
    "group_description": "Alice personal documents",
    "embed_model": "nomic-embed-text",
    "title": "Research Paper Draft",
    "content": "Abstract: This paper explores...",
    "metadata": {
      "author": "Alice",
      "version": "draft-1",
      "tags": ["research", "draft"]
    },
    "chunk_size": 800,
    "chunk_overlap": 150
  }'
```

**Use Cases:**
- Quick prototyping and testing
- Simple document upload interfaces
- Webhook handlers for automatic document ingestion
- URL crawler integration (auto-inject crawled content)
- Chat interfaces where users paste content

---

### RAG Group Management

#### `POST /groups`
Create a new RAG group.

**Request Body:**
```json
{
  "name": "string",           // Required: unique group name
  "scope": "string",          // Default: "global"
  "owner": "string",          // Optional: owner identifier
  "description": "string",    // Optional: group description
  "embed_model": "string"     // Required: embedding model (e.g., "nomic-embed-text")
}
```

**Response:** `RAGGroupResponse`

#### `GET /groups`
List all RAG groups with optional filtering.

**Query Parameters:**
- `scope` (default: "global"): Filter by scope
- `owner` (optional): Filter by owner

**Response:** Array of `RAGGroupResponse`

#### `GET /groups/{group_name}`
Get details of a specific RAG group.

**Query Parameters:**
- `scope` (default: "global"): Group scope

**Response:** `RAGGroupResponse`

#### `PATCH /groups/{group_name}`
Update RAG group metadata.

**Request Body:**
```json
{
  "description": "string",  // Optional
  "owner": "string"         // Optional
}
```

**Note:** Cannot change `embed_model` after creation. To use a different model, create a new group.

#### `DELETE /groups/{group_name}`
Delete a RAG group and all its documents.

**Query Parameters:**
- `scope` (default: "global"): Group scope

**Response:**
```json
{
  "status": "deleted",
  "group": "group_name"
}
```

### Document Injection

#### `POST /inject/{rag_group_name}`
Inject a single document into a RAG group.

**Request Body:**
```json
{
  "title": "string",          // Required: document title
  "content": "string",        // Required: document content
  "url_id": "string",         // Optional: source URL ID
  "metadata": {},             // Optional: custom metadata
  "chunk_size": 1000,         // Default: 1000
  "chunk_overlap": 200        // Default: 200
}
```

**Query Parameters:**
- `scope` (default: "global"): Group scope

**Response:** `DocumentResponse`

**Processing:**
1. Content is hashed for deduplication
2. Document stored in `rag_documents` table
3. Content split into chunks (RecursiveCharacterTextSplitter)
4. Each chunk embedded using group's `embed_model`
5. Embeddings stored in PGVector collection

#### `POST /inject/{rag_group_name}/batch`
Inject multiple documents in a single request.

**Request Body:**
```json
{
  "documents": [
    {
      "title": "string",
      "content": "string",
      "metadata": {},
      "chunk_size": 1000,
      "chunk_overlap": 200
    }
  ],
  "chunk_size": 1000,          // Default for documents without explicit size
  "chunk_overlap": 200         // Default for documents without explicit overlap
}
```

**Response:**
```json
{
  "rag_group": "group_name",
  "total": 10,
  "successful": 9,
  "failed": 1,
  "results": [
    {
      "document_id": "doc-123",
      "title": "Document 1",
      "status": "completed",
      "chunks_created": 5
    },
    {
      "title": "Document 2",
      "status": "skipped",
      "reason": "duplicate content"
    },
    {
      "title": "Document 3",
      "status": "failed",
      "error": "Processing error message"
    }
  ]
}
```

#### `GET /documents/{rag_group_name}`
List documents in a RAG group.

**Query Parameters:**
- `scope` (default: "global"): Group scope
- `limit` (default: 20): Max documents to return
- `offset` (default: 0): Pagination offset

**Response:**
```json
{
  "rag_group": "group_name",
  "total": 100,
  "limit": 20,
  "offset": 0,
  "documents": [
    {
      "id": "doc-123",
      "title": "Document Title",
      "content_hash": "a1b2c3...",
      "metadata": {},
      "created_at": "2024-01-15T10:35:00",
      "content_length": 5000
    }
  ]
}
```

#### `DELETE /documents/{document_id}`
Delete a specific document and its embeddings.

**Response:**
```json
{
  "status": "deleted",
  "document_id": "doc-123",
  "rag_group": "group_name"
}
```

### Health & Info

#### `GET /`
Service information and available endpoints.

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "langchain_available": true,
  "database_connected": true,
  "embed_model": "nomic-embed-text",
  "collection": "document_embeddings"
}
```

## Embedding Model Compatibility

⚠️ **CRITICAL**: Different embedding models produce incompatible vector representations.

### Key Principles

1. **One Model Per Group**: Each RAG group uses exactly one embedding model
2. **Model Tracked in Database**: `rag_groups.embed_model` stores the model name
3. **Consistent Embedding**: All documents in a group are embedded with the same model
4. **Retrieval Consistency**: RAG MCP service uses the group's model for searches

### Why This Matters

Different embedding models:
- Produce vectors of different dimensions (e.g., 768 vs 1536)
- Use different semantic spaces
- **Cannot be mixed** in similarity searches

### Example: Creating Groups with Different Models

```bash
# Group 1: Using nomic-embed-text (768 dimensions)
curl -X POST http://localhost:4002/groups \
  -H "Content-Type: application/json" \
  -d '{
    "name": "papers_nomic",
    "embed_model": "nomic-embed-text"
  }'

# Group 2: Using text-embedding-ada-002 (1536 dimensions)
curl -X POST http://localhost:4002/groups \
  -H "Content-Type: application/json" \
  -d '{
    "name": "papers_ada",
    "embed_model": "text-embedding-ada-002"
  }'
```

### Migration Strategy

If you need to change embedding models:

**Option 1: Create New Group** (Recommended)
```bash
# Create new group with new model
curl -X POST http://localhost:4002/groups \
  -d '{"name": "papers_v2", "embed_model": "new-model"}'

# Re-inject all documents into new group
curl -X POST http://localhost:4002/inject/papers_v2/batch \
  -d '{"documents": [...]}'

# Update application to use new group
# Delete old group when ready
curl -X DELETE http://localhost:4002/groups/papers_v1
```

**Option 2: In-Place Re-embedding** (Not Yet Implemented)
```bash
# Future endpoint for re-embedding existing group
curl -X POST http://localhost:4002/groups/papers_v1/reembed \
  -d '{"new_embed_model": "new-model"}'
```

For detailed information, see [EMBEDDING_MODELS.md](../rag_mcp/EMBEDDING_MODELS.md)

## Advanced Usage

### Custom Chunking Strategy

```bash
curl -X POST http://localhost:4002/inject/research_papers \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Long Research Paper",
    "content": "...",
    "chunk_size": 2000,      // Larger chunks for dense content
    "chunk_overlap": 400     // More overlap for context preservation
  }'
```

### Rich Metadata

```bash
curl -X POST http://localhost:4002/inject/research_papers \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Quantum Computing Review",
    "content": "...",
    "metadata": {
      "author": "Dr. Jane Doe",
      "institution": "MIT",
      "publication_year": 2024,
      "doi": "10.1234/example",
      "tags": ["quantum", "computing", "review"],
      "citation_count": 42,
      "language": "en"
    }
  }'
```

Metadata is:
- Stored in `rag_documents.metadata` (JSONB)
- Included in each chunk's metadata
- Returned in search results
- Queryable via RAG MCP service

### Batch Injection from URL Crawler

```python
import requests

# Fetch crawled URLs from backend
urls_response = requests.get("http://localhost:8000/urls?status=completed")
urls = urls_response.json()

# Prepare batch
documents = []
for url_data in urls:
    documents.append({
        "title": url_data["title"],
        "content": url_data["content"],
        "url_id": url_data["id"],
        "metadata": {
            "url": url_data["url"],
            "crawled_at": url_data["created_at"],
            "content_type": url_data.get("content_type")
        }
    })

# Batch inject
response = requests.post(
    "http://localhost:4002/inject/research_papers/batch",
    json={"documents": documents}
)
print(f"Injected {response.json()['successful']} documents")
```

## Integration with RAG Workflow

### Complete RAG Pipeline

```bash
# 1. Create RAG group
curl -X POST http://localhost:4002/groups \
  -d '{"name": "physics_kb", "embed_model": "nomic-embed-text"}'

# 2. Inject documents
curl -X POST http://localhost:4002/inject/physics_kb/batch \
  -d '{"documents": [...]}'

# 3. Search via RAG MCP (port 4001)
curl -X POST http://localhost:4001/rag_search \
  -d '{"query": "quantum mechanics", "rag_group": "physics_kb"}'

# 4. Use in LLM prompts
# Retrieved context automatically injected into prompts
```

### Multi-tenant Setup

```bash
# User 1's group
curl -X POST http://localhost:4002/groups \
  -d '{
    "name": "user1_docs",
    "scope": "user",
    "owner": "user1@example.com",
    "embed_model": "nomic-embed-text"
  }'

# User 2's group
curl -X POST http://localhost:4002/groups \
  -d '{
    "name": "user2_docs",
    "scope": "user",
    "owner": "user2@example.com",
    "embed_model": "nomic-embed-text"
  }'

# List user's groups
curl "http://localhost:4002/groups?scope=user&owner=user1@example.com"
```

## Monitoring & Management

### Check Service Status

```bash
# Inside backend container
docker exec -it agentks-backend-1 supervisorctl status

# Expected output:
# app                              RUNNING   pid 15, uptime 0:10:00
# rag_injector                     RUNNING   pid 18, uptime 0:10:00
# rag_mcp                          RUNNING   pid 17, uptime 0:10:00
# url_watcher                      RUNNING   pid 16, uptime 0:10:00
# web                              RUNNING   pid 14, uptime 0:10:00
```

### View Logs

```bash
# All services
docker logs agentks-backend-1

# Filter for rag_injector
docker logs agentks-backend-1 2>&1 | grep rag_injector
```

### Restart Service

```bash
docker exec -it agentks-backend-1 supervisorctl restart rag_injector
```

## Database Schema

### `rag_groups` Table

```sql
CREATE TABLE rag_groups (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    scope VARCHAR(50) DEFAULT 'global',
    owner VARCHAR(255),
    description TEXT,
    embed_model VARCHAR(100) NOT NULL,  -- Critical: tracks embedding model
    doc_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    UNIQUE(name, scope)
);
```

### `rag_documents` Table

```sql
CREATE TABLE rag_documents (
    id UUID PRIMARY KEY,
    rag_group_id UUID REFERENCES rag_groups(id) ON DELETE CASCADE,
    url_id UUID,
    title VARCHAR(500),
    content TEXT,
    content_hash VARCHAR(64),  -- SHA256 for deduplication
    metadata JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### PGVector Collection

Embeddings stored in `document_embeddings` collection with metadata:
```json
{
  "rag_group": "physics_kb",
  "rag_group_id": "550e8400-...",
  "document_id": "doc-123",
  "title": "Quantum Paper",
  "chunk_index": 0,
  "total_chunks": 5,
  "author": "Dr. Smith",
  ...
}
```

## Environment Variables

```bash
DATABASE_URL=postgresql+psycopg://user:pass@postgres:5432/dbname
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_EMBED_MODEL=nomic-embed-text  # Default model
COLLECTION_DOCS=document_embeddings  # PGVector collection name
```

## Error Handling

### Common Errors

**409 Conflict: Group Already Exists**
```json
{
  "detail": "RAG group 'papers' already exists in scope 'global'"
}
```

**404 Not Found: Group Doesn't Exist**
```json
{
  "detail": "RAG group 'unknown' not found in scope 'global'"
}
```

**409 Conflict: Duplicate Document**
```json
{
  "detail": "Document with same content already exists in this group"
}
```

### Debugging Tips

1. **Check service health**: `GET /health`
2. **Verify group exists**: `GET /groups/{name}`
3. **Check logs**: `docker logs agentks-backend-1 | grep rag_injector`
4. **Test embedding**: Inject small test document
5. **Verify chunks**: Check `chunks_created` in response

## Performance Considerations

### Chunking Strategy

- **Smaller chunks (500-800)**: Better precision, more chunks, slower
- **Larger chunks (1500-2000)**: More context, fewer chunks, faster
- **Overlap (100-300)**: Preserves context across boundaries

### Batch Size

- **Small batches (10-20)**: More responsive, easier error handling
- **Large batches (100+)**: More efficient, but all-or-nothing on errors

### Embedding Performance

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| nomic-embed-text | 768 | Fast | Good |
| text-embedding-ada-002 | 1536 | Medium | Excellent |
| bge-large | 1024 | Medium | Very Good |

## Related Services

- **RAG MCP Service** (port 4001): Query and retrieve from RAG
- **Admin API** (port 4000): Admin UI and management
- **Web API** (port 8000): Main application API
- **Basic Tools MCP** (port 5000): General utility tools

## Troubleshooting

### Service Won't Start

```bash
# Check supervisord status
docker exec -it agentks-backend-1 supervisorctl status

# View startup logs
docker logs agentks-backend-1 | tail -50

# Check if port is exposed
docker port agentks-backend-1
```

### Database Connection Issues

```bash
# Test database connection
docker exec -it agentks-backend-1 python -c \
  "import psycopg; conn = psycopg.connect('$DATABASE_URL'); print('OK')"
```

### Embedding Errors

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Test embedding model
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "test"}'
```

## Future Enhancements

- [ ] In-place re-embedding for model migration
- [ ] Async batch processing with job queue
- [ ] Document versioning and history
- [ ] Automatic embedding cleanup on document delete
- [ ] Webhook notifications for injection status
- [ ] Support for file uploads (PDF, DOCX, etc.)
- [ ] Document preprocessing pipeline
- [ ] Custom embedding model endpoints

## Contributing

When adding features to this service:

1. Maintain embedding model consistency
2. Update API documentation
3. Add integration tests
4. Update README with new endpoints
5. Follow FastAPI best practices

## License

Part of the AgentKS project.
