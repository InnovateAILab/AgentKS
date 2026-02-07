# Embedding Model Management for RAG

## The Critical Issue: Embedding Model Compatibility

### Why Embedding Models Matter

Different embedding models are **NOT compatible** with each other. This is critical to understand:

1. **Different Vector Dimensions**
   - `nomic-embed-text`: 768 dimensions
   - `text-embedding-ada-002` (OpenAI): 1536 dimensions
   - `bge-large`: 1024 dimensions
   - `all-MiniLM-L6-v2`: 384 dimensions

2. **Different Semantic Spaces**
   - Even if two models have the same dimensions, their vector spaces have different semantic meanings
   - A query embedded with Model A cannot meaningfully search documents embedded with Model B
   - Results will be random/nonsensical

3. **Cannot Mix Models**
   - You MUST use the same model for both:
     - Embedding documents (ingestion)
     - Embedding queries (search)

## Our Solution: Per-Group Embedding Model Tracking

### Database Schema

The `rag_groups` table tracks which embedding model was used:

```sql
CREATE TABLE rag_groups (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    embed_model TEXT,  -- ← Stores embedding model name
    doc_count INTEGER DEFAULT 0,
    ...
);
```

### How It Works

1. **Document Ingestion**: When documents are added to a RAG group, the `embed_model` field is set
2. **Query Search**: When searching, the system uses the same model specified in the RAG group
3. **Model Isolation**: Each RAG group can use a different embedding model

### RAG MCP Implementation

```python
# When searching, we:
1. Query the rag_groups table for the embed_model
2. Create embeddings instance with that specific model
3. Use it to embed the query
4. Search only within that group's documents
```

## Best Practices

### ✅ DO

1. **Specify RAG Group in Searches**
   ```python
   rag_search(
       query="What is machine learning?",
       rag_group="ml-docs",  # ← ALWAYS specify when possible
       k=5
   )
   ```

2. **Use Consistent Models per Group**
   - Pick one embedding model when creating a RAG group
   - Stick with it for the lifetime of that group

3. **Document Your Model Choice**
   - Use the `description` field in `rag_groups` to note why you chose that model
   - Example: "Uses nomic-embed-text for fast local embeddings"

4. **Separate Groups for Different Models**
   ```
   ✓ "python-docs-nomic" (nomic-embed-text)
   ✓ "python-docs-ada" (text-embedding-ada-002)
   ```

### ❌ DON'T

1. **Don't Search Across All Groups**
   ```python
   # BAD: May mix documents from different embedding models
   rag_search(query="python", k=10)  # No rag_group specified
   ```

2. **Don't Change Models Mid-Stream**
   ```python
   # BAD: Documents embedded with model A, searching with model B
   # Original ingestion used nomic-embed-text
   # Later switched to text-embedding-ada-002
   # → Old documents won't be found properly
   ```

3. **Don't Assume Models Are Interchangeable**
   - They are not!
   - Always re-embed all documents when changing models

## Migration Strategy: Changing Embedding Models

If you need to upgrade or change embedding models:

### Option 1: Create New RAG Group (Recommended)

```python
# 1. Create new group with new model
POST /admin/rags/add
{
    "name": "ml-docs-v2",
    "embed_model": "bge-large",
    "description": "ML docs with upgraded embedding model"
}

# 2. Re-ingest all documents into new group
# (Copy URLs from old group to new group)

# 3. Update applications to use new group
rag_search(query="...", rag_group="ml-docs-v2")

# 4. Delete old group when migration complete
DELETE /admin/rags/d1
```

### Option 2: In-Place Re-embedding (Complex)

```bash
# 1. Note current embedding model
current_model=$(psql -c "SELECT embed_model FROM rag_groups WHERE name='ml-docs'")

# 2. Update embed_model in database
UPDATE rag_groups SET embed_model='bge-large' WHERE name='ml-docs';

# 3. Delete old embeddings from vector store
DELETE FROM langchain_pg_embedding WHERE cmetadata->>'rag_group' = 'ml-docs';

# 4. Re-run ingestion for all documents
python -m backend_app.daemons.rag_injector --rag-group ml-docs --force-reembed

# 5. Update doc_count
UPDATE rag_groups SET doc_count = (
    SELECT COUNT(*) FROM rag_documents WHERE rag_group_id = '...'
) WHERE name='ml-docs';
```

## Choosing an Embedding Model

### Factors to Consider

1. **Quality vs Speed**
   - Large models (768-1536 dim): Better semantic understanding, slower
   - Small models (384 dim): Faster, less nuanced

2. **Local vs Cloud**
   - Local (Ollama): Free, private, but requires GPU for speed
   - Cloud (OpenAI): Fast, scalable, but costs money and sends data externally

3. **Language Support**
   - Some models are English-only
   - Multilingual models exist but may have lower quality per language

4. **Domain Specificity**
   - General: `nomic-embed-text`, `text-embedding-ada-002`
   - Code: `code-search-ada-code-001`
   - Scientific: Fine-tuned models like `specter`

### Recommended Models

**For Most Use Cases (Local):**
```
nomic-embed-text (768 dim)
- Good quality
- Fast with GPU
- Works with Ollama
- Free
```

**For Production (Cloud):**
```
text-embedding-3-small (OpenAI, 1536 dim)
- Excellent quality
- Fast via API
- Costs ~$0.02 per 1M tokens
```

**For Large Scale (Local):**
```
all-MiniLM-L6-v2 (384 dim)
- Smaller vectors = less storage
- Fast even on CPU
- Good enough for many use cases
```

## Monitoring and Troubleshooting

### Check Which Model a Group Uses

```bash
curl http://localhost:4001/mcp/resources/rag://groups | jq '.groups[] | {name, embed_model}'
```

### Verify Consistency

```sql
-- Check if all documents in a group use the same model
SELECT 
    g.name,
    g.embed_model as group_model,
    COUNT(*) as doc_count
FROM rag_groups g
JOIN rag_documents d ON g.id = d.rag_group_id
GROUP BY g.name, g.embed_model;
```

### Warning Signs

1. **Poor Search Results**: May indicate model mismatch
2. **Empty Results**: Check if documents were embedded with group's model
3. **Inconsistent Scores**: Different models have different similarity ranges

### Debug Search Issues

```python
# Search returns embedding_model in response
response = rag_search(
    query="test",
    rag_group="my-docs",
    k=1
)
print(response['embedding_model'])  # Should match group's embed_model
```

## API Integration

### When Creating RAG Group

```python
POST /admin/rags/add
{
    "name": "product-docs",
    "embed_model": "nomic-embed-text",  # ← IMPORTANT: Set this!
    "description": "Product documentation",
    "scope": "global"
}
```

### When Ingesting Documents

The backend should:
1. Read `embed_model` from `rag_groups` table
2. Use that model to generate embeddings
3. Store embeddings in PGVector with metadata: `{"rag_group": "product-docs"}`

### When Searching

The RAG MCP service automatically:
1. Looks up `embed_model` from `rag_groups` table
2. Creates embeddings instance with that model
3. Embeds the query
4. Searches within that group

## Performance Considerations

### Storage Requirements

```
nomic-embed-text (768 dim):
- ~3 KB per document
- 1M docs = ~3 GB

text-embedding-ada-002 (1536 dim):
- ~6 KB per document  
- 1M docs = ~6 GB

all-MiniLM-L6-v2 (384 dim):
- ~1.5 KB per document
- 1M docs = ~1.5 GB
```

### Search Speed

With HNSW index:
- 768 dim: ~5-10ms per query
- 1536 dim: ~10-20ms per query
- 384 dim: ~2-5ms per query

(Times vary with corpus size and hardware)

## Example Workflow

### Complete RAG Setup with Model Tracking

```bash
# 1. Create RAG group with specific model
curl -X POST http://localhost:8000/admin/rags/add \
  -H "Content-Type: application/json" \
  -d '{
    "name": "api-reference",
    "embed_model": "nomic-embed-text",
    "description": "API documentation using Nomic embeddings"
  }'

# 2. Add URLs to group
curl -X POST http://localhost:8000/admin/url_rag_groups/add \
  -d "url_id=u1&rag_group_id=r1"

# 3. Documents are ingested with nomic-embed-text embeddings

# 4. Search uses the same model automatically
curl -X POST http://localhost:4001/sse \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "rag_search",
      "arguments": {
        "query": "How do I authenticate?",
        "rag_group": "api-reference",
        "k": 5
      }
    },
    "id": 1
  }'

# Response includes embedding_model field confirming which model was used
```

## Future Enhancements

1. **Model Version Tracking**: Track model version (e.g., `nomic-embed-text-v1.5`)
2. **Automatic Migration**: Tool to re-embed group with new model
3. **Multi-Model Search**: Ensemble methods combining multiple models
4. **Model Compatibility Check**: Warn when mixing incompatible models
5. **Performance Metrics**: Track search quality per model

## Summary

✅ **Always specify `rag_group` in searches**
✅ **Set `embed_model` when creating RAG groups**
✅ **Use consistent models within a group**
✅ **Create new groups when changing models**
❌ **Never mix documents from different embedding models**
❌ **Don't search across all groups without understanding model mixing**
