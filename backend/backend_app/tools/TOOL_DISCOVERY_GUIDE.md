# Tool Discovery and Selection Strategy

## Overview

For LLM workflows that need to dynamically discover and bind tools, we use a **hybrid approach** combining:

1. **Semantic Search** (via embeddings) - for intent matching
2. **Metadata Filtering** - for constraints (enabled, scope, tags)
3. **Relevance Scoring** - hybrid ranking combining multiple signals

## Architecture

```
User Query
    ↓
Tool Discovery Engine
    ├─→ Semantic Search (embeddings in PGVector)
    ├─→ Metadata Filtering (enabled, scope, tags)
    └─→ Hybrid Scoring (semantic + keyword)
    ↓
Ranked Tool List
    ↓
Fetch Full Tool Metadata (from tools + mcps tables)
    ↓
Bind Tools to LLM
    ↓
LLM decides which tools to call
```

## Why This Approach?

### ✅ Semantic Search with Embeddings (Recommended)

**Pros:**
- Understands user intent ("find papers" → matches "arxiv_search", "inspire_search")
- Handles synonyms and variations
- Works across languages
- Scales well with large tool libraries (1000+ tools)

**Cons:**
- Requires vector database (already have PGVector)
- Needs periodic re-indexing
- Embedding quality depends on model

**When to use:**
- Large tool libraries (50+ tools)
- Natural language queries
- Need fuzzy matching

### Alternative: Keyword/Tag Search

**Pros:**
- Simple and fast
- No ML dependencies
- Exact matching

**Cons:**
- Brittle (typos, synonyms fail)
- Doesn't understand intent
- Requires perfect tagging

**When to use:**
- Small tool sets (< 20 tools)
- Structured queries with known keywords
- As supplement to semantic search

## Implementation Details

### 1. Tool Embedding Structure

Each tool is embedded with:
```python
{
    "page_content": "Tool description (natural language)",
    "metadata": {
        "tool_id": "uuid",
        "name": "tool_name",
        "enabled": true,
        "scope": "global|private",
        "tags": ["tag1", "tag2"]
    }
}
```

### 2. Discovery Flow

```python
from tool_discovery import discover_tools, bind_discovered_tools_to_llm

# Step 1: Discover relevant tools
tools = discover_tools(
    query="I need to search for recent physics papers",
    user_scope="user@example.com",  # For private tools
    top_k=5,                         # Max tools to return
    enabled_only=True,               # Filter
    min_score=0.3                    # Similarity threshold
)

# Step 2: Bind to LLM
llm_with_tools = bind_discovered_tools_to_llm(llm, tools)

# Step 3: LLM decides which tools to call
result = llm_with_tools.invoke("Find papers about quantum computing from 2024")
```

### 3. Scoring Mechanism

#### Semantic Score
```
similarity = 1 - cosine_distance(query_embedding, tool_embedding)
```

#### Keyword Score
```
keyword_score = (name_matches × 2 + desc_matches) / (query_words × 3)
```

#### Combined Score
```
final_score = (0.7 × semantic_score) + (0.3 × keyword_score)
```

Weights configurable via `semantic_weight` and `keyword_weight` parameters.

## Database Schema

### Existing `tools` Table
```sql
CREATE TABLE tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT,
    mcp_id TEXT REFERENCES mcps(id),
    metadata JSONB DEFAULT '{}'::jsonb,  -- Stores description, enabled, scope
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);
```

### Vector Store Collection
```
Collection: tool_embeddings
- Stores embeddings of tool descriptions
- Links back to tools.id via metadata.tool_id
```

## Usage Examples

### Example 1: Basic Discovery
```python
# Discover tools for a specific task
tools = discover_tools(
    query="I need to fetch web pages and extract content",
    top_k=3
)

for tool in tools:
    print(f"{tool['name']}: {tool['description']}")
    print(f"Relevance: {tool['similarity_score']:.2%}\n")
```

**Output:**
```
fetch_url: Fetch content from a URL and return HTML or text
Relevance: 87%

web_scraper: Extract structured data from web pages
Relevance: 72%

html_parser: Parse HTML content and extract elements
Relevance: 65%
```

### Example 2: Hybrid Search
```python
# Combine semantic and keyword matching
tools = discover_tools_hybrid(
    query="calculate statistical significance",
    semantic_weight=0.7,  # Prioritize semantic understanding
    keyword_weight=0.3,   # Add keyword boost
    top_k=5
)
```

### Example 3: Tag-based Filtering
```python
# Search by tags
tools = search_tools_by_tags(
    tags=["search", "academic"],
    match_all=False  # Match ANY tag
)
```

### Example 4: Complete Workflow
```python
from langchain_ollama import ChatOllama
from tool_discovery import discover_tools, bind_discovered_tools_to_llm

# Initialize LLM
llm = ChatOllama(model="llama2", base_url="http://ollama:11434")

# Discover relevant tools based on user's task
user_query = "I want to search for papers about machine learning and summarize them"

tools = discover_tools(
    query=user_query,
    user_scope="researcher@university.edu",
    top_k=5,
    tags=["search", "summary"]  # Optional filter
)

print(f"Discovered {len(tools)} relevant tools:")
for tool in tools:
    print(f"  - {tool['name']} (relevance: {tool['similarity_score']:.0%})")

# Bind tools to LLM
llm_with_tools = bind_discovered_tools_to_llm(llm, tools)

# Let LLM decide which tools to use
response = llm_with_tools.invoke(user_query)
```

## Configuration

### Environment Variables
```bash
# Tool embedding configuration
OLLAMA_EMBED_MODEL=nomic-embed-text     # Embedding model
COLLECTION_TOOLS=tool_embeddings        # Vector store collection
TOOL_SELECT_TOPK=6                      # Default top-k for discovery

# Scoring weights (can be overridden per call)
TOOL_SEMANTIC_WEIGHT=0.7
TOOL_KEYWORD_WEIGHT=0.3
```

### Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 6 | Maximum tools to return |
| `min_score` | 0.3 | Minimum similarity threshold (0-1) |
| `semantic_weight` | 0.7 | Weight for semantic similarity |
| `keyword_weight` | 0.3 | Weight for keyword matching |
| `enabled_only` | True | Filter to enabled tools only |

## Maintenance Operations

### Re-index All Tools
```python
from tool_discovery import reindex_all_tools

# After bulk updates or schema changes
reindex_all_tools()
```

### Index New Tool
```python
from app.main import index_tool_desc

index_tool_desc(
    tool_id="tool-123",
    name="arxiv_search",
    description="Search arXiv for academic papers by keywords or authors",
    enabled=True,
    scope="global"
)
```

### Update Tool Description
```python
# Update database
db_exec("""
    UPDATE tools 
    SET metadata = jsonb_set(metadata, '{description}', %s::jsonb)
    WHERE id = %s
""", ('"New description"', tool_id))

# Re-index
index_tool_desc(tool_id, name, new_description, enabled, scope)
```

## Performance Considerations

### Vector Search Performance
- **Small library (< 100 tools)**: < 10ms
- **Medium library (100-1000 tools)**: 10-50ms
- **Large library (1000+ tools)**: 50-200ms

### Optimization Tips
1. **Use appropriate top_k**: Don't fetch more tools than needed
2. **Enable indexing**: Ensure HNSW index on vector column
3. **Cache frequent queries**: Cache discovery results for common queries
4. **Batch indexing**: Index multiple tools in single operation
5. **Filter early**: Use metadata filters before semantic search

### Memory Usage
- **Embeddings**: ~3KB per tool (768-dim nomic-embed-text)
- **1000 tools**: ~3MB of embeddings
- **10,000 tools**: ~30MB of embeddings

## Best Practices

### 1. Tool Descriptions
Write clear, detailed descriptions:
```python
# ❌ Bad
"Search tool"

# ✅ Good
"Search arXiv for academic papers using keywords, author names, or categories. Returns paper titles, abstracts, authors, and publication dates."
```

### 2. Tagging Strategy
Use hierarchical tags:
```python
tags = [
    "search",           # Category
    "academic",         # Domain
    "arxiv",           # Source
    "physics"          # Subject
]
```

### 3. Scope Management
- `global`: Available to all users
- `private`: User-specific tools (use user email/ID as scope value)
- `team`: Team-specific tools (use team ID as scope)

### 4. Enable/Disable Tools
Disable tools without deleting:
```python
db_exec("""
    UPDATE tools 
    SET metadata = jsonb_set(metadata, '{enabled}', 'false'::jsonb)
    WHERE id = %s
""", (tool_id,))
```

### 5. Monitor Tool Usage
Track which tools are actually being used:
```python
# Log tool calls
db_exec("""
    INSERT INTO tool_runs (id, tool_id, input, status, started_at)
    VALUES (%s, %s, %s, 'running', now())
""", (run_id, tool_id, json.dumps(input)))

# Analyze usage
db_exec("""
    SELECT tool_id, COUNT(*) as usage_count
    FROM tool_runs
    WHERE started_at > now() - interval '7 days'
    GROUP BY tool_id
    ORDER BY usage_count DESC
    LIMIT 10
""")
```

## Comparison with Alternatives

### vs. Rule-based Selection
| Aspect | Semantic Search | Rule-based |
|--------|----------------|------------|
| Flexibility | High | Low |
| Maintenance | Low | High |
| Accuracy | Good | Perfect (if rules match) |
| Scalability | Excellent | Poor |

### vs. LLM-based Selection
| Aspect | Pre-Discovery | LLM-only |
|--------|---------------|----------|
| Speed | Fast (< 50ms) | Slow (1-5s) |
| Cost | One-time indexing | Per-query LLM call |
| Reliability | Deterministic | Variable |
| Explainability | Score-based | Black box |

### Recommended: Hybrid Approach
Use semantic discovery to narrow down tools, then let LLM choose from the relevant subset. This combines:
- **Speed** of vector search
- **Intelligence** of LLM decision-making
- **Cost efficiency** (fewer tools in context = cheaper LLM calls)

## Troubleshooting

### Issue: Poor tool discovery accuracy
**Solutions:**
- Improve tool descriptions (more detailed, use keywords)
- Adjust `min_score` threshold
- Use hybrid search instead of pure semantic
- Add more tags for filtering

### Issue: Slow discovery
**Solutions:**
- Reduce `top_k` value
- Add metadata filters (enabled, scope)
- Enable vector index (HNSW)
- Cache common queries

### Issue: Tools not being selected
**Solutions:**
- Check tool is enabled: `metadata->>'enabled' = 'true'`
- Verify scope matches user
- Lower `min_score` threshold
- Check embedding quality (re-index if needed)

## Future Enhancements

1. **Contextual Discovery**: Use conversation history for better tool selection
2. **Learning**: Track which tools are actually useful and boost their scores
3. **Multi-modal**: Support image/code-based tool discovery
4. **Federated Search**: Search across multiple MCP servers
5. **Tool Composition**: Suggest tool chains (pipe outputs)

## References

- [LangChain Tool Binding](https://python.langchain.com/docs/how_to/tools_bind/)
- [PGVector Documentation](https://github.com/pgvector/pgvector)
- [nomic-embed-text Model](https://ollama.com/library/nomic-embed-text)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
