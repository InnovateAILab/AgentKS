# LangGraph RAG Skill

A Retrieval-Augmented Generation (RAG) skill using LangGraph that retrieves and generates responses using content from the RAG MCP service.

## Overview

The RAG Skill is a LangGraph-based workflow that:
1. **Retrieves** relevant documents from the RAG MCP service using vector similarity search
2. **Ranks** results by similarity score
3. **Generates** answers using the LLM with retrieved context
4. **Cites** sources with document references and metadata

## Architecture

```
User Query
    ↓
[Retrieve Node] ← Call RAG MCP service (rag_search tool)
    ↓
[Should Generate?]
    ├─ Yes (has docs) → [Generate Node] ← LLM with context
    │                        ↓
    └─ No (no docs) ────────→ [END]
```

### Workflow Nodes

1. **Retrieve Node**: Calls RAG MCP service to get relevant documents
   - Uses `rag_search` tool from MCP
   - Performs vector similarity search
   - Filters by RAG group (optional)
   - Applies similarity threshold
   - Builds context from retrieved documents

2. **Generate Node**: LLM generates answer with retrieved context
   - System prompt instructs to use only retrieved information
   - Formats context with document numbers
   - Generates citation-rich answer
   - Appends source references

## Key Features

### 1. Vector Similarity Search
- Semantic search using embedding models
- Configurable number of results (k)
- Similarity score threshold filtering
- RAG group-specific search

### 2. Context-Aware Generation
- LLM receives full document context
- Instructed to cite sources [1], [2], etc.
- Clear handling of insufficient information
- Transparent about limitations

### 3. Source Citations
- Automatic source tracking
- Document URLs and metadata
- RAG group information
- Similarity scores included

### 4. Error Handling
- Graceful MCP service failures
- Clear error messages to user
- Continues workflow when possible
- Detailed logging for debugging

## Usage

### Basic Usage

```python
from app.rag_skill import run_rag_skill

# Simple query
result = run_rag_skill(
    query="What is quantum computing?",
    k=5,
    score_threshold=0.3,
)
print(result)
```

### Async Usage

```python
from app.rag_skill import run_rag_skill_async
import asyncio

async def main():
    result = await run_rag_skill_async(
        query="Explain CERN's Large Hadron Collider",
        rag_group="physics_docs",  # Search in specific group
        k=10,
        score_threshold=0.5,
    )
    print(result)

asyncio.run(main())
```

### With Custom LLM

```python
from app.rag_skill import run_rag_skill
from langchain_ollama import ChatOllama

# Use a different model
custom_llm = ChatOllama(model="mixtral:8x7b", temperature=0.5)

result = run_rag_skill(
    query="What are the latest AI developments?",
    llm=custom_llm,
    k=8,
    score_threshold=0.4,
)
```

### Custom Graph Usage

```python
from app.rag_skill import create_rag_skill_graph

# Create custom graph with specific settings
graph = create_rag_skill_graph(
    llm=None,  # Use default
    rag_mcp_url="http://custom-rag-mcp:4002/mcp",
    default_k=10,
    default_score_threshold=0.5,
)

# Run with custom state
initial_state = {
    "messages": [],
    "query": "What is machine learning?",
    "rag_group": "ai_papers",
    "k": 15,
    "score_threshold": 0.6,
    "retrieved_docs": [],
    "context": "",
    "sources": [],
}

result = await graph.ainvoke(initial_state)
```

## Configuration

Environment variables:

```bash
# LLM model for RAG skill
OLLAMA_MODEL=llama3.2:3b

# RAG MCP service URL
RAG_MCP_URL=http://localhost:4002/mcp

# Default retrieval settings
DEFAULT_RAG_K=5                    # Number of documents to retrieve
DEFAULT_SCORE_THRESHOLD=0.3        # Minimum similarity score
```

## Integration with Main API

### Adding to FastAPI Endpoint

```python
from app.rag_skill import run_rag_skill_async

@app.post("/api/rag-skill/ask")
async def rag_skill_ask(
    query: str,
    rag_group: Optional[str] = None,
    k: int = 5,
    score_threshold: float = 0.3,
):
    """Ask a question using RAG skill"""
    result = await run_rag_skill_async(
        query=query,
        rag_group=rag_group,
        k=k,
        score_threshold=score_threshold,
    )
    return {"answer": result}
```

### Streaming Support

```python
from app.rag_skill import create_rag_skill_graph
from fastapi.responses import StreamingResponse

@app.post("/api/rag-skill/stream")
async def rag_skill_stream(
    query: str,
    rag_group: Optional[str] = None,
):
    """Stream responses from RAG skill"""
    graph = create_rag_skill_graph()
    
    async def generate():
        initial_state = {
            "messages": [],
            "query": query,
            "rag_group": rag_group,
            "k": 5,
            "score_threshold": 0.3,
            "retrieved_docs": [],
            "context": "",
            "sources": [],
        }
        
        async for event in graph.astream(initial_state):
            yield json.dumps(event) + "\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## State Schema

```python
class RAGSkillState(TypedDict):
    """State maintained throughout the workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Conversation history
    query: str                          # User question
    rag_group: Optional[str]            # RAG group to search within
    k: int                              # Number of documents to retrieve
    score_threshold: float              # Minimum similarity score
    retrieved_docs: List[Dict]          # Retrieved documents from MCP
    context: str                        # Formatted context for LLM
    sources: List[Dict[str, str]]       # Source citations
```

## RAG MCP Service Integration

### Available Tools

The RAG skill uses the following tools from RAG MCP:

1. **rag_search**: Vector similarity search
   ```json
   {
     "query": "quantum computing",
     "k": 5,
     "rag_group": "physics_docs",
     "score_threshold": 0.3
   }
   ```

2. **rag_query**: Database pattern matching (not used by default, but available)
   ```json
   {
     "rag_group": "ai_papers",
     "title_pattern": "%neural%",
     "limit": 10
   }
   ```

3. **rag_get_document**: Get specific document by ID
   ```json
   {
     "document_id": "uuid-here"
   }
   ```

### Response Format

RAG MCP returns results in this format:
```json
{
  "query": "quantum computing",
  "num_results": 5,
  "rag_group": "physics_docs",
  "embedding_model": "nomic-embed-text",
  "results": [
    {
      "content": "Document content...",
      "metadata": {
        "source_url": "https://example.com/doc",
        "rag_group": "physics_docs",
        "title": "Document Title"
      },
      "similarity_score": 0.85,
      "embedding_model": "nomic-embed-text"
    }
  ]
}
```

## Examples

### Example 1: General Question

```python
result = run_rag_skill(
    query="What is the Higgs boson?",
    k=5,
    score_threshold=0.4,
)
# Output: Detailed answer with citations like:
# "The Higgs boson is... [1] [2]
# 
# Sources:
# [1] https://cern.ch/higgs-discovery (group: physics_docs, score: 0.87)
# [2] https://arxiv.org/particle-physics (group: physics_docs, score: 0.65)"
```

### Example 2: Domain-Specific Search

```python
result = run_rag_skill(
    query="How do neural networks learn?",
    rag_group="ai_papers",  # Search only in AI papers
    k=10,
    score_threshold=0.6,    # Higher threshold for quality
)
```

### Example 3: Low Information Case

```python
result = run_rag_skill(
    query="What is the capital of Atlantis?",
    k=5,
    score_threshold=0.3,
)
# Output: "Based on the retrieved documents, I don't have information
# about the capital of Atlantis. No relevant documents were found."
```

## Debugging

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("app.rag_skill")

# The skill prints debug info:
# 🔍 Retrieving documents for query: 'quantum computing'
#    RAG group: physics_docs
#    k=5, score_threshold=0.3
# ✅ Retrieved 5 documents
# 🤖 Generating answer using LLM
# ✅ Generated answer (1234 chars)
```

## Performance Considerations

1. **Retrieval Speed**: Vector search is fast (~50-200ms) depending on database size
2. **Generation Latency**: LLM generation is the main bottleneck (~1-5s)
3. **Context Size**: Larger k increases LLM context, may slow generation
4. **Score Threshold**: Higher thresholds reduce false positives but may miss relevant docs

## Best Practices

### 1. Choose Appropriate k
- Small k (3-5): Fast, focused answers
- Medium k (5-10): Balanced coverage
- Large k (10-20): Comprehensive but slower

### 2. Set Score Thresholds
- Low (0.1-0.3): Broad search, may include irrelevant docs
- Medium (0.3-0.5): Good balance
- High (0.5-1.0): High precision, may miss relevant docs

### 3. Use RAG Groups
- Always specify rag_group when possible
- Ensures consistent embedding models
- Improves search accuracy
- Reduces search space

### 4. Handle No Results
- The skill gracefully handles empty results
- LLM generates appropriate "I don't know" responses
- Consider lowering score_threshold if no results

## Comparison with Direct RAG

| Feature | RAG Skill | Direct RAG Search |
|---------|-----------|-------------------|
| Retrieval | MCP service call | Direct DB/vector store |
| Generation | LLM with context | No generation |
| Citations | Automatic | Manual |
| Error Handling | Workflow-managed | Manual |
| Streaming | Supported | N/A |
| Modularity | High (LangGraph) | Low |
| Testability | Easy (mock MCP) | Harder |

## Troubleshooting

### No Documents Retrieved
- Check RAG MCP service is running (`RAG_MCP_URL`)
- Verify RAG group name exists
- Lower `score_threshold`
- Check if documents exist in that group

### Poor Answer Quality
- Increase k to get more context
- Raise score_threshold for higher quality docs
- Check embedding model consistency in RAG group
- Verify document content is relevant

### MCP Connection Errors
- Verify `RAG_MCP_URL` is correct
- Check network connectivity
- Ensure RAG MCP service is running
- Check logs for authentication issues

### LLM Not Citing Sources
- Check system prompt in generate_node
- Verify sources are being tracked in retrieve_node
- Ensure context formatting includes [1], [2], etc.

## Related Documentation

- [RAG MCP Service](../../rag/rag_mcp/RAG_MCP.md) - MCP service details
- [Tools Skill Guide](./TOOLS_SKILL_GUIDE.md) - Dynamic tool discovery
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - Graph framework

## Future Enhancements

- [ ] Multi-query retrieval (query expansion)
- [ ] Re-ranking with cross-encoder
- [ ] Hybrid search (semantic + keyword)
- [ ] Document chunking strategies
- [ ] Response streaming support
- [ ] Conversation history integration
- [ ] Multi-RAG group aggregation
- [ ] Answer confidence scoring
