# LangGraph Tools Skill

A dynamic tool discovery and invocation system using LangGraph that intelligently discovers and uses tools based on user queries.

## Overview

The Tools Skill is a LangGraph-based workflow that:
1. **Discovers** relevant tools using semantic search based on the user's query
2. **Binds** discovered tools to the LLM dynamically
3. **Invokes** tools through a structured agent workflow
4. **Handles** multi-step tool usage with proper state management

## Architecture

```
User Query
    ↓
[Discover Tools Node]
    ↓
[Bind Tools Node]
    ↓
[Agent Node] ←──────┐
    ↓              │
[Should Continue?] │
    ├─ Yes ─→ [Tool Node]
    └─ No ──→ [END]
```

### Workflow Nodes

1. **Discover Tools Node**: Uses semantic/hybrid search to find relevant tools
2. **Bind Tools Node**: Binds discovered tools to the LLM
3. **Agent Node**: LLM decides whether to use tools and how
4. **Tool Node**: Executes tool calls and returns results

## Key Features

### 1. Dynamic Tool Discovery
- Uses the `tools` module's semantic search capabilities
- Supports both semantic-only and hybrid (semantic + keyword) search
- Configurable similarity thresholds and result limits
- Scope-aware: respects user permissions and tool scopes

### 2. Automatic Tool Binding
- Leverages `bind_discovered_tools_to_llm` from the tools module
- Creates LangChain tools from MCP server tools
- Handles authentication and headers automatically
- Supports async tool execution

### 3. State Management
- Uses LangGraph's StateGraph for workflow orchestration
- Maintains conversation context across tool calls
- Tracks discovered tools and execution state
- Supports multi-turn conversations

### 4. Error Handling
- Graceful tool execution errors
- Fallback to LLM without tools if discovery fails
- Detailed logging for debugging

## Usage

### Basic Usage

```python
from app.tools_skill import run_tools_skill

# Simple synchronous call
result = run_tools_skill(
    query="Find papers about quantum computing",
    user_id="user123",
    role="user",
)
print(result)
```

### Async Usage

```python
from app.tools_skill import run_tools_skill_async
import asyncio

async def main():
    result = await run_tools_skill_async(
        query="Search arXiv for papers on machine learning",
        user_id="user123",
        role="user",
    )
    print(result)

asyncio.run(main())
```

### Advanced Configuration

```python
from app.tools_skill import run_tools_skill
from langchain_ollama import ChatOllama

# Custom LLM and discovery settings
custom_llm = ChatOllama(model="llama3.2:3b", temperature=0.7)

result = run_tools_skill(
    query="What are the latest CERN experiments?",
    user_id="user123",
    role="admin",
    llm=custom_llm,
    discovery_k=10,  # Discover up to 10 tools
    min_score=0.5,   # Higher similarity threshold
    use_hybrid_search=True,  # Use hybrid search
)
```

### Custom Graph Usage

```python
from app.tools_skill import create_tools_skill_graph
from langchain_core.messages import HumanMessage

# Create custom graph
graph = create_tools_skill_graph(
    llm=None,  # Use default
    discovery_k=5,
    min_score=0.3,
    use_hybrid_search=True,
)

# Run with custom initial state
initial_state = {
    "messages": [HumanMessage(content="Search for quantum papers")],
    "user_id": "user123",
    "role": "user",
    "query": "Search for quantum papers",
    "discovered_tools": [],
    "bound_tools": [],
    "tool_executor": None,
}

result = await graph.ainvoke(initial_state)
```

## Configuration

Environment variables:

```bash
# LLM model for tool skill
OLLAMA_MODEL=llama3.2:3b

# Tool discovery settings
TOOL_DISCOVERY_K=5              # Number of tools to discover
TOOL_DISCOVERY_MIN_SCORE=0.3   # Minimum similarity score
```

## Integration with Main API

### Adding to FastAPI Endpoint

```python
from app.tools_skill import run_tools_skill_async

@app.post("/api/tools-skill/query")
async def tools_skill_query(
    query: str,
    user_id: str = Header(None, alias="X-Authentik-Email"),
    role: str = Header("user", alias="X-Authentik-Groups"),
):
    """Query using dynamic tool discovery"""
    result = await run_tools_skill_async(
        query=query,
        user_id=user_id or "anonymous",
        role=role,
    )
    return {"result": result}
```

### Streaming Support

```python
from app.tools_skill import create_tools_skill_graph
from fastapi.responses import StreamingResponse

@app.post("/api/tools-skill/stream")
async def tools_skill_stream(
    query: str,
    user_id: str = Header(None, alias="X-Authentik-Email"),
):
    """Stream responses from tools skill"""
    graph = create_tools_skill_graph()
    
    async def generate():
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "user_id": user_id or "anonymous",
            "role": "user",
            "query": query,
            "discovered_tools": [],
            "bound_tools": [],
            "tool_executor": None,
        }
        
        async for event in graph.astream(initial_state):
            yield json.dumps(event) + "\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## State Schema

```python
class ToolsSkillState(TypedDict):
    """State maintained throughout the workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Conversation history
    user_id: str                    # User identifier
    role: str                       # User role (for permissions)
    query: str                      # Original user query
    discovered_tools: List[Dict]    # Tools discovered by semantic search
    bound_tools: List[BaseTool]     # LangChain tools bound to LLM
    tool_executor: Optional[ToolExecutor]  # Tool execution engine
```

## Examples

### Example 1: Search Tools

```python
result = run_tools_skill(
    query="Find papers about deep learning on arXiv",
    user_id="researcher@example.com",
    role="user",
)
# Output: Discovers arxiv_search tool, executes search, returns formatted results
```

### Example 2: Multi-Tool Query

```python
result = run_tools_skill(
    query="Search CERN documents about Higgs boson and calculate its mass",
    user_id="physicist@cern.ch",
    role="admin",
)
# Output: Discovers cds_search and calculator tools, uses both, returns answer
```

### Example 3: With Custom Filters

```python
result = run_tools_skill(
    query="Search physics papers",
    user_id="user@example.com",
    role="user",
    discovery_k=3,  # Only top 3 tools
    min_score=0.6,  # High relevance threshold
)
```

## Debugging

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("app.tools_skill")

# The skill prints debug info:
# 🔍 Discovered 3 tools for query: '...'
#   - arxiv_search (score: 0.87)
#   - cds_search (score: 0.65)
#   - calculator (score: 0.45)
# 🔧 Bound 3 tools to LLM
# 🛠️ Executing tool: arxiv_search with args: {...}
# ✅ Tool arxiv_search executed successfully
```

## Performance Considerations

1. **Tool Discovery**: Semantic search is fast (~50-100ms) but scales with tool count
2. **LLM Calls**: Main bottleneck, consider caching for repeated queries
3. **Tool Execution**: Async execution prevents blocking
4. **State Size**: Keep conversation history bounded for production

## Comparison with Static Tools

| Feature | Static Tools | Tools Skill |
|---------|-------------|-------------|
| Tool Selection | Hardcoded | Dynamic, query-based |
| Scalability | Limited | Handles 100s of tools |
| Relevance | All tools loaded | Only relevant tools |
| Memory | High (all tools) | Low (filtered tools) |
| Maintenance | Manual updates | Auto-discovers new tools |
| Context | Generic | Query-specific |

## Troubleshooting

### No Tools Discovered
- Check tool descriptions in database
- Lower `min_score` threshold
- Verify user scope and permissions
- Ensure tools are enabled

### Tool Execution Errors
- Check MCP server connectivity
- Verify authentication headers
- Review tool input schema
- Check tool_executor initialization

### LLM Not Using Tools
- Improve tool descriptions for clarity
- Adjust LLM temperature (lower = more deterministic)
- Check system message instructions
- Verify tool binding succeeded

## Related Documentation

- [Tools Module README](../tools/README.md) - Tool discovery and MCP integration
- [Tool Discovery Guide](../tools/TOOL_DISCOVERY_GUIDE.md) - Semantic search details
- [MCP Watcher Summary](../tools/MCP_WATCHER_SUMMARY.md) - Auto-discovery daemon
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - Graph framework

## Future Enhancements

- [ ] Tool result caching
- [ ] Multi-agent collaboration
- [ ] Tool chain optimization
- [ ] User feedback loop for tool selection
- [ ] Cost/latency tracking per tool
- [ ] A/B testing different discovery strategies
