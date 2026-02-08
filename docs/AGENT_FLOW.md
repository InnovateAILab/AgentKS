# Agent Flow Architecture

## Overview

The AgentKS system uses LangGraph to orchestrate multiple AI skills for intelligent query handling. The agent can conditionally execute multiple skills (RAG, Tools, Calculator, Direct) and synthesize their results for comprehensive answers.

## Architecture Components

### Core Skills

1. **RAG Skill** (`rag_skill.py`)
   - Knowledge base search and retrieval
   - Document similarity search via RAG MCP service
   - Context extraction with source citations

2. **Tools Skill** (`tools_skill.py`)
   - Dynamic tool discovery using semantic search
   - External API integration (arXiv, CDS, INSPIRE-HEP, SearXNG)
   - Tool binding and execution via MCP services

3. **Agent Skill** (`agent_skill.py`)
   - Main orchestrator using LangGraph StateGraph
   - Query analysis and routing
   - Multi-skill execution coordination
   - Result synthesis

## Agent State

The agent maintains state throughout execution:

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]      # Conversation history
    user_id: str                     # User identifier
    role: str                        # User role (admin/user)
    query: str                       # Current user query
    route: str                       # Routing decision
    used_urls: Dict[str, int]        # URL citation tracking
    rag_context: Optional[str]       # Context from RAG skill
    rag_completed: bool              # RAG execution status
    tools_result: Optional[str]      # Result from tools skill
    tools_completed: bool            # Tools execution status
    needs_synthesis: bool            # Synthesis requirement flag
    final_answer: str                # Final response
```

## Execution Flow

### Entry Point: Analyze Node

The agent starts with query analysis to determine execution strategy:

```
┌─────────────┐
│   ANALYZE   │ ← Entry: Query analysis using LLM
│   (LLM)     │
└──────┬──────┘
       │
       └─→ Determines: calculator / rag_only / tools_only / rag_then_tools / direct
```

**Analysis Decisions:**
- `calculator`: Simple math expression detected
- `rag_only`: Query answerable from internal knowledge base
- `tools_only`: Requires external search/computation
- `rag_then_tools`: Needs both internal and external sources
- `direct`: Simple conversational response

### Flow Diagram

```
                    ┌─────────────┐
                    │   ANALYZE   │ ← Entry point
                    │   (LLM)     │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    calculator      rag_only/           tools_only        direct
         │          rag_then_tools          │              │
         ↓                 │                 ↓              ↓
    ┌────────┐            ↓            ┌────────┐    ┌─────────┐
    │  CALC  │       ┌─────────┐       │ TOOLS  │    │ DIRECT  │
    └────┬───┘       │   RAG   │       └───┬────┘    └────┬────┘
         │           └────┬────┘           │              │
         │                │                │              │
         │       ┌────────┴────────┐       │              │
         │       │                 │       │              │
         │   needs_synthesis?   tools_needed?            │
         │       │                 │       │              │
         │      NO                YES     NO             │
         │       │                 │       │              │
         │       ↓                 ↓       ↓              │
         │      END          ┌─────────┐  END            │
         │                   │  TOOLS  │                 │
         │                   └────┬────┘                 │
         │                        │                      │
         │               needs_synthesis?               │
         │                        │                      │
         │                       YES                     │
         │                        ↓                      │
         │                 ┌──────────────┐              │
         └─────────────────│  SYNTHESIZE  │──────────────┘
                           │     (LLM)    │
                           └──────┬───────┘
                                  │
                                  ↓
                                 END
```

## Execution Patterns

### Pattern 1: Calculator (Fast Path)

```
Query: "12 * (3 + 4)"
Flow:  analyze → calculator → END
```

- **Trigger**: Query matches math expression pattern
- **Processing**: Direct evaluation using `safe_eval()`
- **Output**: Numerical result
- **Duration**: Fastest path (no LLM needed)

### Pattern 2: RAG Only

```
Query: "What do you know about quantum computing?"
Flow:  analyze → rag → END
```

- **Trigger**: Query about stored/known information
- **Processing**: 
  1. RAG skill retrieves documents from knowledge base
  2. LLM generates answer using retrieved context
  3. Sources cited with [1], [2], etc.
- **Output**: Answer with citations from knowledge base

### Pattern 3: Tools Only

```
Query: "Search arXiv for recent papers on transformers"
Flow:  analyze → tools → END
```

- **Trigger**: Query requires external search/computation
- **Processing**:
  1. Tools skill discovers relevant tools (semantic search)
  2. Tools bound to LLM dynamically
  3. LLM invokes appropriate tool(s)
  4. Results formatted and returned
- **Output**: External search results with metadata

### Pattern 4: RAG + Tools (Multi-Skill with Synthesis)

```
Query: "Compare our internal research on LLMs with recent arXiv papers"
Flow:  analyze → rag → tools → synthesize → END
```

- **Trigger**: Query benefits from both internal and external sources
- **Processing**:
  1. **RAG Node**: Retrieves internal documents, stores context
  2. **Tools Node**: Searches external sources, receives RAG context
  3. **Synthesize Node**: LLM combines both results intelligently
- **Output**: Comprehensive answer citing both internal and external sources

**Synthesis Prompt:**
```
User Question: {query}

Information from Knowledge Base:
{rag_context}

Information from External Tools:
{tools_result}

Instructions:
1. Provide comprehensive answer combining both sources
2. Prioritize most relevant information
3. Cite sources appropriately
4. Be clear about what comes from KB vs external
5. Acknowledge conflicts if any
```

### Pattern 5: Direct Response

```
Query: "Hello, how are you?"
Flow:  analyze → direct → END
```

- **Trigger**: Greeting, thanks, meta-questions
- **Processing**: LLM generates conversational response
- **Output**: Natural language response

## Conditional Edges

### After RAG Node

```python
def should_continue_after_rag(state: AgentState) -> str:
    if not state["tools_completed"]:
        return "tools"  # Continue to tools
    elif state["needs_synthesis"]:
        return "synthesize"  # Synthesize if both complete
    else:
        return "end"  # Direct end if RAG only
```

### After Tools Node

```python
def should_continue_after_tools(state: AgentState) -> str:
    if state["needs_synthesis"]:
        return "synthesize"  # Synthesize RAG + Tools
    else:
        return "end"  # Direct end if Tools only
```

## State Tracking

### Execution Flags

| Flag | Purpose | Set By |
|------|---------|--------|
| `rag_completed` | RAG execution status | RAG node |
| `tools_completed` | Tools execution status | Tools node |
| `needs_synthesis` | Synthesis required | Analyze node |

### Context Passing

**RAG → Tools:**
```python
# RAG node stores context
state["rag_context"] = retrieved_documents

# Tools node enhances query with RAG context
if rag_context:
    enhanced_query = f"Context: {rag_context}\n\nQuery: {query}"
```

**RAG + Tools → Synthesis:**
```python
# Synthesis node combines both
synthesis_input = {
    "query": original_query,
    "rag_context": state["rag_context"],
    "tools_result": state["tools_result"]
}
```

## Query Analysis Examples

### RAG Only Detection

```
Query: "What did we discuss about machine learning?"
Analysis: Internal reference → rag_only
```

### Tools Only Detection

```
Query: "Find recent papers on neural networks"
Analysis: External search needed → tools_only
```

### Multi-Skill Detection

```
Query: "Compare our ML approach with current research"
Analysis: Needs both sources → rag_then_tools
```

## Performance Characteristics

| Pattern | LLM Calls | API Calls | Avg Duration |
|---------|-----------|-----------|--------------|
| Calculator | 0 | 0 | <100ms |
| RAG Only | 2 | 1 (RAG MCP) | 2-3s |
| Tools Only | 2-3 | 2-3 (Tool MCP) | 3-5s |
| RAG + Tools | 4 | 2+ (RAG + Tools) | 6-10s |
| Direct | 1 | 0 | 1-2s |

**LLM Calls:**
1. Analysis (all patterns except calculator)
2. Skill execution (RAG/Tools/Direct)
3. Synthesis (multi-skill only)

## Error Handling

### Graceful Degradation

```python
try:
    rag_result = await rag_graph.ainvoke(rag_input)
except Exception as e:
    # Store error in context
    state["rag_context"] = f"RAG error: {e}"
    # Continue to tools if planned
    if not state["tools_completed"]:
        continue_to_tools()
```

### Synthesis Fallback

```python
try:
    synthesized = llm.invoke(synthesis_prompt)
except Exception as e:
    # Fallback: concatenate results
    fallback = f"KB: {rag_context}\n\nTools: {tools_result}"
    return fallback
```

## Configuration

### Environment Variables

```bash
# Model configuration
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://ollama:11434

# RAG configuration
RAG_MCP_URL=http://localhost:4002/mcp
DEFAULT_RAG_K=5
DEFAULT_SCORE_THRESHOLD=0.3

# Tools configuration
TOOL_DISCOVERY_K=5
TOOL_DISCOVERY_MIN_SCORE=0.3
```

### Adjustable Parameters

**RAG Skill:**
- `k`: Number of documents to retrieve (default: 5)
- `score_threshold`: Minimum similarity score (default: 0.3)
- `rag_group`: Filter by document group (default: all)

**Tools Skill:**
- `discovery_k`: Number of tools to discover (default: 5)
- `min_score`: Minimum tool relevance score (default: 0.3)
- `use_hybrid_search`: Enable hybrid search (default: true)

## Usage Examples

### Basic Usage

```python
from .agent_skill import run_agent

# Simple query
answer = run_agent(
    user_id="user@example.com",
    role="user",
    messages=[
        {"role": "user", "content": "What is quantum computing?"}
    ],
    llm=llm
)
```

### With Conversation History

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about AI"},
    {"role": "assistant", "content": "AI is..."},
    {"role": "user", "content": "Can you search for recent papers?"}
]

answer = run_agent(user_id, role, messages, llm)
```

### Async Usage

```python
answer = await run_agent_async(user_id, role, messages, llm)
```

## Integration with Main Application

### FastAPI Endpoint

```python
@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest, ...):
    user_id = get_user_id(...)
    role = get_user_role(...)
    
    # Agent handles all routing and execution
    answer = run_agent(user_id, role, req.messages)
    
    return format_openai_response(answer)
```

### Streaming Support

The agent flow can be adapted for streaming by:
1. Yielding tokens during LLM generation
2. Streaming intermediate results (RAG → Tools)
3. Progressive synthesis output

## Monitoring and Debugging

### Execution Logs

Each node logs its execution:

```
🎯 AGENT: Analyzing query: 'search for quantum papers'
🎯 Decision: rag_then_tools
📚 RAG: Querying knowledge base
✅ RAG completed: 1234 chars of context
🔧 TOOLS: Using dynamic tools
🔍 Discovered 3 tools: arxiv_search, cds_search, web_search
✅ Tools result: Found 5 papers
🔄 SYNTHESIZE: Combining results
✅ Synthesized answer: 456 chars
```

### State Inspection

```python
# After execution, inspect final state
final_state = agent.invoke(initial_state)

print(f"Route taken: {final_state['route']}")
print(f"RAG used: {final_state['rag_completed']}")
print(f"Tools used: {final_state['tools_completed']}")
print(f"Synthesis needed: {final_state['needs_synthesis']}")
```

## Benefits of Multi-Skill Architecture

1. **Comprehensive Answers**: Combines multiple information sources
2. **Context Awareness**: RAG context enhances tool queries
3. **Flexibility**: Adapts execution based on query needs
4. **Efficiency**: Fast-path for calculators, skips unnecessary skills
5. **Scalability**: Easy to add new skills to the graph
6. **Transparency**: Clear execution flow with state tracking
7. **Robustness**: Graceful degradation on errors

## Future Enhancements

### Planned Features

- [ ] Parallel skill execution (RAG + Tools simultaneously)
- [ ] Multi-turn tool usage (iterative refinement)
- [ ] Confidence scoring for routing decisions
- [ ] User feedback loop for routing optimization
- [ ] Skill result caching
- [ ] Custom synthesis strategies per query type

### Extensibility

Adding a new skill:

```python
# 1. Define skill graph
new_skill_graph = create_new_skill_graph(llm)

# 2. Add to agent skill
def new_skill_node(state: AgentState) -> Dict[str, Any]:
    result = await new_skill_graph.ainvoke(...)
    return {"new_skill_result": result, "new_skill_completed": True}

# 3. Add node to workflow
workflow.add_node("new_skill", new_skill_node)

# 4. Update routing logic
def route_from_analyze(state: AgentState) -> str:
    if condition:
        return "new_skill"
    ...

# 5. Add conditional edges
workflow.add_conditional_edges("new_skill", should_continue, {...})
```

## Troubleshooting

### Common Issues

**Issue: Agent always routes to one skill**
- Check analysis prompt in `analyze_query()`
- Verify LLM responds with expected decision strings
- Review query patterns in decision rules

**Issue: Synthesis not triggered**
- Ensure `needs_synthesis=True` set in analyze node
- Verify both `rag_completed` and `tools_completed` are True
- Check conditional edge logic in `should_continue_after_*`

**Issue: Context not passed between skills**
- Verify state updates in node return values
- Check context extraction in skills
- Ensure state fields are defined in `AgentState`

## References

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **RAG Skill**: `backend_app/app/rag_skill.py`
- **Tools Skill**: `backend_app/app/tools_skill.py`
- **Agent Skill**: `backend_app/app/agent_skill.py`
- **Main Application**: `backend_app/app/main.py`
