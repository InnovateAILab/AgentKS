"""
LangGraph Tools Skill

A dynamic tool discovery and invocation system using LangGraph.
This skill can:
1. Discover relevant tools based on user query using semantic search
2. Bind discovered tools to the LLM dynamically
3. Invoke tools through LangGraph's agent workflow
4. Handle multi-step tool usage with state management
"""

import os
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from tools import discover_tools, discover_tools_hybrid, bind_discovered_tools_to_llm


# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
TOOL_DISCOVERY_K = int(os.getenv("TOOL_DISCOVERY_K", "5"))
TOOL_DISCOVERY_MIN_SCORE = float(os.getenv("TOOL_DISCOVERY_MIN_SCORE", "0.3"))


class ToolsSkillState(TypedDict):
    """State for the tools skill workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    role: str
    query: str
    discovered_tools: List[Dict[str, Any]]
    bound_tools: List[BaseTool]
    tool_executor: Optional[ToolExecutor]


def create_tools_skill_graph(
    llm: Optional[ChatOllama] = None,
    discovery_k: int = TOOL_DISCOVERY_K,
    min_score: float = TOOL_DISCOVERY_MIN_SCORE,
    use_hybrid_search: bool = True,
):
    """
    Create a LangGraph workflow for dynamic tool discovery and invocation.
    
    Args:
        llm: Language model (defaults to ChatOllama with configured model)
        discovery_k: Number of tools to discover (default: 5)
        min_score: Minimum similarity score for tool discovery (default: 0.3)
        use_hybrid_search: Use hybrid (semantic + keyword) search (default: True)
    
    Returns:
        Compiled LangGraph workflow
    """
    
    if llm is None:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    
    # Node 1: Discover tools based on user query
    def discover_tools_node(state: ToolsSkillState) -> Dict[str, Any]:
        """Discover relevant tools using semantic search"""
        query = state["query"]
        user_id = state.get("user_id", "")
        role = state.get("role", "user")
        
        # Determine user scope
        user_scope = [f"user:{user_id}", "global"]
        
        # Discover tools using hybrid or semantic search
        if use_hybrid_search:
            discovered = discover_tools_hybrid(
                query=query,
                user_scope=user_scope,
                top_k=discovery_k,
                enabled_only=True,
                min_score=min_score,
                semantic_weight=0.7,
                keyword_weight=0.3,
            )
        else:
            discovered = discover_tools(
                query=query,
                user_scope=user_scope,
                top_k=discovery_k,
                enabled_only=True,
                min_score=min_score,
            )
        
        print(f"🔍 Discovered {len(discovered)} tools for query: '{query}'")
        for tool in discovered:
            print(f"  - {tool['name']} (score: {tool.get('similarity_score', 'N/A')})")
        
        return {
            "discovered_tools": discovered,
        }
    
    # Node 2: Bind tools to LLM
    def bind_tools_node(state: ToolsSkillState) -> Dict[str, Any]:
        """Bind discovered tools to the LLM"""
        discovered = state["discovered_tools"]
        
        if not discovered:
            print("⚠️ No tools discovered, proceeding without tools")
            return {
                "bound_tools": [],
                "tool_executor": None,
            }
        
        # Bind tools to LLM using tool_discovery module
        llm_with_tools = bind_discovered_tools_to_llm(llm, discovered)
        
        # Extract bound tools for the executor
        bound_tools = []
        if hasattr(llm_with_tools, 'bound') and hasattr(llm_with_tools.bound, 'tools'):
            bound_tools = llm_with_tools.bound.tools
        
        # Create tool executor
        tool_executor = ToolExecutor(bound_tools) if bound_tools else None
        
        print(f"🔧 Bound {len(bound_tools)} tools to LLM")
        
        return {
            "bound_tools": bound_tools,
            "tool_executor": tool_executor,
        }
    
    # Node 3: Agent - call LLM with tools
    def agent_node(state: ToolsSkillState) -> Dict[str, Any]:
        """Call LLM to decide on tool usage"""
        messages = state["messages"]
        discovered = state["discovered_tools"]
        
        # Create system message with tool instructions
        tool_names = [t["name"] for t in discovered] if discovered else []
        system_content = (
            "You are a helpful AI assistant with access to tools.\n"
            f"Available tools: {', '.join(tool_names) if tool_names else 'none'}\n"
            "Use tools when appropriate to answer the user's question.\n"
            "Provide clear, concise answers and cite sources when using tools."
        )
        
        # Bind tools to LLM
        if discovered:
            llm_with_tools = bind_discovered_tools_to_llm(llm, discovered)
        else:
            llm_with_tools = llm
        
        # Invoke LLM
        system_msg = SystemMessage(content=system_content)
        response = llm_with_tools.invoke([system_msg] + list(messages))
        
        return {
            "messages": [response],
        }
    
    # Node 4: Execute tools
    async def tool_node(state: ToolsSkillState) -> Dict[str, Any]:
        """Execute tool calls from the LLM"""
        messages = state["messages"]
        tool_executor = state.get("tool_executor")
        
        if not tool_executor:
            return {"messages": []}
        
        # Get the last AI message with tool calls
        last_message = messages[-1]
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"messages": []}
        
        # Execute tool calls
        tool_messages = []
        for tool_call in last_message.tool_calls:
            try:
                print(f"🛠️ Executing tool: {tool_call['name']} with args: {tool_call['args']}")
                
                # Create tool invocation
                invocation = ToolInvocation(
                    tool=tool_call['name'],
                    tool_input=tool_call['args'],
                )
                
                # Execute tool
                result = await tool_executor.ainvoke(invocation)
                
                # Create tool message
                tool_msg = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id'],
                    name=tool_call['name'],
                )
                tool_messages.append(tool_msg)
                
                print(f"✅ Tool {tool_call['name']} executed successfully")
                
            except Exception as e:
                print(f"❌ Error executing tool {tool_call['name']}: {e}")
                error_msg = ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_call['id'],
                    name=tool_call['name'],
                )
                tool_messages.append(error_msg)
        
        return {"messages": tool_messages}
    
    # Conditional edge: should continue or end?
    def should_continue(state: ToolsSkillState) -> Literal["continue", "end"]:
        """Determine if we should continue with tool calls or end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, continue to tool node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        # Otherwise, we're done
        return "end"
    
    # Build the graph
    workflow = StateGraph(ToolsSkillState)
    
    # Add nodes
    workflow.add_node("discover", discover_tools_node)
    workflow.add_node("bind", bind_tools_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "bind")
    workflow.add_edge("bind", "agent")
    
    # Conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        }
    )
    
    # After tools, go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app


async def run_tools_skill_async(
    query: str,
    user_id: str = "",
    role: str = "user",
    llm: Optional[ChatOllama] = None,
    discovery_k: int = TOOL_DISCOVERY_K,
    min_score: float = TOOL_DISCOVERY_MIN_SCORE,
    use_hybrid_search: bool = True,
) -> str:
    """
    Run the tools skill workflow asynchronously.
    
    Args:
        query: User query
        user_id: User ID for scoped tool access
        role: User role (e.g., 'admin', 'user')
        llm: Language model instance
        discovery_k: Number of tools to discover
        min_score: Minimum similarity score for tools
        use_hybrid_search: Use hybrid search
    
    Returns:
        Assistant's response as a string
    """
    
    # Create the graph
    app = create_tools_skill_graph(
        llm=llm,
        discovery_k=discovery_k,
        min_score=min_score,
        use_hybrid_search=use_hybrid_search,
    )
    
    # Prepare initial state
    initial_state: ToolsSkillState = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "role": role,
        "query": query,
        "discovered_tools": [],
        "bound_tools": [],
        "tool_executor": None,
    }
    
    # Run the workflow
    result = await app.ainvoke(initial_state)
    
    # Extract the final response
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    
    return "No response generated"


def run_tools_skill(
    query: str,
    user_id: str = "",
    role: str = "user",
    llm: Optional[ChatOllama] = None,
    discovery_k: int = TOOL_DISCOVERY_K,
    min_score: float = TOOL_DISCOVERY_MIN_SCORE,
    use_hybrid_search: bool = True,
) -> str:
    """
    Run the tools skill workflow (synchronous wrapper).
    
    Args:
        query: User query
        user_id: User ID for scoped tool access
        role: User role (e.g., 'admin', 'user')
        llm: Language model instance
        discovery_k: Number of tools to discover
        min_score: Minimum similarity score for tools
        use_hybrid_search: Use hybrid search
    
    Returns:
        Assistant's response as a string
    """
    import asyncio
    
    return asyncio.run(run_tools_skill_async(
        query=query,
        user_id=user_id,
        role=role,
        llm=llm,
        discovery_k=discovery_k,
        min_score=min_score,
        use_hybrid_search=use_hybrid_search,
    ))


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_tools_skill():
        """Test the tools skill"""
        
        # Example 1: Search query
        print("\n" + "="*60)
        print("Test 1: Search for papers")
        print("="*60)
        result = await run_tools_skill_async(
            query="Find papers about quantum computing on arXiv",
            user_id="test_user",
            role="user",
        )
        print(f"\n📝 Result:\n{result}\n")
        
        # Example 2: Multiple tools
        print("\n" + "="*60)
        print("Test 2: Multiple tool discovery")
        print("="*60)
        result = await run_tools_skill_async(
            query="Search for information about CERN experiments and calculate 2+2",
            user_id="test_user",
            role="user",
        )
        print(f"\n📝 Result:\n{result}\n")
    
    asyncio.run(test_tools_skill())
