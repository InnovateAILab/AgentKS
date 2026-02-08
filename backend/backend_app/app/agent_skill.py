"""
LangGraph Agent Skill

A unified agent orchestrator using LangGraph that coordinates RAG and Tools skills
to answer user questions intelligently.

This agent:
1. Analyzes user query to determine if RAG search or tool usage is needed
2. Routes to RAG skill for knowledge base questions
3. Routes to Tools skill for computational/search tasks
4. Handles calculator queries directly for efficiency
5. Combines results and generates final response

Architecture:
- Uses StateGraph for workflow management
- Integrates rag_skill for document retrieval
- Integrates tools_skill for dynamic tool execution
- Provides streaming and non-streaming responses
"""

import os
import ast
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .rag_skill import create_rag_skill_graph
from .tools_skill import create_tools_skill_graph


# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


class AgentState(TypedDict):
    """State for the unified agent workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    role: str
    query: str
    route: Optional[Literal["calculator", "rag", "tools", "direct"]]
    used_urls: Dict[str, int]  # Track URLs for citation numbering
    rag_context: Optional[str]  # Context from RAG skill
    rag_completed: bool  # Whether RAG has been executed
    tools_result: Optional[str]  # Result from tools skill
    tools_completed: bool  # Whether tools have been executed
    needs_synthesis: bool  # Whether we need to synthesize results
    final_answer: str


# Calculator helper functions (for direct math evaluation)
_ALLOWED = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd}


def safe_eval(expr: str) -> float:
    """Safely evaluate a mathematical expression"""
    node = ast.parse(expr, mode="eval").body

    def _eval(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED:
            v = _eval(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED:
            a = _eval(n.left)
            b = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow):
                return a**b
            if isinstance(n.op, ast.Mod):
                return a % b
        raise ValueError("Unsupported expression")

    return float(_eval(node))


def looks_like_math(text: str) -> bool:
    """Check if text looks like a math expression"""
    allowed = set("0123456789+-*/().% ^\n\t ")
    t = text.strip()
    return t and all(ch in allowed for ch in t)


def create_agent_graph(
    llm: Optional[ChatOllama] = None,
):
    """
    Create a unified LangGraph agent that coordinates RAG and Tools skills.
    
    Args:
        llm: Language model (defaults to ChatOllama with configured model)
    
    Returns:
        Compiled LangGraph workflow
    """
    
    if llm is None:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2
        )
    
    # Create sub-graphs for RAG and Tools
    rag_graph = create_rag_skill_graph(llm)
    tools_graph = create_tools_skill_graph(llm)
    
    # Node 1: Analyze query and plan execution
    def analyze_query(state: AgentState) -> Dict[str, Any]:
        """Analyze query to determine which skills to use"""
        query = state["query"]
        
        print(f"\n{'='*60}")
        print(f"🎯 AGENT: Analyzing query: '{query}'")
        print(f"{'='*60}\n")
        
        # Quick check: is it a simple math expression?
        if looks_like_math(query):
            print("📊 Route: CALCULATOR (direct math)")
            return {
                "route": "calculator",
                "rag_completed": True,
                "tools_completed": True,
                "needs_synthesis": False,
            }
        
        # Analyze query to determine which skills are needed
        analysis_prompt = f"""Analyze the following user query and determine which skills are needed:

Query: "{query}"

Available Skills:
1. RAG - Knowledge base search (for stored documents, previous conversations, saved information)
2. TOOLS - External searches and computations (arXiv, CDS, INSPIRE-HEP, web search)
3. DIRECT - Simple conversational response (greetings, thanks, clarifications)

Decision Rules:
- Use ONLY RAG if: Query is about stored/saved information or previous knowledge
- Use ONLY TOOLS if: Query needs external/live data (scientific papers, web info)
- Use RAG+TOOLS if: Query benefits from both internal knowledge AND external search
- Use DIRECT if: Simple greeting, thanks, or meta question about capabilities

Respond with ONE of these exact phrases:
- "rag_only"
- "tools_only" 
- "rag_then_tools"
- "direct"
"""
        
        try:
            response = llm.invoke([HumanMessage(content=analysis_prompt)])
            decision = response.content.strip().lower()
            
            print(f"🎯 Decision: {decision}")
            
            # Map decision to execution plan
            if decision == "rag_only":
                return {
                    "route": "rag",
                    "rag_completed": False,
                    "tools_completed": True,  # Skip tools
                    "needs_synthesis": False,  # RAG will provide final answer
                }
            elif decision == "tools_only":
                return {
                    "route": "tools",
                    "rag_completed": True,  # Skip RAG
                    "tools_completed": False,
                    "needs_synthesis": False,  # Tools will provide final answer
                }
            elif decision == "rag_then_tools":
                return {
                    "route": "rag",
                    "rag_completed": False,
                    "tools_completed": False,
                    "needs_synthesis": True,  # Need to synthesize both results
                }
            elif decision == "direct":
                return {
                    "route": "direct",
                    "rag_completed": True,  # Skip RAG
                    "tools_completed": True,  # Skip tools
                    "needs_synthesis": False,
                }
            else:
                # Default: try both RAG and tools
                print(f"⚠️ Unknown decision '{decision}', defaulting to rag_then_tools")
                return {
                    "route": "rag",
                    "rag_completed": False,
                    "tools_completed": False,
                    "needs_synthesis": True,
                }
            
        except Exception as e:
            print(f"❌ Error in analysis: {e}, defaulting to rag_then_tools")
            return {
                "route": "rag",
                "rag_completed": False,
                "tools_completed": False,
                "needs_synthesis": True,
            }
    
    # Node 2: Handle calculator
    def calculator_node(state: AgentState) -> Dict[str, Any]:
        """Handle simple math calculations directly"""
        query = state["query"]
        
        print(f"🧮 CALCULATOR: Evaluating '{query}'")
        
        try:
            result = safe_eval(query)
            answer = str(result)
            print(f"✅ Result: {answer}")
            
            return {
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
            }
        except Exception as e:
            error_msg = f"Error calculating: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "messages": [AIMessage(content=error_msg)],
                "final_answer": error_msg,
            }
    
    # Node 3: Handle RAG queries
    async def rag_node(state: AgentState) -> Dict[str, Any]:
        """Route to RAG skill for knowledge base queries"""
        query = state["query"]
        user_id = state["user_id"]
        
        print(f"📚 RAG: Querying knowledge base for '{query}'")
        
        try:
            # Prepare RAG skill input
            rag_input = {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "rag_group": None,  # Search all groups
                "k": 5,
                "score_threshold": 0.3,
            }
            
            # Run RAG skill graph
            result = await rag_graph.ainvoke(rag_input)
            
            # Extract context from result
            context = result.get("context", "")
            sources = result.get("sources", [])
            
            # If we don't need synthesis, extract the answer too
            answer = ""
            if not state.get("needs_synthesis"):
                if result.get("messages"):
                    last_msg = result["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        answer = last_msg.content
                    elif hasattr(last_msg, "content"):
                        answer = last_msg.content
                    else:
                        answer = str(last_msg)
                
                if not answer:
                    answer = context or "No relevant information found in knowledge base."
            
            print(f"✅ RAG completed: {len(context)} chars of context")
            
            return {
                "rag_context": context,
                "rag_completed": True,
                "messages": [AIMessage(content=answer)] if answer else [],
                "final_answer": answer if not state.get("needs_synthesis") else "",
            }
            
        except Exception as e:
            error_msg = f"RAG error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "rag_context": error_msg,
                "rag_completed": True,
                "messages": [],
                "final_answer": error_msg if not state.get("needs_synthesis") else "",
            }
    
    # Node 4: Handle tool queries
    async def tools_node(state: AgentState) -> Dict[str, Any]:
        """Route to Tools skill for tool-based queries"""
        query = state["query"]
        user_id = state["user_id"]
        role = state["role"]
        messages = state["messages"]
        rag_context = state.get("rag_context", "")
        
        print(f"🔧 TOOLS: Using dynamic tools for '{query}'")
        
        # If we have RAG context, include it in the query
        enhanced_query = query
        if rag_context:
            enhanced_query = f"Context from knowledge base:\n{rag_context[:500]}...\n\nUser query: {query}"
            print(f"📝 Including RAG context in tools query")
        
        try:
            # Prepare Tools skill input
            tools_input = {
                "messages": list(messages) + [HumanMessage(content=enhanced_query)],
                "user_id": user_id,
                "role": role,
                "query": enhanced_query,
                "discovered_tools": [],
                "bound_tools": [],
                "tool_executor": None,
            }
            
            # Run Tools skill graph
            result = await tools_graph.ainvoke(tools_input)
            
            # Extract result
            tools_result = ""
            if result.get("messages"):
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        tools_result = msg.content
                        break
                    elif hasattr(msg, "content") and hasattr(msg, "type"):
                        if getattr(msg, "type", "") == "ai":
                            tools_result = msg.content
                            break
            
            if not tools_result:
                tools_result = "No response generated from tools."
            
            print(f"✅ Tools result: {tools_result[:100]}...")
            
            # If we don't need synthesis, this is the final answer
            if not state.get("needs_synthesis"):
                return {
                    "tools_result": tools_result,
                    "tools_completed": True,
                    "messages": [AIMessage(content=tools_result)],
                    "final_answer": tools_result,
                }
            else:
                # Store result for synthesis
                return {
                    "tools_result": tools_result,
                    "tools_completed": True,
                    "messages": [],
                }
            
        except Exception as e:
            error_msg = f"Tools error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "tools_result": error_msg,
                "tools_completed": True,
                "messages": [AIMessage(content=error_msg)] if not state.get("needs_synthesis") else [],
                "final_answer": error_msg if not state.get("needs_synthesis") else "",
            }
    
    # Node 5: Handle direct responses
    def direct_node(state: AgentState) -> Dict[str, Any]:
        """Handle direct responses without RAG or tools"""
        query = state["query"]
        messages = state["messages"]
        
        print(f"💬 DIRECT: Responding directly to '{query}'")
        
        try:
            # Use LLM directly for simple responses
            system_msg = SystemMessage(
                content="You are a helpful AI assistant. Respond concisely and naturally."
            )
            response = llm.invoke([system_msg] + list(messages) + [HumanMessage(content=query)])
            answer = response.content
            
            print(f"✅ Direct answer: {answer[:100]}...")
            
            return {
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
            }
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "messages": [AIMessage(content=error_msg)],
                "final_answer": error_msg,
            }
    
    # Node 6: Synthesize results from multiple skills
    def synthesize_node(state: AgentState) -> Dict[str, Any]:
        """Synthesize results from RAG and/or Tools into final answer"""
        query = state["query"]
        rag_context = state.get("rag_context", "")
        tools_result = state.get("tools_result", "")
        
        print(f"🔄 SYNTHESIZE: Combining results for '{query}'")
        print(f"   RAG context: {len(rag_context)} chars")
        print(f"   Tools result: {len(tools_result)} chars")
        
        try:
            # Build synthesis prompt
            synthesis_prompt = f"""You are a helpful AI assistant. Synthesize the following information to answer the user's question.

User Question: {query}

"""
            if rag_context:
                synthesis_prompt += f"""Information from Knowledge Base:
{rag_context}

"""
            
            if tools_result:
                synthesis_prompt += f"""Information from External Tools:
{tools_result}

"""
            
            synthesis_prompt += """Instructions:
1. Provide a comprehensive answer combining both sources
2. Prioritize the most relevant information
3. Cite sources appropriately
4. Be clear about what comes from knowledge base vs external sources
5. If sources conflict, acknowledge the difference

Your synthesized answer:"""
            
            response = llm.invoke([HumanMessage(content=synthesis_prompt)])
            answer = response.content
            
            print(f"✅ Synthesized answer: {answer[:100]}...")
            
            return {
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
            }
            
        except Exception as e:
            error_msg = f"Synthesis error: {str(e)}"
            print(f"❌ {error_msg}")
            # Fallback: concatenate results
            fallback = ""
            if rag_context:
                fallback += f"From knowledge base:\n{rag_context}\n\n"
            if tools_result:
                fallback += f"From external tools:\n{tools_result}"
            if not fallback:
                fallback = error_msg
            
            return {
                "messages": [AIMessage(content=fallback)],
                "final_answer": fallback,
            }
    
    # Conditional edges: determine next step based on state
    def should_continue_after_rag(state: AgentState) -> str:
        """After RAG: go to tools if needed, synthesis if both complete, or end"""
        tools_completed = state.get("tools_completed", False)
        needs_synthesis = state.get("needs_synthesis", False)
        
        if not tools_completed:
            return "tools"
        elif needs_synthesis:
            return "synthesize"
        else:
            return "end"
    
    def should_continue_after_tools(state: AgentState) -> str:
        """After Tools: go to synthesis if needed, or end"""
        needs_synthesis = state.get("needs_synthesis", False)
        
        if needs_synthesis:
            return "synthesize"
        else:
            return "end"
    
    def route_from_analyze(state: AgentState) -> str:
        """Route from analyze node based on decision"""
        route = state.get("route")
        
        if route == "calculator":
            return "calculator"
        elif route == "rag":
            return "rag"
        elif route == "tools":
            return "tools"
        elif route == "direct":
            return "direct"
        else:
            # Default
            return "rag"
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("direct", direct_node)
    workflow.add_node("synthesize", synthesize_node)
    
    # Add edges
    workflow.set_entry_point("analyze")
    
    # Conditional routing from analyze node
    workflow.add_conditional_edges(
        "analyze",
        route_from_analyze,
        {
            "calculator": "calculator",
            "rag": "rag",
            "tools": "tools",
            "direct": "direct",
        }
    )
    
    # Calculator always ends
    workflow.add_edge("calculator", END)
    
    # RAG can continue to tools, synthesize, or end
    workflow.add_conditional_edges(
        "rag",
        should_continue_after_rag,
        {
            "tools": "tools",
            "synthesize": "synthesize",
            "end": END,
        }
    )
    
    # Tools can continue to synthesize or end
    workflow.add_conditional_edges(
        "tools",
        should_continue_after_tools,
        {
            "synthesize": "synthesize",
            "end": END,
        }
    )
    
    # Direct and synthesize always end
    workflow.add_edge("direct", END)
    workflow.add_edge("synthesize", END)
    
    # Compile and return
    return workflow.compile()


# Convenience functions for external use
async def run_agent_async(
    user_id: str,
    role: str,
    messages: List[Dict[str, str]],
    llm: Optional[ChatOllama] = None,
) -> str:
    """
    Run the agent asynchronously.
    
    Args:
        user_id: User identifier
        role: User role (admin, user, etc.)
        messages: List of message dicts with 'role' and 'content'
        llm: Optional language model instance
    
    Returns:
        Final answer string
    """
    # Create agent graph
    agent = create_agent_graph(llm)
    
    # Extract latest user query
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
    
    if not user_query:
        return "No user query provided."
    
    # Convert messages to LangChain format
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:  # user
            lc_messages.append(HumanMessage(content=content))
    
    # Prepare initial state
    initial_state = {
        "messages": lc_messages,
        "user_id": user_id,
        "role": role,
        "query": user_query,
        "route": None,
        "used_urls": {},
        "rag_context": None,
        "rag_completed": False,
        "tools_result": None,
        "tools_completed": False,
        "needs_synthesis": False,
        "final_answer": "",
    }
    
    # Run the agent
    try:
        result = await agent.ainvoke(initial_state)
        return result.get("final_answer", "No response generated.")
    except Exception as e:
        print(f"❌ Agent execution error: {e}")
        return f"Agent error: {str(e)}"


def run_agent(
    user_id: str,
    role: str,
    messages: List[Dict[str, str]],
    llm: Optional[ChatOllama] = None,
) -> str:
    """
    Run the agent synchronously (wrapper around async version).
    
    Args:
        user_id: User identifier
        role: User role (admin, user, etc.)
        messages: List of message dicts with 'role' and 'content'
        llm: Optional language model instance
    
    Returns:
        Final answer string
    """
    import asyncio
    
    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run async function
    return loop.run_until_complete(
        run_agent_async(user_id, role, messages, llm)
    )
