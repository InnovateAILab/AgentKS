"""
LangGraph RAG Skill

A RAG (Retrieval-Augmented Generation) skill using LangGraph that retrieves
and generates responses using content from the RAG MCP service.

This skill:
1. Queries the RAG MCP service for relevant documents
2. Retrieves and ranks content using vector similarity
3. Generates answers based on retrieved context
4. Provides source citations and metadata
"""

import os
import json
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools import run_mcp_tool_async


# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
RAG_MCP_URL = os.getenv("RAG_MCP_URL", "http://localhost:4002/mcp")
DEFAULT_RAG_K = int(os.getenv("DEFAULT_RAG_K", "5"))
DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", "0.3"))


class RAGSkillState(TypedDict):
    """State for the RAG skill workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    rag_group: Optional[str]
    k: int
    score_threshold: float
    retrieved_docs: List[Dict[str, Any]]
    context: str
    sources: List[Dict[str, str]]


def create_rag_skill_graph(
    llm: Optional[ChatOllama] = None,
    rag_mcp_url: str = RAG_MCP_URL,
    default_k: int = DEFAULT_RAG_K,
    default_score_threshold: float = DEFAULT_SCORE_THRESHOLD,
):
    """
    Create a LangGraph workflow for RAG-based question answering.
    
    Args:
        llm: Language model (defaults to ChatOllama with configured model)
        rag_mcp_url: URL of the RAG MCP service
        default_k: Default number of documents to retrieve
        default_score_threshold: Default minimum similarity score
    
    Returns:
        Compiled LangGraph workflow
    """
    
    if llm is None:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)
    
    # Node 1: Retrieve documents from RAG MCP
    async def retrieve_node(state: RAGSkillState) -> Dict[str, Any]:
        """Retrieve relevant documents from RAG MCP service"""
        query = state["query"]
        rag_group = state.get("rag_group")
        k = state.get("k", default_k)
        score_threshold = state.get("score_threshold", default_score_threshold)
        
        print(f"🔍 Retrieving documents for query: '{query}'")
        if rag_group:
            print(f"   RAG group: {rag_group}")
        print(f"   k={k}, score_threshold={score_threshold}")
        
        try:
            # Call RAG MCP service
            result = await run_mcp_tool_async(
                mcp_url=rag_mcp_url,
                headers={},
                tool_name="rag_search",
                payload={
                    "query": query,
                    "k": k,
                    "rag_group": rag_group,
                    "score_threshold": score_threshold,
                }
            )
            
            # Parse result
            if isinstance(result, str):
                result = json.loads(result)
            
            if "error" in result:
                print(f"❌ RAG retrieval error: {result.get('message', 'Unknown error')}")
                return {
                    "retrieved_docs": [],
                    "context": f"Error retrieving documents: {result.get('message', 'Unknown error')}",
                    "sources": [],
                }
            
            retrieved_docs = result.get("results", [])
            print(f"✅ Retrieved {len(retrieved_docs)} documents")
            
            # Build context from retrieved documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(retrieved_docs, 1):
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                score = doc.get("similarity_score", 0)
                
                # Extract source information
                source_url = metadata.get("source_url", "unknown")
                rag_group_name = metadata.get("rag_group", "unknown")
                
                # Add to context
                context_parts.append(
                    f"[Document {i}] (score: {score:.3f}, group: {rag_group_name})\n{content}"
                )
                
                # Track sources
                sources.append({
                    "index": str(i),
                    "url": source_url,
                    "rag_group": rag_group_name,
                    "score": f"{score:.3f}",
                })
            
            context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."
            
            return {
                "retrieved_docs": retrieved_docs,
                "context": context,
                "sources": sources,
            }
            
        except Exception as e:
            print(f"❌ Exception during retrieval: {e}")
            return {
                "retrieved_docs": [],
                "context": f"Error: {str(e)}",
                "sources": [],
            }
    
    # Node 2: Generate answer using LLM with retrieved context
    def generate_node(state: RAGSkillState) -> Dict[str, Any]:
        """Generate answer using LLM with retrieved context"""
        query = state["query"]
        context = state.get("context", "")
        sources = state.get("sources", [])
        
        print(f"🤖 Generating answer using LLM")
        
        # Create system message with instructions
        system_content = """You are a helpful AI assistant that answers questions based on retrieved documents.

Instructions:
1. Use ONLY the information from the provided documents to answer the question
2. If the documents don't contain relevant information, say so clearly
3. Cite sources using [1], [2], etc. corresponding to the document numbers
4. Be concise and direct
5. If information is uncertain or incomplete, acknowledge it

Context from retrieved documents:
---
{context}
---
"""
        
        system_msg = SystemMessage(content=system_content.format(context=context))
        user_msg = HumanMessage(content=f"Question: {query}")
        
        # Generate response
        try:
            response = llm.invoke([system_msg, user_msg])
            answer = response.content
            
            # Add source references if we have sources
            if sources:
                refs = [
                    f"[{s['index']}] {s['url']} (group: {s['rag_group']}, score: {s['score']})"
                    for s in sources
                ]
                answer += "\n\n---\n\n**Sources:**\n" + "\n".join(refs)
            
            print(f"✅ Generated answer ({len(answer)} chars)")
            
            return {
                "messages": [AIMessage(content=answer)],
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            error_msg = f"Error generating answer: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)],
            }
    
    # Conditional edge: check if we should proceed to generation
    def should_generate(state: RAGSkillState) -> Literal["generate", "end"]:
        """Determine if we should generate an answer"""
        retrieved_docs = state.get("retrieved_docs", [])
        
        # If we have documents, proceed to generation
        if retrieved_docs:
            return "generate"
        
        # If no documents but we have a context (e.g., error message), still generate
        context = state.get("context", "")
        if context:
            return "generate"
        
        # Otherwise, end
        return "end"
    
    # Build the graph
    workflow = StateGraph(RAGSkillState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    # Add edges
    workflow.set_entry_point("retrieve")
    
    # Conditional edge from retrieve
    workflow.add_conditional_edges(
        "retrieve",
        should_generate,
        {
            "generate": "generate",
            "end": END,
        }
    )
    
    # After generate, we're done
    workflow.add_edge("generate", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


async def run_rag_skill_async(
    query: str,
    rag_group: Optional[str] = None,
    k: int = DEFAULT_RAG_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    llm: Optional[ChatOllama] = None,
    rag_mcp_url: str = RAG_MCP_URL,
) -> str:
    """
    Run the RAG skill workflow asynchronously.
    
    Args:
        query: User question
        rag_group: Optional RAG group to search within
        k: Number of documents to retrieve
        score_threshold: Minimum similarity score
        llm: Language model instance
        rag_mcp_url: URL of RAG MCP service
    
    Returns:
        Generated answer as a string
    """
    
    # Create the graph
    app = create_rag_skill_graph(
        llm=llm,
        rag_mcp_url=rag_mcp_url,
        default_k=k,
        default_score_threshold=score_threshold,
    )
    
    # Prepare initial state
    initial_state: RAGSkillState = {
        "messages": [],
        "query": query,
        "rag_group": rag_group,
        "k": k,
        "score_threshold": score_threshold,
        "retrieved_docs": [],
        "context": "",
        "sources": [],
    }
    
    # Run the workflow
    result = await app.ainvoke(initial_state)
    
    # Extract the final response
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    
    return "No response generated"


def run_rag_skill(
    query: str,
    rag_group: Optional[str] = None,
    k: int = DEFAULT_RAG_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    llm: Optional[ChatOllama] = None,
    rag_mcp_url: str = RAG_MCP_URL,
) -> str:
    """
    Run the RAG skill workflow (synchronous wrapper).
    
    Args:
        query: User question
        rag_group: Optional RAG group to search within
        k: Number of documents to retrieve
        score_threshold: Minimum similarity score
        llm: Language model instance
        rag_mcp_url: URL of RAG MCP service
    
    Returns:
        Generated answer as a string
    """
    import asyncio
    
    return asyncio.run(run_rag_skill_async(
        query=query,
        rag_group=rag_group,
        k=k,
        score_threshold=score_threshold,
        llm=llm,
        rag_mcp_url=rag_mcp_url,
    ))


async def retrieve_documents_async(
    query: str,
    rag_group: Optional[str] = None,
    k: int = DEFAULT_RAG_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    rag_mcp_url: str = RAG_MCP_URL,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents from RAG MCP service without LLM generation.
    
    This function only performs the retrieval step, returning raw documents
    with their metadata and similarity scores. No LLM generation is performed.
    
    Args:
        query: Search query
        rag_group: Optional RAG group to search within
        k: Number of documents to retrieve
        score_threshold: Minimum similarity score
        rag_mcp_url: URL of RAG MCP service
    
    Returns:
        List of document dictionaries with content, metadata, and scores
    """
    try:
        # Call RAG MCP service
        result = await run_mcp_tool_async(
            mcp_url=rag_mcp_url,
            headers={},
            tool_name="rag_search",
            payload={
                "query": query,
                "k": k,
                "rag_group": rag_group,
                "score_threshold": score_threshold,
            }
        )
        
        # Parse result
        if isinstance(result, str):
            result = json.loads(result)
        
        if "error" in result:
            print(f"❌ RAG retrieval error: {result.get('message', 'Unknown error')}")
            return []
        
        retrieved_docs = result.get("results", [])
        return retrieved_docs
        
    except Exception as e:
        print(f"❌ Exception during retrieval: {e}")
        return []


def retrieve_documents(
    query: str,
    rag_group: Optional[str] = None,
    k: int = DEFAULT_RAG_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    rag_mcp_url: str = RAG_MCP_URL,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents from RAG MCP service without LLM generation (synchronous wrapper).
    
    This function only performs the retrieval step, returning raw documents
    with their metadata and similarity scores. No LLM generation is performed.
    
    Args:
        query: Search query
        rag_group: Optional RAG group to search within
        k: Number of documents to retrieve
        score_threshold: Minimum similarity score
        rag_mcp_url: URL of RAG MCP service
    
    Returns:
        List of document dictionaries with content, metadata, and scores
    """
    import asyncio
    
    return asyncio.run(retrieve_documents_async(
        query=query,
        rag_group=rag_group,
        k=k,
        score_threshold=score_threshold,
        rag_mcp_url=rag_mcp_url,
    ))


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_rag_skill():
        """Test the RAG skill"""
        
        # Example 1: General search
        print("\n" + "="*60)
        print("Test 1: General knowledge base search")
        print("="*60)
        result = await run_rag_skill_async(
            query="What is quantum computing?",
            k=3,
            score_threshold=0.3,
        )
        print(f"\n📝 Result:\n{result}\n")
        
        # Example 2: Group-specific search
        print("\n" + "="*60)
        print("Test 2: Search within specific RAG group")
        print("="*60)
        result = await run_rag_skill_async(
            query="Tell me about CERN experiments",
            rag_group="physics_docs",
            k=5,
            score_threshold=0.4,
        )
        print(f"\n📝 Result:\n{result}\n")
    
    asyncio.run(test_rag_skill())
