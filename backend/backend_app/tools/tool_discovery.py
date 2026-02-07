"""
Tool Discovery and Selection Module

This module provides intelligent tool discovery using a hybrid approach:
1. Semantic search via embeddings (for intent matching)
2. Metadata filtering (enabled, scope, tags)
3. Relevance scoring and ranking

Usage:
    from tool_discovery import discover_tools, bind_discovered_tools_to_llm
    
    # Discover relevant tools based on user query
    tools = discover_tools(
        query="I need to search for physics papers",
        user_scope="user@example.com",
        top_k=5
    )
    
    # Bind to LLM
    llm_with_tools = bind_discovered_tools_to_llm(llm, tools)
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

DATABASE_URL = os.getenv("DATABASE_URL", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
COLLECTION_TOOLS = os.getenv("COLLECTION_TOOLS", "tool_embeddings")

# Initialize embeddings and vector store for tools
tool_embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)

tool_vs = PGVector(
    embeddings=tool_embeddings,
    collection_name=COLLECTION_TOOLS,
    connection=DATABASE_URL,
    use_jsonb=True,
)


# ===========================
# Tool Indexing Functions
# ===========================

def index_tool_with_mcp_context(
    tool_id: str, 
    name: str, 
    description: str, 
    enabled: bool, 
    scope: str,
    mcp_description: Optional[str] = None,
    mcp_context: Optional[str] = None,
    mcp_resource: Optional[str] = None
):
    """
    Index a tool with combined tool + MCP descriptions for richer semantic search.
    
    The combined description includes:
    1. Tool's own description (primary)
    2. MCP server description (context about the tool provider)
    3. MCP context (additional usage context)
    4. MCP resource info (what resources it provides)
    
    This provides richer semantic context for better tool discovery.
    
    Example:
        Tool: "arxiv_search"
        Description: "Search arXiv for academic papers"
        MCP Description: "High Energy Physics research tools"
        MCP Context: "Tools for searching academic databases in physics"
        
        Combined: "Search arXiv for academic papers | Provider: High Energy 
                   Physics research tools | Context: Tools for searching 
                   academic databases in physics"
    
    Args:
        tool_id: Unique tool identifier
        name: Tool name
        description: Tool's own description
        enabled: Whether tool is enabled
        scope: Tool scope (global, private, etc.)
        mcp_description: Description of the MCP server providing this tool
        mcp_context: Additional context from MCP server
        mcp_resource: Resource information from MCP server
    """
    # Build combined description for semantic search
    description_parts = []
    
    # Primary: Tool's own description
    if description:
        description_parts.append(description)
    
    # Secondary: MCP provider description
    if mcp_description:
        description_parts.append(f"Provider: {mcp_description}")
    
    # Tertiary: MCP context
    if mcp_context:
        description_parts.append(f"Context: {mcp_context}")
    
    # Quaternary: MCP resources
    if mcp_resource:
        description_parts.append(f"Resources: {mcp_resource}")
    
    # Combine with separator
    combined_description = " | ".join(description_parts) if description_parts else f"Tool: {name}"
    
    # Index with metadata
    tool_vs.add_documents(
        [
            Document(
                page_content=combined_description,
                metadata={
                    "tool_id": tool_id, 
                    "name": name, 
                    "enabled": enabled, 
                    "scope": scope,
                    "has_mcp_context": bool(mcp_description or mcp_context or mcp_resource)
                },
            )
        ]
    )


def index_tool_simple(tool_id: str, name: str, description: str, enabled: bool, scope: str):
    """
    Simple tool indexing without MCP context (backward compatibility).
    
    Args:
        tool_id: Unique tool identifier
        name: Tool name
        description: Tool description
        enabled: Whether tool is enabled
        scope: Tool scope (global, private, etc.)
    """
    tool_vs.add_documents(
        [
            Document(
                page_content=description or f"Tool: {name}",
                metadata={
                    "tool_id": tool_id, 
                    "name": name, 
                    "enabled": enabled, 
                    "scope": scope,
                    "has_mcp_context": False
                },
            )
        ]
    )


# ===========================
# Tool Discovery Functions
# ===========================



def discover_tools(
    query: str,
    user_scope: Optional[str] = None,
    top_k: int = 6,
    enabled_only: bool = True,
    tags: Optional[List[str]] = None,
    min_score: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Discover relevant tools using semantic search + metadata filtering.
    
    Args:
        query: User's natural language query or task description
        user_scope: User identifier for scope filtering (email or user_id)
        top_k: Maximum number of tools to return
        enabled_only: Only return enabled tools
        tags: Optional list of tags to filter by
        min_score: Minimum similarity score threshold (0-1)
    
    Returns:
        List of tool dictionaries with metadata
    
    Example:
        >>> tools = discover_tools(
        ...     "search for recent papers about quantum computing",
        ...     user_scope="user@example.com",
        ...     top_k=5
        ... )
        >>> for tool in tools:
        ...     print(f"{tool['name']}: {tool['description']}")
    """
    # Build metadata filter
    filter_dict = {}
    if enabled_only:
        filter_dict["enabled"] = True
    
    # Semantic search in vector store
    docs_with_scores = tool_vs.similarity_search_with_score(
        query,
        k=top_k * 2,  # Fetch more for filtering
        filter=filter_dict if filter_dict else None
    )
    
    # Process and filter results
    discovered_tools = []
    seen_tool_ids = set()
    
    for doc, score in docs_with_scores:
        # Convert distance to similarity score (lower distance = higher similarity)
        # For cosine distance: similarity = 1 - distance
        similarity = 1.0 - score
        
        if similarity < min_score:
            continue
        
        tool_id = doc.metadata.get("tool_id")
        if not tool_id or tool_id in seen_tool_ids:
            continue
        
        # Scope filtering
        tool_scope = doc.metadata.get("scope", "global")
        if tool_scope == "private" and user_scope and tool_scope != user_scope:
            continue
        
        # Tag filtering
        if tags:
            tool_tags = doc.metadata.get("tags", [])
            if not any(tag in tool_tags for tag in tags):
                continue
        
        seen_tool_ids.add(tool_id)
        
        discovered_tools.append({
            "tool_id": tool_id,
            "name": doc.metadata.get("name", "Unknown"),
            "description": doc.page_content,
            "scope": tool_scope,
            "enabled": doc.metadata.get("enabled", True),
            "tags": doc.metadata.get("tags", []),
            "similarity_score": similarity,
            "metadata": doc.metadata
        })
        
        if len(discovered_tools) >= top_k:
            break
    
    # Sort by similarity score descending
    discovered_tools.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return discovered_tools


def discover_tools_hybrid(
    query: str,
    user_scope: Optional[str] = None,
    top_k: int = 6,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Hybrid tool discovery combining semantic and keyword search.
    
    This approach:
    1. Performs semantic search via embeddings
    2. Performs keyword search on name/description
    3. Combines scores with weights
    
    Args:
        query: User query
        user_scope: User identifier
        top_k: Number of tools to return
        semantic_weight: Weight for semantic similarity (0-1)
        keyword_weight: Weight for keyword match (0-1)
    
    Returns:
        Ranked list of tools
    """
    # Semantic search
    semantic_results = discover_tools(
        query=query,
        user_scope=user_scope,
        top_k=top_k * 2,
        min_score=0.0  # Don't filter yet
    )
    
    # Keyword search (simple implementation)
    # In production, use PostgreSQL full-text search
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Score each tool
    final_scores = {}
    for tool in semantic_results:
        tool_id = tool["tool_id"]
        
        # Semantic score
        semantic_score = tool["similarity_score"]
        
        # Keyword score
        name_lower = tool["name"].lower()
        desc_lower = tool["description"].lower()
        
        # Count matching words
        name_matches = sum(1 for word in query_words if word in name_lower)
        desc_matches = sum(1 for word in query_words if word in desc_lower)
        
        # Normalize keyword score
        max_matches = len(query_words)
        keyword_score = (name_matches * 2 + desc_matches) / (max_matches * 3) if max_matches > 0 else 0
        
        # Combined score
        combined_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
        
        tool["keyword_score"] = keyword_score
        tool["combined_score"] = combined_score
        final_scores[tool_id] = tool
    
    # Sort by combined score
    ranked_tools = sorted(
        final_scores.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )
    
    return ranked_tools[:top_k]


def get_tool_metadata_from_db(tool_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch full tool metadata from database.
    
    Args:
        tool_ids: List of tool IDs
    
    Returns:
        Dictionary mapping tool_id to full metadata
    """
    import psycopg
    
    if not tool_ids:
        return {}
    
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(tool_ids))
            cur.execute(f"""
                SELECT t.id, t.name, t.kind, t.mcp_id, t.metadata, t.tags,
                       m.endpoint as mcp_endpoint, m.kind as mcp_kind
                FROM tools t
                LEFT JOIN mcps m ON t.mcp_id = m.id
                WHERE t.id IN ({placeholders})
            """, tuple(tool_ids))
            
            result = {}
            for row in cur.fetchall():
                tool_id = row[0]
                result[tool_id] = {
                    "id": row[0],
                    "name": row[1],
                    "kind": row[2],
                    "mcp_id": row[3],
                    "metadata": row[4] or {},
                    "tags": row[5] or [],
                    "mcp_endpoint": row[6],
                    "mcp_kind": row[7]
                }
            
            return result


def bind_discovered_tools_to_llm(llm, discovered_tools: List[Dict[str, Any]]):
    """
    Bind discovered tools to an LLM instance.
    
    This function:
    1. Converts discovered tools to LangChain tool format
    2. Binds them to the LLM for structured output
    
    Args:
        llm: LangChain LLM instance
        discovered_tools: List of tools from discover_tools()
    
    Returns:
        LLM with tools bound
    
    Example:
        >>> llm = ChatOllama(model="llama2")
        >>> tools = discover_tools("search for papers")
        >>> llm_with_tools = bind_discovered_tools_to_llm(llm, tools)
        >>> result = llm_with_tools.invoke("Find papers about AI")
    """
    from langchain_core.tools import tool as tool_decorator
    from .client import run_mcp_tool_async
    
    # Get full metadata from database
    tool_ids = [t["tool_id"] for t in discovered_tools]
    full_metadata = get_tool_metadata_from_db(tool_ids)
    
    # Create LangChain tools
    langchain_tools = []
    
    for disc_tool in discovered_tools:
        tool_id = disc_tool["tool_id"]
        full_meta = full_metadata.get(tool_id, {})
        
        # Create a closure to capture tool_id and mcp info
        def create_tool_func(tid, mcp_endpoint, tool_name, tool_meta):
            async def tool_func(query: str) -> str:
                """Execute the tool via MCP."""
                result = await run_mcp_tool_async(
                    mcp_endpoint=mcp_endpoint,
                    tool_name=tool_name,
                    arguments={"query": query}  # Adjust based on tool schema
                )
                return str(result)
            return tool_func
        
        # Dynamically create tool
        tool_func = create_tool_func(
            tool_id,
            full_meta.get("mcp_endpoint"),
            disc_tool["name"],
            full_meta.get("metadata", {})
        )
        
        # Decorate with @tool
        decorated_tool = tool_decorator(
            name=disc_tool["name"],
            description=disc_tool["description"]
        )(tool_func)
        
        langchain_tools.append(decorated_tool)
    
    # Bind tools to LLM
    return llm.bind_tools(langchain_tools)


# ===========================
# Utility Functions
# ===========================

def reindex_all_tools():
    """
    Re-index all tools from database into vector store with MCP context.
    Combines tool description with MCP server description, context, and resources.
    Useful after bulk updates or schema changes.
    """
    import psycopg
    
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Fetch tools with joined MCP information
            cur.execute("""
                SELECT 
                    t.id, 
                    t.name, 
                    t.metadata->>'description' as tool_description,
                    t.metadata->>'enabled' as enabled, 
                    t.metadata->>'scope' as scope,
                    t.tags,
                    m.description as mcp_description,
                    m.context as mcp_context,
                    m.resource as mcp_resource
                FROM tools t
                LEFT JOIN mcps m ON t.mcp_id = m.id
            """)
            
            documents = []
            for row in cur.fetchall():
                (tool_id, name, tool_description, enabled, scope, tags,
                 mcp_description, mcp_context, mcp_resource) = row
                
                # Build combined description
                description_parts = []
                
                if tool_description:
                    description_parts.append(tool_description)
                
                if mcp_description:
                    description_parts.append(f"Provider: {mcp_description}")
                
                if mcp_context:
                    description_parts.append(f"Context: {mcp_context}")
                
                if mcp_resource:
                    description_parts.append(f"Resources: {mcp_resource}")
                
                combined_description = " | ".join(description_parts) if description_parts else f"Tool: {name}"
                
                documents.append(Document(
                    page_content=combined_description,
                    metadata={
                        "tool_id": tool_id,
                        "name": name,
                        "enabled": enabled == "true" or enabled is True,
                        "scope": scope or "global",
                        "tags": tags or [],
                        "has_mcp_context": bool(mcp_description or mcp_context or mcp_resource)
                    }
                ))
            
            # Clear and re-index
            tool_vs.delete(filter={})  # Clear all
            if documents:
                tool_vs.add_documents(documents)
                print(f"Re-indexed {len(documents)} tools with MCP context")


def search_tools_by_tags(tags: List[str], match_all: bool = False) -> List[Dict[str, Any]]:
    """
    Search tools by tags without semantic search.
    
    Args:
        tags: List of tags to search for
        match_all: If True, tool must have all tags. If False, any tag matches.
    
    Returns:
        List of matching tools
    """
    import psycopg
    
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            if match_all:
                # Tool must contain all tags
                tag_conditions = " AND ".join([f"tags::jsonb @> '[%s]'::jsonb" for _ in tags])
                cur.execute(f"""
                    SELECT id, name, metadata->>'description' as description,
                           tags, metadata->>'scope' as scope
                    FROM tools
                    WHERE {tag_conditions}
                    AND (metadata->>'enabled')::boolean = true
                """, tuple(tags))
            else:
                # Tool must contain any tag
                tag_list = ",".join([f"'{tag}'" for tag in tags])
                cur.execute(f"""
                    SELECT id, name, metadata->>'description' as description,
                           tags, metadata->>'scope' as scope
                    FROM tools
                    WHERE tags::jsonb ?| array[{tag_list}]
                    AND (metadata->>'enabled')::boolean = true
                """)
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "tool_id": row[0],
                    "name": row[1],
                    "description": row[2] or "",
                    "tags": row[3] or [],
                    "scope": row[4] or "global"
                })
            
            return results


if __name__ == "__main__":
    # Example usage
    print("Tool Discovery Examples\n" + "="*50)
    
    # Example 1: Semantic search
    print("\n1. Semantic Search:")
    tools = discover_tools("I need to search for physics papers from arXiv", top_k=3)
    for i, tool in enumerate(tools, 1):
        print(f"   {i}. {tool['name']} (score: {tool['similarity_score']:.3f})")
        print(f"      {tool['description'][:80]}...")
    
    # Example 2: Hybrid search
    print("\n2. Hybrid Search:")
    tools = discover_tools_hybrid("calculate statistics on data", top_k=3)
    for i, tool in enumerate(tools, 1):
        print(f"   {i}. {tool['name']} (combined: {tool['combined_score']:.3f})")
    
    # Example 3: Tag-based search
    print("\n3. Tag-based Search:")
    tools = search_tools_by_tags(["search", "web"])
    for tool in tools[:3]:
        print(f"   - {tool['name']}: {', '.join(tool['tags'])}")
