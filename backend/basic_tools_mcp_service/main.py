"""
Basic Tools MCP Server

A FastMCP-based Model Context Protocol server exposing utility and search tools.
Uses SSE (Server-Sent Events) transport over HTTP for web compatibility.
Leverages LangChain Community tools for robust, maintained implementations.
"""
import os
import logging
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

# LangChain Community imports
try:
    from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
    from langchain_community.tools.arxiv.tool import ArxivQueryRun
    from langchain_community.utilities.arxiv import ArxivAPIWrapper
    from langchain_community.retrievers.wikipedia import WikipediaRetriever
    from langchain_community.retrievers.pubmed import PubMedRetriever
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
    from langchain_community.utilities.searx_search import SearxSearchWrapper
    from langchain.tools.retriever import create_retriever_tool
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_ERROR = str(e)

# For SearXNG direct API calls (fallback if LangChain not available)
import requests

# Import HEP tools module
from hep import register_hep_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance with metadata
mcp = FastMCP(
    "basic-tools-mcp",
    version="1.0.0"
)

# Add server description, context, and resources via prompts
@mcp.prompt()
def server_info():
    """
    Basic Tools MCP Server - Comprehensive search and research toolkit.
    
    This server provides:
    - General web search (DuckDuckGo, SearXNG meta-search)
    - Academic research (arXiv, Wikipedia, PubMed)
    - AI-powered search (Tavily)
    - High Energy Physics literature (INSPIRE-HEP, CERN CDS, arXiv HEP)
    
    Context: Designed for AgentKS knowledge stack, integrating multiple search sources
    for comprehensive information retrieval and research assistance.
    
    Resources:
    - LangChain Community tools for reliable implementations
    - SearXNG for privacy-focused meta-search
    - INSPIRE-HEP for HEP literature (https://inspirehep.net)
    - CERN CDS for institutional repository (https://cds.cern.ch)
    - arXiv for preprints (https://arxiv.org)
    
    Use cases:
    - Literature review and research
    - Web information gathering
    - Scientific paper discovery
    - Medical literature search
    - High Energy Physics research
    """
    return "Server information available via tool discovery"


@mcp.resource("server://metadata")
def get_server_metadata() -> str:
    """
    Server metadata and capabilities.
    
    Returns comprehensive information about available tools and their purposes.
    """
    return """
# Basic Tools MCP Server Metadata

## Description
Comprehensive search and research toolkit providing access to multiple information sources
through a unified MCP interface. Built with FastMCP and LangChain Community tools.

## Available Tool Categories

### General Search (5 tools)
- ddg_search: DuckDuckGo web search
- searxng_search: Privacy-focused meta-search (multiple engines)
- wikipedia_search: Wikipedia article lookup
- tavily_search: AI-powered search with summarization
- arxiv_search: Academic paper search

### Medical & Scientific (1 tool)
- pubmed_search: Medical and life sciences literature

### High Energy Physics (3 tools)
- inspirehep_search: INSPIRE-HEP literature database
- cds_search: CERN Document Server
- arxiv_hep_search: arXiv with HEP category filtering

### Utilities (2 tools)
- echo: Simple echo utility
- add: Math operations demo

## Context
This MCP server is part of the AgentKS (Agent Knowledge Stack) system, designed to provide
comprehensive research and information retrieval capabilities for AI agents and applications.

## Resources
- GitHub: https://github.com/InnovateAILab/AgentKS
- LangChain Community: https://python.langchain.com/docs/integrations/tools/
- MCP Protocol: https://modelcontextprotocol.io/

## Configuration
Set these environment variables as needed:
- TAVILY_API_KEY: For Tavily AI search
- SEARXNG_URL: SearXNG instance URL (default: http://searxng:8080)
- INSPIRE_BASE_URL: INSPIRE-HEP base URL
- CDS_BASE_URL: CERN CDS base URL
- ARXIV_API_URL: arXiv API endpoint

## Version
1.0.0
"""


@mcp.resource("context://general-search")
def general_search_context() -> str:
    """Context and guidance for general web search tools."""
    return """
# General Web Search Tools Context

## Available Tools
1. **ddg_search**: DuckDuckGo web search - privacy-focused, no tracking
2. **searxng_search**: Meta-search aggregating results from multiple engines
3. **wikipedia_search**: Wikipedia articles with summaries
4. **tavily_search**: AI-powered search with answer generation (requires API key)

## When to Use Each Tool

### Use ddg_search when:
- Quick web search needed
- Privacy is important
- No API key required
- General information gathering

### Use searxng_search when:
- Need results from multiple search engines
- Want aggregated/diverse sources
- Have access to SearXNG instance
- Category-specific search (general, science, news, etc.)

### Use wikipedia_search when:
- Need factual, encyclopedic information
- Want authoritative reference content
- Research background information
- Educational purposes

### Use tavily_search when:
- Need AI-powered answer summarization
- Want most relevant, filtered results
- Have Tavily API key
- Need quick, concise answers

## Best Practices
- Start with ddg_search or searxng_search for broad queries
- Use wikipedia_search for definitions and background
- Use tavily_search when you need AI-summarized answers
- Combine multiple tools for comprehensive research
"""


@mcp.resource("context://academic-research")
def academic_research_context() -> str:
    """Context and guidance for academic research tools."""
    return """
# Academic Research Tools Context

## Available Tools
1. **arxiv_search**: General arXiv preprint search (all categories)
2. **pubmed_search**: Medical and life sciences literature
3. **inspirehep_search**: High Energy Physics literature (INSPIRE-HEP)
4. **cds_search**: CERN Document Server (institutional repository)
5. **arxiv_hep_search**: arXiv with HEP category filtering

## When to Use Each Tool

### Use arxiv_search when:
- Searching for preprints across all disciplines
- Need recent research (papers before peer review)
- Looking for physics, math, CS, or related fields
- Want comprehensive arXiv coverage

### Use pubmed_search when:
- Medical or biomedical research
- Life sciences literature
- Clinical studies and trials
- Health-related information

### Use inspirehep_search when:
- High Energy Physics research
- Particle physics literature
- Need citation counts for HEP papers
- Comprehensive HEP database required

### Use cds_search when:
- Looking for CERN publications
- Need theses or technical reports
- Accessing institutional repository
- CERN-specific documents

### Use arxiv_hep_search when:
- Need HEP-specific arXiv search
- Want category filtering (hep-ph, hep-th, hep-ex, hep-lat)
- Prefer arXiv interface over INSPIRE
- Need PDF links and arXiv IDs

## Best Practices
- For HEP: Start with inspirehep_search (most comprehensive)
- For medical: Always use pubmed_search
- For recent research: Use arxiv_search or arxiv_hep_search
- For CERN-specific: Use cds_search
- Check citation counts in inspirehep_search for impact assessment
"""


@mcp.resource("context://hep-research")
def hep_research_context() -> str:
    """Context and guidance for High Energy Physics research."""
    return """
# High Energy Physics Research Context

## HEP-Specific Tools
1. **inspirehep_search**: Primary HEP literature database
2. **cds_search**: CERN Document Server
3. **arxiv_hep_search**: arXiv with HEP category filters

## HEP arXiv Categories
- **hep-ph**: Phenomenology (particle physics theory and experiment connection)
- **hep-th**: Theory (theoretical high energy physics)
- **hep-ex**: Experiment (experimental results and techniques)
- **hep-lat**: Lattice (lattice field theory and QCD)

## Research Workflow

### For Literature Review:
1. Start with inspirehep_search - most comprehensive HEP database
2. Sort by "mostcited" to find influential papers
3. Use arxiv_hep_search for recent preprints in specific categories
4. Check cds_search for CERN technical reports and theses

### For Recent Papers:
1. Use arxiv_hep_search with sort_by="submittedDate"
2. Filter by relevant HEP category (hep-ph, hep-th, etc.)
3. Cross-reference with inspirehep_search for citation context

### For Impact Assessment:
1. Use inspirehep_search with sort="mostcited"
2. Check citation_count in results
3. Look for papers with high citation counts and recent dates

## Best Practices
- INSPIRE-HEP includes arXiv + journal publications + citations
- CDS is best for CERN-specific technical documents
- arXiv HEP search is fastest for recent preprints
- Always check multiple sources for comprehensive coverage
- Use inspirehep_search for citation analysis
"""

# Lazy-loaded tool instances
_ddg_tool = None
_arxiv_tool = None
_wikipedia_tool = None
_pubmed_tool = None
_tavily_tool = None
_searxng_tool = None


def get_ddg_tool():
    """Lazy initialization of DuckDuckGo search tool."""
    global _ddg_tool
    if _ddg_tool is None:
        _ddg_tool = DuckDuckGoSearchRun()
    return _ddg_tool


def get_arxiv_tool():
    """Lazy initialization of Arxiv search tool."""
    global _arxiv_tool
    if _arxiv_tool is None:
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=2000)
        _arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    return _arxiv_tool


def get_wikipedia_tool():
    """Lazy initialization of Wikipedia retriever tool."""
    global _wikipedia_tool
    if _wikipedia_tool is None:
        retriever = WikipediaRetriever(top_k_results=3, doc_content_chars_max=2000)
        _wikipedia_tool = create_retriever_tool(
            retriever,
            "wikipedia_search",
            "Search Wikipedia for information on a topic. Returns article summaries."
        )
    return _wikipedia_tool


def get_pubmed_tool():
    """Lazy initialization of PubMed retriever tool."""
    global _pubmed_tool
    if _pubmed_tool is None:
        retriever = PubMedRetriever(top_k_results=3)
        _pubmed_tool = create_retriever_tool(
            retriever,
            "pubmed_search",
            "Search PubMed for medical and scientific research papers."
        )
    return _pubmed_tool


def get_tavily_tool():
    """Lazy initialization of Tavily search tool."""
    global _tavily_tool
    if _tavily_tool is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return None
        tavily_wrapper = TavilySearchAPIWrapper(tavily_api_key=api_key)
        _tavily_tool = TavilySearchResults(api_wrapper=tavily_wrapper, max_results=5)
    return _tavily_tool


def get_searxng_tool():
    """Lazy initialization of SearXNG search tool."""
    global _searxng_tool
    if _searxng_tool is None:
        searxng_url = os.getenv("SEARXNG_URL", "http://searxng:8080")
        try:
            # Try to use LangChain's SearxSearchWrapper
            _searxng_tool = SearxSearchWrapper(searx_host=searxng_url)
        except Exception:
            # If LangChain wrapper fails, we'll use direct API calls
            _searxng_tool = None
    return _searxng_tool


# ============================================================================
# MCP Tool Definitions
# ============================================================================

@mcp.tool()
def echo(text: str) -> str:
    """
    Return the provided input text.
    
    Args:
        text: Text to echo back
    """
    return f"Echo: {text}"


@mcp.tool()
def add(a: float, b: float) -> dict:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Dictionary with the result
    """
    result = a + b
    return {"result": result, "operation": "addition", "inputs": {"a": a, "b": b}}


@mcp.tool()
def ddg_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo.
    Uses LangChain Community's DuckDuckGoSearchRun tool.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Dictionary with search results or error information
    """
    if not LANGCHAIN_AVAILABLE:
        return {
            "error": "LangChain Community not available",
            "details": LANGCHAIN_ERROR,
            "query": query
        }
    
    try:
        tool = get_ddg_tool()
        # DuckDuckGoSearchRun returns a string with results
        results = tool.run(query)
        return {
            "query": query,
            "results": results,
            "source": "duckduckgo"
        }
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
        return {
            "error": str(e),
            "query": query,
            "source": "duckduckgo"
        }


@mcp.tool()
def arxiv_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search academic papers on Arxiv.
    Uses LangChain Community's ArxivQueryRun tool.
    
    Args:
        query: Search query for academic papers
        max_results: Maximum number of papers to return (default: 5)
    
    Returns:
        Dictionary with paper information or error details
    """
    if not LANGCHAIN_AVAILABLE:
        return {
            "error": "LangChain Community not available",
            "details": LANGCHAIN_ERROR,
            "query": query
        }
    
    try:
        tool = get_arxiv_tool()
        # ArxivQueryRun returns formatted string with paper details
        results = tool.run(query)
        return {
            "query": query,
            "results": results,
            "source": "arxiv"
        }
    except Exception as e:
        logger.error(f"Arxiv search error: {e}")
        return {
            "error": str(e),
            "query": query,
            "source": "arxiv"
        }


@mcp.tool()
def wikipedia_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search Wikipedia for article summaries.
    Uses LangChain Community's WikipediaRetriever.
    
    Args:
        query: Topic to search for on Wikipedia
        max_results: Maximum number of articles to return (default: 3)
    
    Returns:
        Dictionary with article summaries or error information
    """
    if not LANGCHAIN_AVAILABLE:
        return {
            "error": "LangChain Community not available",
            "details": LANGCHAIN_ERROR,
            "query": query
        }
    
    try:
        tool = get_wikipedia_tool()
        # Retriever tool returns a string with documents
        results = tool.run(query)
        return {
            "query": query,
            "results": results,
            "source": "wikipedia"
        }
    except Exception as e:
        logger.error(f"Wikipedia search error: {e}")
        return {
            "error": str(e),
            "query": query,
            "source": "wikipedia"
        }


@mcp.tool()
def pubmed_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search PubMed for medical and scientific research papers.
    Uses LangChain Community's PubMedRetriever.
    
    Args:
        query: Medical/scientific topic to search
        max_results: Maximum number of papers to return (default: 3)
    
    Returns:
        Dictionary with paper information or error details
    """
    if not LANGCHAIN_AVAILABLE:
        return {
            "error": "LangChain Community not available",
            "details": LANGCHAIN_ERROR,
            "query": query
        }
    
    try:
        tool = get_pubmed_tool()
        # Retriever tool returns a string with documents
        results = tool.run(query)
        return {
            "query": query,
            "results": results,
            "source": "pubmed"
        }
    except Exception as e:
        logger.error(f"PubMed search error: {e}")
        return {
            "error": str(e),
            "query": query,
            "source": "pubmed"
        }


@mcp.tool()
def tavily_search(query: str, max_results: int = 5, include_answer: bool = True) -> Dict[str, Any]:
    """
    Search using Tavily AI-powered search engine.
    Uses LangChain Community's TavilySearchResults tool.
    Requires TAVILY_API_KEY environment variable.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        include_answer: Whether to include AI-generated answer (default: True)
    
    Returns:
        Dictionary with search results or error information
    """
    if not LANGCHAIN_AVAILABLE:
        return {
            "error": "LangChain Community not available",
            "details": LANGCHAIN_ERROR,
            "query": query
        }
    
    try:
        tool = get_tavily_tool()
        if tool is None:
            return {
                "error": "TAVILY_API_KEY environment variable not set",
                "query": query,
                "source": "tavily"
            }
        
        # TavilySearchResults returns list of result dicts
        results = tool.run(query)
        return {
            "query": query,
            "results": results,
            "source": "tavily",
            "include_answer": include_answer
        }
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return {
            "error": str(e),
            "query": query,
            "source": "tavily"
        }


@mcp.tool()
def searxng_search(query: str, max_results: int = 5, categories: str = "general") -> Dict[str, Any]:
    """
    Search using SearXNG meta-search engine.
    SearXNG aggregates results from multiple search engines.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        categories: Search categories, e.g., "general", "science", "news" (default: "general")
    
    Returns:
        Dictionary with search results from multiple engines or error information
    """
    searxng_url = os.getenv("SEARXNG_URL", "http://searxng:8080")
    
    # Try LangChain wrapper first
    if LANGCHAIN_AVAILABLE:
        try:
            tool = get_searxng_tool()
            if tool is not None:
                results = tool.run(query)
                return {
                    "query": query,
                    "results": results,
                    "source": "searxng",
                    "url": searxng_url
                }
        except Exception as e:
            logger.warning(f"LangChain SearXNG wrapper failed: {e}, falling back to direct API")
    
    # Fallback to direct API call
    try:
        response = requests.get(
            f"{searxng_url.rstrip('/')}/search",
            params={
                "q": query,
                "format": "json",
                "categories": categories
            },
            timeout=20
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in (data.get("results") or [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "engine": item.get("engine", "")
            })
        
        return {
            "query": query,
            "total": len(results),
            "results": results,
            "source": "searxng",
            "url": searxng_url,
            "categories": categories
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"SearXNG search error: {e}")
        return {
            "error": f"SearXNG API error: {str(e)}",
            "query": query,
            "source": "searxng",
            "url": searxng_url
        }
    except Exception as e:
        logger.error(f"Unexpected SearXNG error: {e}")
        return {
            "error": str(e),
            "query": query,
            "source": "searxng"
        }


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Basic Tools MCP Service with LangChain Community tools")
    logger.info("LangChain Community available: %s", LANGCHAIN_AVAILABLE)
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain Community not available: %s", LANGCHAIN_ERROR)
    
    # Register HEP-specific tools
    logger.info("Registering High Energy Physics (HEP) tools...")
    register_hep_tools(mcp)
    
    # Add custom HTTP endpoint for discovery metadata
    # This allows the admin UI to discover the MCP server
    from fastapi import FastAPI, Response
    from fastapi.responses import JSONResponse
    
    # Get the underlying FastAPI app from FastMCP
    app = mcp.app
    
    @app.get("/")
    @app.get("/.well-known/mcp")
    @app.get("/mcp")
    def discovery_endpoint():
        """
        Discovery endpoint for admin UI and MCP clients.
        Provides server metadata in a standard format.
        """
        return JSONResponse({
            "name": "basic-tools-mcp",
            "title": "Basic Tools MCP Server",
            "version": "1.0.0",
            "description": "Comprehensive search and research toolkit providing access to multiple information sources through a unified MCP interface. Includes web search, academic databases, and High Energy Physics literature.",
            "service": "basic-tools-mcp",
            "protocol": "mcp",
            "transport": "sse",
            "context": "Designed for AgentKS knowledge stack, integrating multiple search sources for comprehensive information retrieval and research assistance.",
            "resource": "https://github.com/InnovateAILab/AgentKS",
            "tags": ["search", "research", "academic", "hep", "langchain", "arxiv", "pubmed", "wikipedia"],
            "capabilities": {
                "tools": 11,
                "categories": ["general-search", "academic-research", "hep-literature", "utilities"],
                "resources": ["server://metadata", "context://general-search", "context://academic-research", "context://hep-research"],
                "prompts": ["server_info"]
            },
            "endpoints": {
                "sse": "/sse",
                "health": "/health",
                "discovery": "/.well-known/mcp"
            },
            "tools": [
                {"name": "echo", "category": "utility", "description": "Echo text input"},
                {"name": "add", "category": "utility", "description": "Add two numbers"},
                {"name": "ddg_search", "category": "general-search", "description": "DuckDuckGo web search"},
                {"name": "arxiv_search", "category": "academic-research", "description": "Search arXiv preprints"},
                {"name": "wikipedia_search", "category": "general-search", "description": "Search Wikipedia articles"},
                {"name": "pubmed_search", "category": "academic-research", "description": "Search PubMed medical literature"},
                {"name": "tavily_search", "category": "general-search", "description": "AI-powered search with Tavily"},
                {"name": "searxng_search", "category": "general-search", "description": "Meta-search via SearXNG"},
                {"name": "inspirehep_search", "category": "hep-literature", "description": "INSPIRE-HEP literature database"},
                {"name": "cds_search", "category": "hep-literature", "description": "CERN Document Server"},
                {"name": "arxiv_hep_search", "category": "hep-literature", "description": "arXiv with HEP category filtering"}
            ],
            "configuration": {
                "required": [],
                "optional": ["TAVILY_API_KEY", "SEARXNG_URL", "INSPIRE_BASE_URL", "CDS_BASE_URL", "ARXIV_API_URL"]
            }
        })
    
    # Run FastMCP server with SSE transport on port 5000
    logger.info("Starting FastMCP server on http://0.0.0.0:5000")
    logger.info("Discovery endpoint: http://0.0.0.0:5000/.well-known/mcp")
    mcp.run(transport="sse", port=5000, host="0.0.0.0")
