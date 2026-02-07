# Basic Tools MCP Server

A Model Context Protocol (MCP) server exposing basic utility, math, and search tools.
Built with FastMCP using SSE (Server-Sent Events) transport over HTTP.

**Uses LangChain Community Tools**: Leverages battle-tested, maintained tools from the LangChain Community package for robust and consistent search functionality.

## Features

- **Protocol Compliant**: Uses FastMCP for proper MCP JSON-RPC 2.0 protocol
- **HTTP/SSE Transport**: Accessible via HTTP with streaming support (port 4001)
- **Type-Safe Tools**: Python type hints automatically generate tool schemas
- **LangChain Community Integration**: Uses maintained, feature-rich tools from langchain-community
- **Search & Research Tools**:
  - `echo`: Returns provided text (utility)
  - `add`: Adds two numbers (math demo)
  - `ddg_search`: DuckDuckGo web search (LangChain's DuckDuckGoSearchRun)
  - `arxiv_search`: Search academic papers (LangChain's ArxivQueryRun)
  - `wikipedia_search`: Wikipedia articles (LangChain's WikipediaRetriever)
  - `pubmed_search`: Medical literature (LangChain's PubMedRetriever)
  - `tavily_search`: AI-powered search (LangChain's TavilySearchResults, requires API key)
  - `searxng_search`: Meta-search aggregating multiple engines (LangChain's SearxSearchWrapper + fallback)

- **High Energy Physics (HEP) Tools** (from `hep.py`):
  - `inspirehep_search`: INSPIRE-HEP literature database for HEP papers
  - `cds_search`: CERN Document Server for theses, reports, and publications
  - `arxiv_hep_search`: arXiv search specialized for HEP categories (hep-ph, hep-th, hep-ex, hep-lat)

## Why LangChain Community Tools?

1. **Battle-tested**: Proven tools used by thousands of developers
2. **Consistent Interface**: Uniform error handling and result formats
3. **Maintained**: Regular updates and bug fixes from the community
4. **Feature-rich**: Advanced options and edge case handling built-in

## Running the Server

### Via Docker (recommended)

```bash
# Build and run via docker-compose
docker compose up -d basic_tools_mcp_service

# Check logs
docker compose logs -f basic_tools_mcp_service

# Server will be available at http://localhost:4001
```

### Local Development

```bash
cd backend/basic_tools_mcp_service
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Server runs at http://localhost:4001

## API Endpoints

FastMCP with SSE transport provides:
- `GET /sse` - SSE endpoint for MCP protocol communication
- `GET /health` - Health check endpoint
- `GET /.well-known/mcp` - Discovery endpoint (returns server metadata as JSON)
- `GET /mcp` - Alternative discovery endpoint
- `GET /` - Root discovery endpoint

## Server Metadata & Context

The MCP server provides rich metadata and context through MCP resources and prompts:

### Server Information
- **Prompt**: `server_info` - Overview of server capabilities and use cases
- **Resource**: `server://metadata` - Complete server metadata with tool categories

### Contextual Guidance
- **Resource**: `context://general-search` - Guide for web search tools (DDG, SearXNG, Wikipedia, Tavily)
- **Resource**: `context://academic-research` - Guide for academic tools (arXiv, PubMed, INSPIRE, CDS)
- **Resource**: `context://hep-research` - Specialized guidance for High Energy Physics research

### Accessing Metadata via MCP Client

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client("http://localhost:4001/sse") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # Get server metadata
        resources = await session.list_resources()
        metadata = await session.read_resource("server://metadata")
        print(metadata.contents)
        
        # Get contextual guidance
        search_guide = await session.read_resource("context://general-search")
        hep_guide = await session.read_resource("context://hep-research")
```

### Admin UI Discovery

The admin UI can automatically discover this MCP server:

**Steps**:
1. Navigate to: Admin UI → MCPs → Add MCP
2. Enter endpoint: `http://basic_tools_mcp_service:4001` (or `http://localhost:4001` for local)
3. Click "Discover" button
4. Server metadata auto-fills the form

**Auto-filled fields**:
- **Name**: "basic-tools-mcp"
- **Description**: "Comprehensive search and research toolkit providing access to multiple information sources..."
- **Resource**: "https://github.com/InnovateAILab/AgentKS"
- **Context**: "Designed for AgentKS knowledge stack, integrating multiple search sources..."
- **Tags**: search, research, academic, hep, langchain

**Discovery endpoint**: The server exposes `/.well-known/mcp` which returns JSON metadata including:
- Server name, version, description, context, and resource links
- Complete list of 11 tools with categories
- Available MCP resources and prompts
- Configuration requirements

**Example response**:
```json
{
  "name": "basic-tools-mcp",
  "version": "1.0.0",
  "description": "Comprehensive search and research toolkit...",
  "context": "Designed for AgentKS knowledge stack...",
  "resource": "https://github.com/InnovateAILab/AgentKS",
  "tags": ["search", "research", "academic", "hep"],
  "capabilities": {
    "tools": 11,
    "categories": ["general-search", "academic-research", "hep-literature"]
  }
}
```

## Using with MCP Clients

### Python Client Example

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client("http://localhost:4001/sse") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # List tools
        tools = await session.list_tools()
        
        # Call a tool
        result = await session.call_tool("add", {"a": 5, "b": 3})
        print(result)
```

### Backend Integration

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "basic-tools": {
        "transport": "sse",
        "url": "http://basic_tools_mcp_service:4001/sse"
    }
})

tools = await client.get_tools()
result = await tools[0].ainvoke({"a": 5, "b": 3})
```

### Admin UI Discovery

The admin UI can discover this MCP server:
1. Enter endpoint: `http://basic_tools_mcp_service:4001` (or `http://localhost:4001` for local)
2. Click "Discover"
3. Server metadata will be auto-filled

## Configuration

### Environment Variables

- `TAVILY_API_KEY` - Required for Tavily AI-powered search
- `SEARXNG_URL` - SearXNG instance URL (default: `http://searxng:8080`)
- `INSPIRE_BASE_URL` - INSPIRE-HEP base URL (default: `https://inspirehep.net`)
- `CDS_BASE_URL` - CERN Document Server base URL (default: `https://cds.cern.ch`)
- `ARXIV_API_URL` - arXiv API URL (default: `http://export.arxiv.org/api/query`)

### SearXNG Integration

The `searxng_search` tool connects to a SearXNG meta-search engine instance:
- **Aggregated Results**: Gets results from multiple search engines (Google, Bing, DuckDuckGo, etc.)
- **Privacy-focused**: SearXNG doesn't track users
- **Configurable Categories**: Search in specific categories (general, science, news, images, etc.)
- **Fallback Strategy**: Uses LangChain wrapper when available, falls back to direct API calls

Example usage:
```python
# General web search
result = await session.call_tool("searxng_search", {
    "query": "latest AI research",
    "max_results": 5
})

# Science-focused search
result = await session.call_tool("searxng_search", {
    "query": "quantum computing",
    "max_results": 3,
    "categories": "science"
})
```

### High Energy Physics Tools

The HEP tools (defined in `hep.py`) provide specialized access to physics literature:

#### INSPIRE-HEP Search
```python
# Search for recent papers on Higgs boson
result = await session.call_tool("inspirehep_search", {
    "query": "Higgs boson",
    "max_results": 10,
    "sort": "mostrecent"  # or "mostcited"
})

# Returns: titles, authors, citation counts, journal info, arXiv IDs
```

#### CERN Document Server (CDS)
```python
# Search CDS for LHC reports
result = await session.call_tool("cds_search", {
    "query": "LHC luminosity",
    "max_results": 5
})

# Returns: record IDs, titles, authors, abstracts, creation dates
```

#### arXiv HEP Search
```python
# Search arXiv with HEP category filter
result = await session.call_tool("arxiv_hep_search", {
    "query": "supersymmetry",
    "max_results": 10,
    "category": "hep-ph",  # hep-ph, hep-th, hep-ex, hep-lat
    "sort_by": "submittedDate",
    "sort_order": "descending"
})

# Returns: arXiv IDs, titles, authors, abstracts, PDF URLs, categories
```

## Adding New Tools

### In main.py

Simply add decorated functions:

```python
@mcp.tool()
def my_tool(param1: str, param2: int = 10) -> dict:
    """
    Tool description here.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Result dictionary
    """
    return {"result": f"Processed {param1} with {param2}"}
```

### In Separate Module (like hep.py)

For domain-specific tools, create a module with a registration function:

```python
# my_tools.py
def register_my_tools(mcp):
    @mcp.tool()
    def specialized_tool(query: str) -> dict:
        """Tool description."""
        return {"result": "..."}
```

Then import and register in `main.py`:
```python
from my_tools import register_my_tools
register_my_tools(mcp)
```

This modular approach keeps related tools organized and `main.py` clean.

FastMCP automatically:
- Generates JSON schema from type hints
- Registers the tool
- Handles invocation and validation

## Protocol Details

- **Transport**: SSE (Server-Sent Events) over HTTP
- **Format**: MCP JSON-RPC 2.0
- **Port**: 4001
- **Streaming**: Supported via SSE
- **Compatible with**: MCP clients, web browsers, standard HTTP tools
