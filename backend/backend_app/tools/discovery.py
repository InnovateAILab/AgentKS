"""MCP Discovery utilities

Functions for connecting to MCP servers and discovering available tools.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None

logger = logging.getLogger(__name__)


async def discover_mcp_tools_async(mcp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Connect to an MCP server and retrieve available tools.
    
    Args:
        mcp: MCP server dictionary with keys:
            - endpoint: MCP server URL
            - auth: Authentication config (optional)
            
    Returns:
        List of tool dictionaries with keys:
        - name: Tool name
        - description: Tool description
        - inputSchema: Tool input schema (JSON Schema)
    """
    if MultiServerMCPClient is None:
        raise RuntimeError("langchain_mcp_adapters not available")
    
    endpoint = mcp["endpoint"]
    auth = mcp.get("auth")
    
    # Build headers from auth config
    headers = {}
    if auth:
        if isinstance(auth, str):
            try:
                auth = json.loads(auth)
            except Exception:
                pass
        
        if isinstance(auth, dict):
            auth_type = auth.get("type", "")
            auth_token = auth.get("token", "")
            auth_headers = auth.get("headers", {})
            
            if auth_type == "bearer" and auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            elif auth_type == "token" and auth_token:
                headers["Authorization"] = auth_token
            
            if auth_headers:
                headers.update(auth_headers)
    
    # Create MCP client
    server_config = {
        "transport": "http",
        "url": endpoint,
    }
    if headers:
        server_config["headers"] = headers
    
    client = MultiServerMCPClient({"mcp_server": server_config})
    
    try:
        # Get tools from MCP
        tools = await client.get_tools()
        
        # Convert to dictionary format
        tool_list = []
        for tool in tools:
            tool_dict = {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
            }
            
            # Try to extract input schema if available
            if hasattr(tool, "args_schema"):
                try:
                    schema = tool.args_schema.schema() if tool.args_schema else {}
                    tool_dict["inputSchema"] = schema
                except Exception:
                    tool_dict["inputSchema"] = {}
            else:
                tool_dict["inputSchema"] = {}
            
            tool_list.append(tool_dict)
        
        logger.info(f"Discovered {len(tool_list)} tools from MCP {mcp.get('name', 'unknown')}")
        return tool_list
    
    except Exception as e:
        logger.error(f"Failed to discover tools from MCP {mcp.get('name', 'unknown')}: {e}")
        raise
    finally:
        # Clean up client connection if needed
        try:
            await client.aclose()
        except Exception:
            pass


def discover_mcp_tools(mcp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Synchronous wrapper for discover_mcp_tools_async."""
    return asyncio.run(discover_mcp_tools_async(mcp))
