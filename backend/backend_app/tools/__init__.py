"""
MCP (Model Context Protocol) Module

This module contains utilities and functions for working with MCP servers:
- client: MCP client utilities for connecting to and invoking MCP tools
- discovery: Functions for discovering tools from MCP servers
- models: Pydantic models for MCP-related data structures
- watcher: Daemon for monitoring and auto-discovering MCP tools
- tool_discovery: Tool discovery and semantic search utilities
"""

from .client import run_mcp_tool_async
from .discovery import discover_mcp_tools, discover_mcp_tools_async
from .models import MCPSyncRequest, MCPServerConfig
from .tool_discovery import (
    discover_tools,
    discover_tools_hybrid,
    bind_discovered_tools_to_llm,
    index_tool_with_mcp_context,
    index_tool_simple,
    reindex_all_tools,
)

__all__ = [
    "run_mcp_tool_async",
    "discover_mcp_tools",
    "discover_mcp_tools_async",
    "MCPSyncRequest",
    "MCPServerConfig",
    "discover_tools",
    "discover_tools_hybrid",
    "bind_discovered_tools_to_llm",
    "index_tool_with_mcp_context",
    "index_tool_simple",
    "reindex_all_tools",
]
