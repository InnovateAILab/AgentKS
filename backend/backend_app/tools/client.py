"""MCP helper utilities."""
from typing import Any, Dict
import asyncio

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception:
    MultiServerMCPClient = None


async def run_mcp_tool_async(mcp_url: str, headers: Dict[str, str], tool_name: str, payload: Dict[str, Any]) -> Any:
    if MultiServerMCPClient is None:
        raise RuntimeError("MCP client not available in this environment")
    client = MultiServerMCPClient({"srv": {"transport": "http", "url": mcp_url, "headers": headers}})
    tools = await client.get_tools()
    target = next((t for t in tools if t.name == tool_name), None)
    if not target:
        raise ValueError(f"MCP tool not found: {tool_name}")
    return await target.ainvoke(payload)
