import asyncio
from typing import Any, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient


async def _get_client_tools(mcp_url: str, headers: Dict[str, str]):
    client = MultiServerMCPClient({"srv": {"transport": "http", "url": mcp_url, "headers": headers}})
    return await client.get_tools()


def get_mcp_tools(mcp_url: str, headers: Dict[str, str]):
    return asyncio.run(_get_client_tools(mcp_url, headers))


async def run_mcp_tool_async(mcp_url: str, headers: Dict[str, str], tool_name: str, payload: Dict[str, Any]) -> Any:
    client = MultiServerMCPClient({"srv": {"transport": "http", "url": mcp_url, "headers": headers}})
    tools = await client.get_tools()
    target = next((t for t in tools if t.name == tool_name), None)
    if not target:
        raise ValueError(f"MCP tool not found: {tool_name}")
    return await target.ainvoke(payload)
