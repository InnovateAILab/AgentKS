"""MCP-related Pydantic models"""
from typing import Dict, Any, Literal
from pydantic import BaseModel


class MCPSyncRequest(BaseModel):
    """Request model for syncing tools from an MCP server"""
    server_name: str
    url: str
    headers: Dict[str, Any] = {}
    scope: Literal["global", "admin_only"] = "global"
    enabled: bool = True


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server"""
    id: str
    name: str
    endpoint: str
    kind: str = "http"
    description: str = ""
    context: str = ""
    resource: str = ""
    auth: Dict[str, Any] | None = None
    tags: list[str] = []
    status: Literal["pending", "processing", "enabled", "disabled", "error"] = "pending"
