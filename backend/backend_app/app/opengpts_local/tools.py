"""Helpers to expose the project's tools in an OpenGPTs-friendly shape.

These functions read tools from the existing DB helpers in `common` and
return plain dictionaries that are easy to feed into a local assistant
creation or to mimic the OpenGPTs 'tools' schema.
"""

import json
from typing import List, Dict, Any, Optional

import common


def _row_to_tool_dict(row: tuple) -> Dict[str, Any]:
    # row shape: id,name,description,type,config,enabled,scope,provider,mcp_server,mcp_tool
    tid, name, description, typ, config_json, enabled, scope, provider, mcp_server, mcp_tool = row
    try:
        config = config_json if isinstance(config_json, dict) else json.loads(config_json or "{}")
    except Exception:
        config = {}
    return {
        "id": tid,
        "name": name,
        "description": description,
        "type": typ,
        "config": config,
        "enabled": bool(enabled),
        "scope": scope,
        "provider": provider,
        "mcp_server": mcp_server,
        "mcp_tool": mcp_tool,
    }


def opengpts_tools_from_db(role: str) -> List[Dict[str, Any]]:
    """Return tools visible to `role` formatted for OpenGPTs-style consumption.

    The returned list contains dictionaries with basic metadata and the
    parsed `config` object. This is intentionally minimal; if you need a
    richer OpenGPTs 'parameters' JSON schema for each tool, we can extend
    `tool_upsert` or store a schema in `config`.
    """
    rows = common.tool_list_for_role(role)
    return [_row_to_tool_dict(r) for r in rows]


def opengpts_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Return a single tool dict by name, or None if not found."""
    row = common.tool_get_by_name(name)
    if not row:
        return None
    return _row_to_tool_dict(row)
