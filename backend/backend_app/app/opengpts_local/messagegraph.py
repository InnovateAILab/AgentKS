"""Simple messagegraph executor for tool call plans.

This module executes a list of planned tool calls (provided as a list of
dicts: {"tool": <name>, "args": {...}}) using the project's existing
tool implementations (DB-registered tools and MCP-backed tools). The
functionality mirrors the execution step of a messagegraph workflow where
the planner (LLM) produces a sequence of tool invocations and the runtime
executes them and returns results to be fed back into the conversation.

This is intentionally small and synchronous.
"""

from typing import List, Dict, Any

import common
from mcp.mcp import run_mcp_tool_async
from tools.tools import run_db_tool


def execute_tool_calls(
    user_id: str, calls: List[Dict[str, Any]], used_urls: Dict[str, int] = None
) -> List[Dict[str, Any]]:
    """Execute a sequence of tool calls and return their results.

    Args:
        user_id: caller identity forwarded to tool runs
        calls: list of {"tool": <name>, "args": {...}}
        used_urls: optional dict to track referenced URLs across calls

    Returns:
        list of dicts: [{"tool": name, "status": "ok"|"error", "result": <any>}]
    """
    if used_urls is None:
        used_urls = {}

    out = []
    for call in calls:
        name = call.get("tool") or call.get("name")
        args = call.get("args") or call.get("input") or {}
        if not name:
            out.append({"tool": None, "status": "error", "error": "missing tool name"})
            continue

        # Lookup tool by name in DB
        row = common.tool_get_by_name(name)
        if not row:
            out.append({"tool": name, "status": "error", "error": "tool not found"})
            continue

        try:
            # If it's an MCP-provided tool, the provider field will indicate that in the row
            provider = row[7] if len(row) > 7 else None
            if provider == "mcp":
                # The DB row includes config with url/headers/tool name; prefer using run_mcp_tool_async
                cfg = row[4] if isinstance(row[4], dict) else {}
                mcp_url = cfg.get("url")
                headers = cfg.get("headers") or {}
                toolname = cfg.get("tool") or row[9]
                res = run_mcp_tool_async(mcp_url, headers, toolname, args)
            else:
                # Regular DB-backed tool; run the generic runner
                res = run_db_tool(row, user_id, args, used_urls)

            out.append({"tool": name, "status": "ok", "result": res})
        except Exception as e:
            out.append({"tool": name, "status": "error", "error": str(e)})

    return out


def make_opened_tool_messages(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert execution results to OpenGPTs-like tool messages to inject into a thread.

    Each result becomes a dict like {"role": "tool", "name": <tool>, "content": <str/result>}.
    """
    msgs = []
    for r in results:
        name = r.get("tool")
        if r.get("status") == "ok":
            content = r.get("result")
            # Ensure content is a string for message insertion
            try:
                cstr = content if isinstance(content, str) else str(content)
            except Exception:
                cstr = "<unserializable result>"
            msgs.append({"role": "tool", "name": name, "content": cstr})
        else:
            msgs.append({"role": "tool", "name": name, "content": f"ERROR: {r.get('error')}"})
    return msgs
