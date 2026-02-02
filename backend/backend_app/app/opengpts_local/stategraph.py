"""Simple stategraph executor.

This module implements a minimal stategraph execution engine inspired by
OpenGPTs' `retrieval.py` state-based workflows. It's intentionally small
and deterministic: a stategraph is a list of ordered nodes. Each node is
a dict with at least:

  - id: unique node id
  - type: "tool" | "llm" | "condition" (currently supports "tool" primarily)
  - config: dict with node-specific config, e.g. {"tool": "web_search", "args": {...}}
  - next: optional id of next node (or list for branching)

The executor walks nodes in order (starting from the first node in the
list) and executes tool nodes using the project's existing tool runners
(via opengpts_local.messagegraph.execute_tool_calls). Results are stored
in a `state` dict that later nodes can inspect.

This is not a full messagegraph/stategraph implementation, but a
lightweight building block you can extend.
"""

from typing import List, Dict, Any, Optional

from .messagegraph import execute_tool_calls, make_opened_tool_messages


def _find_node(nodes: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    for n in nodes:
        if n.get("id") == node_id:
            return n
    return None


def execute_stategraph(user_id: str, nodes: List[Dict[str, Any]], used_urls: Dict[str, int] = None) -> Dict[str, Any]:
    """Execute a simple ordered stategraph and return state/results.

    Args:
        user_id: identity used for tool calls
        nodes: list of node dicts (see module docstring)
        used_urls: optional dict to track referenced URLs

    Returns:
        dict with keys: nodes_executed (list of node ids), results (map id->result), messages (tool messages)
    """
    if used_urls is None:
        used_urls = {}

    results: Dict[str, Any] = {}
    executed: List[str] = []
    tool_messages: List[Dict[str, Any]] = []

    # Simple linear traversal from first node; respect explicit next if provided.
    if not nodes:
        return {"nodes_executed": executed, "results": results, "messages": tool_messages}

    current = nodes[0]
    while current:
        nid = current.get("id")
        ntype = current.get("type")
        cfg = current.get("config") or {}
        executed.append(nid)

        if ntype == "tool":
            # allow single tool or list of tools
            if isinstance(cfg.get("tool"), list):
                calls = []
                for t in cfg.get("tool"):
                    calls.append({"tool": t, "args": cfg.get("args") or {}})
            else:
                calls = [{"tool": cfg.get("tool"), "args": cfg.get("args") or {}}]

            call_results = execute_tool_calls(user_id, calls, used_urls)
            results[nid] = call_results
            # convert to tool messages for injection/parity
            tool_messages.extend(make_opened_tool_messages(call_results))

        elif ntype == "condition":
            # simple check: evaluate a path based on previous node result
            # cfg should contain: {"on": <node_id>, "check": {"contains": "term"}, "true_next": id, "false_next": id}
            on = cfg.get("on")
            check = cfg.get("check") or {}
            val = None
            if on and on in results:
                val = results.get(on)
            take_true = False
            if "contains" in check and val is not None:
                try:
                    s = str(val)
                    take_true = check["contains"] in s
                except Exception:
                    take_true = False

            # record condition outcome
            results[nid] = {"condition": take_true}

            # set next accordingly
            nxt = cfg.get("true_next") if take_true else cfg.get("false_next")
            if nxt:
                target = _find_node(nodes, nxt)
                current = target
                continue

        else:
            # unknown node type: store a placeholder
            results[nid] = {"error": f"unsupported node type: {ntype}"}

        # determine next node
        nxt = current.get("next")
        if not nxt:
            break
        # nxt may be single id or list; pick first for linear executor
        if isinstance(nxt, list):
            nxt_id = nxt[0]
        else:
            nxt_id = nxt
        current = _find_node(nodes, nxt_id)

    return {"nodes_executed": executed, "results": results, "messages": tool_messages}
