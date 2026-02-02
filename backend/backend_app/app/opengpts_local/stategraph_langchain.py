"""Adapter to run a planned stategraph using LangChain's StateGraph when available.

This module attempts to import `langchain.graph.StateGraph` and, if
available, constructs a runtime graph from a planner-produced node list
and executes it. If LangChain's StateGraph is not installed or the API
differs, it falls back to the local `execute_stategraph` implementation.

This provides a best-effort integration: it will work when the target
environment has a compatible LangChain version. The fallback ensures the
feature remains usable without the dependency.
"""

from typing import List, Dict, Any
import traceback

from .stategraph import execute_stategraph
from .messagegraph import execute_tool_calls
from .planner import plan_stategraph


def plan_and_run_with_langchain(user_id: str, role: str, messages: List[Dict[str, Any]]):
    """Attempt to plan and run a stategraph via LangChain StateGraph.

    Returns the same shape as `execute_stategraph` (dict) on success, or
    falls back to `execute_stategraph` on any error.
    """
    try:
        # Lazy import to avoid hard dependency
        from langchain.graph import StateGraph
    except Exception:
        # LangChain StateGraph not available, fallback
        return _fallback_plan_and_run(user_id, role, messages)

    # Produce a planner stategraph (list of nodes)
    try:
        sg = plan_stategraph(user_id, role, messages)
    except Exception:
        # If planner fails, fallback
        return _fallback_plan_and_run(user_id, role, messages)

    try:
        # Construct a StateGraph instance from nodes. This code assumes a
        # simple API where nodes are added with an id and a callable.
        graph = StateGraph()

        # Map node id -> callable
        for node in sg:
            nid = node.get("id")
            ntype = node.get("type")
            cfg = node.get("config") or {}

            if ntype == "tool":
                # create a callable that executes the single tool and returns result
                def make_tool_call(cfg):
                    def _call(context=None):
                        calls = [{"tool": cfg.get("tool"), "args": cfg.get("args") or {}}]
                        res = execute_tool_calls(user_id, calls)
                        return res

                    return _call

                graph.add_node(nid, make_tool_call(cfg))
            else:
                # For unsupported node types, add a no-op node
                graph.add_node(nid, lambda context=None: {"error": f"unsupported node type {ntype}"})

        # Connect nodes according to 'next' keys if available
        for node in sg:
            nid = node.get("id")
            nxt = node.get("next")
            if not nxt:
                continue
            if isinstance(nxt, list):
                for t in nxt:
                    graph.add_edge(nid, t)
            else:
                graph.add_edge(nid, nxt)

        # Execute the graph starting from the first node id
        start = sg[0].get("id") if sg else None
        if start is None:
            return {"nodes_executed": [], "results": {}, "messages": []}

        # Try common execution APIs
        if hasattr(graph, "run"):
            out = graph.run(start)
        elif hasattr(graph, "execute"):
            out = graph.execute(start)
        else:
            # Unknown API; fallback
            return _fallback_plan_and_run(user_id, role, messages)

        # Normalize output shape
        return {"nodes_executed": [], "results": {"langchain": out}, "messages": []}

    except Exception:
        # On any error, fallback to local executor
        traceback.print_exc()
        return _fallback_plan_and_run(user_id, role, messages)


def _fallback_plan_and_run(user_id: str, role: str, messages: List[Dict[str, Any]]):
    # Use the planner + local execute_stategraph flow
    sg = plan_stategraph(user_id, role, messages)
    return execute_stategraph(user_id, sg)
