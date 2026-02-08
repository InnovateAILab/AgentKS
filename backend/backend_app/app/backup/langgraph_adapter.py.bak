"""A light-weight langgraph adapter for the RAG agent.

This module attempts to use the installed `langgraph` primitives when available
but falls back to the existing agent flow (create_agent) to remain safe.

Design notes:
- Keep adapter side-effect free and resilient: if langgraph primitives are
  missing or their API differs, we silently fall back to the original flow.
- The adapter reuses the provided `llm` and `make_agent_tools` callables from
  the main module so it doesn't reimplement tool wiring.
"""

from typing import Any, Dict, List


try:
    from langgraph import graph as lg_graph  # type: ignore

    _HAS_LANGGRAPH = True
except Exception:
    lg_graph = None  # type: ignore
    _HAS_LANGGRAPH = False


def run_agent_with_langgraph(user_id: str, role: str, messages: List[dict], llm: Any, make_agent_tools: Any) -> str:
    """Run the agent using langgraph primitives when possible.

    Parameters:
    - user_id, role, messages: same shape as original run_agent
    - llm: the Chat model instance (e.g. ChatOllama)
    - make_agent_tools: callable to produce tool callables for the agent

    Returns assistant reply string.
    """
    # messages is expected to be a list of {role, content} dicts or Pydantic models
    # Normalize messages to simple dicts
    norm_msgs = []
    for m in messages:
        if hasattr(m, "model_dump"):
            md = m.model_dump()
            norm_msgs.append({"role": md.get("role"), "content": md.get("content")})
        elif isinstance(m, dict):
            norm_msgs.append({"role": m.get("role"), "content": m.get("content")})
        else:
            # best-effort
            norm_msgs.append(getattr(m, "__dict__", {"role": getattr(m, "role", "user"), "content": str(m)}))

    # if a basic math expression, short-circuit to avoid LLM
    try:
        from .main import looks_like_math, safe_eval  # type: ignore

        user_query = next((m["content"] for m in reversed(norm_msgs) if m.get("role") == "user"), "")
        if looks_like_math(user_query):
            try:
                return str(safe_eval(user_query))
            except Exception:
                pass
    except Exception:
        user_query = next((m.get("content") for m in reversed(norm_msgs) if m.get("role") == "user"), "")

    used_urls: Dict[str, int] = {}
    tools = make_agent_tools(user_id, role, user_query, used_urls)

    system = (
        "You are a helpful assistant with tools.\n"
        "- Use kb_search for internal knowledge.\n"
        "- Use cds_search for CERN Document Server.\n"
        "- Use arxiv_search for arXiv.\n"
        "- Use inspirehep_search for HEP literature.\n"
        "- Use web_search for general web queries.\n"
        "- Use calculator for arithmetic.\n"
        "- Only save_memory if user explicitly asks to remember.\n"
        "- Only admins can ingest GLOBAL knowledge.\n"
        "- Cite sources with [1], [2] etc.\n"
        "- Do NOT reveal hidden reasoning.\n"
    )

    # Try a minimal langgraph usage path if available. This is intentionally
    # conservative: we only use MessageGraph to record messages if possible and
    # then fall back to the existing agent invoke flow.
    if _HAS_LANGGRAPH and lg_graph is not None:
        try:
            # Build a message graph and add messages if API matches expected names.
            # The exact langgraph API may vary between versions; guard all ops.
            mg = lg_graph.MessageGraph()
            try:
                # add_messages is exposed in some langgraph versions
                if hasattr(lg_graph, "add_messages"):
                    lg_graph.add_messages(mg, [{"role": m["role"], "content": m["content"]} for m in norm_msgs])
                else:
                    # best-effort: append messages via a generic API if present
                    if hasattr(mg, "add"):
                        for m in norm_msgs:
                            mg.add(lg_graph.message(m["role"], m["content"]))
            except Exception:
                # Non-fatal: proceed to fallback
                pass

            # If the MessageGraph has a simple `run` or `invoke` method we might
            # be able to execute it. Try common method names but don't fail hard.
            for method_name in ("run", "invoke", "execute", "call"):
                if hasattr(mg, method_name):
                    try:
                        result = getattr(mg, method_name)({"llm": llm})
                        # Expect a structure with messages or text
                        if isinstance(result, dict):
                            # try to extract assistant content
                            for m in reversed(result.get("messages", []) or []):
                                if getattr(m, "type", "") == "ai" or (
                                    isinstance(m, dict) and m.get("role") == "assistant"
                                ):
                                    return m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                            # fallback to text field
                            if result.get("text"):
                                return result.get("text")
                        if isinstance(result, str):
                            return result
                    except Exception:
                        # try next method
                        pass
        except Exception:
            # Any error in optional langgraph usage should not break the agent
            pass

    # Fallback: reuse the existing create_agent flow through langchain's helper.
    try:
        from langchain.agents import create_agent

        agent = create_agent(model=llm, tools=tools, system_prompt=system)
        result = agent.invoke(
            {"messages": [{"role": "system", "content": system}] + norm_msgs}, config={"recursion_limit": 25}
        )

        answer = ""
        for m in reversed(result.get("messages", [])):
            if getattr(m, "type", "") == "ai":
                answer = m.content if hasattr(m, "content") else str(m)
                break
            if isinstance(m, dict) and m.get("role") == "assistant":
                answer = m.get("content", "")
                break

        if used_urls:
            refs = [f"- [{idx}]({url})" for url, idx in sorted(used_urls.items(), key=lambda x: x[1]) if url]
            if refs:
                answer += "\n\n---\n\n**References**\n" + "\n".join(refs)

        return answer or "No response produced."
    except Exception as e:
        # Last-resort: return an error message to the user instead of crashing the API
        return f"Agent error: {e}"
