"""Planner -> Stategraph -> Synthesizer example helper.

This module provides a small example flow that:
  1. Calls the LLM to produce a stategraph plan (JSON) given conversation messages.
  2. Executes the stategraph via `execute_stategraph`.
  3. Calls the LLM again to synthesize a final assistant reply using tool outputs.

The functions are intentionally simple examples to illustrate how to wire
the planner and executor together. You can adapt the prompts and parsing
rules to your preferred planner format.
"""

import json
import re
from typing import List, Dict, Any

from langchain.agents import create_agent

import common
from .stategraph import execute_stategraph


_PLANNER_SYSTEM = (
    "You are a planning assistant that outputs a JSON stategraph describing a sequence of actions.\n"
    "The output must be valid JSON (a list of nodes). Each node is an object with: id, type ('tool'|'condition'), config, next (optional).\n"
    "Example:\n"
    "[{\n"
    '  "id": "n1",\n'
    '  "type": "tool",\n'
    '  "config": {"tool": "web_search", "args": {"query": "Higgs boson"}},\n'
    '  "next": "n2"\n'
    "}]\n"
)

_SYNTH_SYSTEM = "You are a helpful assistant. Synthesize a final reply for the user using the original conversation and the tool outputs provided.\n"


def _extract_json(s: str) -> Any:
    # Try to find the first JSON array/object in the text
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", s)
    if not m:
        raise ValueError("No JSON object/array found in planner output")
    text = m.group(1)
    return json.loads(text)


def plan_stategraph(
    user_id: str, role: str, messages: List[Dict[str, Any]], planner_prompt: str | None = None
) -> List[Dict[str, Any]]:
    """Call the LLM to produce a JSON stategraph given messages.

    messages: list of dicts with keys {role, content} or similar. We will
    present them to the model and ask for a JSON list of nodes.
    """
    prompt_system = _PLANNER_SYSTEM
    if planner_prompt:
        prompt_system += "\n" + planner_prompt

    # Build messages for the agent.invoke call
    # messages may be Pydantic objects or dicts; normalize to dicts
    normalized = []
    for m in messages:
        if isinstance(m, dict):
            normalized.append(m)
        else:
            # lightweight normalization
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "")
            normalized.append({"role": role, "content": content})

    agent = create_agent(model=common.llm, tools=[], system_prompt=prompt_system)
    result = agent.invoke({"messages": [{"role": "system", "content": prompt_system}] + normalized}, config={})

    # extract assistant content from result
    out_text = None
    for m in reversed(result.get("messages", [])):
        if getattr(m, "type", "") == "ai":
            out_text = m.content
            break
        if isinstance(m, dict) and m.get("role") == "assistant":
            out_text = m.get("content")
            break

    if not out_text:
        raise RuntimeError("Planner did not produce any assistant output")

    sg = _extract_json(out_text)
    if not isinstance(sg, list):
        raise ValueError("Planner must return a JSON list of nodes")
    return sg


def synthesize_reply(
    user_id: str, role: str, original_messages: List[Dict[str, Any]], tool_messages: List[Dict[str, Any]]
) -> str:
    """Call LLM to synthesize final assistant reply from original messages + tool outputs."""
    normalized = []
    for m in original_messages:
        if isinstance(m, dict):
            normalized.append(m)
        else:
            normalized.append({"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")})

    # append tool_messages (they are dicts with role:name:content)
    for tm in tool_messages:
        normalized.append({"role": tm.get("role", "tool"), "content": f"[{tm.get('name')}] {tm.get('content')}"})

    agent = create_agent(model=common.llm, tools=[], system_prompt=_SYNTH_SYSTEM)
    result = agent.invoke({"messages": [{"role": "system", "content": _SYNTH_SYSTEM}] + normalized}, config={})

    out_text = ""
    for m in reversed(result.get("messages", [])):
        if getattr(m, "type", "") == "ai":
            out_text = m.content
            break
        if isinstance(m, dict) and m.get("role") == "assistant":
            out_text = m.get("content", "")
            break

    return out_text or "No response produced."


def planner_stategraph_flow(user_id: str, role: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """End-to-end example: produce plan, execute, synthesize reply.

    Returns a dict:
      {"stategraph": <list>, "execution": <execute_stategraph result>, "reply": <str>}
    """
    sg = plan_stategraph(user_id, role, messages)
    exec_res = execute_stategraph(user_id, sg)
    final = synthesize_reply(user_id, role, messages, exec_res.get("messages", []))
    return {"stategraph": sg, "execution": exec_res, "reply": final}
