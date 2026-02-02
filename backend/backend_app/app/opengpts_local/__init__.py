"""Local OpenGPTs-compatible utilities.

This package provides minimal helpers to mimic OpenGPTs streaming/run
semantics in-process (no network calls)."""

__all__ = [
    "stream_local_assistant",
    "opengpts_tools_from_db",
    "opengpts_tool_by_name",
    "list_llms",
    "get_default_llm",
    "execute_tool_calls",
    "make_opened_tool_messages",
    "execute_stategraph",
]
