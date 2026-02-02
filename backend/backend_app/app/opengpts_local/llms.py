"""Local OpenGPTs-compatible LLM helpers.

Provides minimal functions to enumerate available LLM backends and return
metadata for the configured default LLM. This mirrors the shape of the
OpenGPTs `llms.py` helpers but stays local and lightweight.
"""

from typing import List, Dict

import common


def list_llms() -> List[Dict[str, object]]:
    """Return a list of available LLM backends (metadata only).

    Each entry contains lightweight metadata similar to what an OpenGPTs
    frontend would expect. This intentionally avoids returning live
    client objects to keep the function safe to call in contexts where
    the LLM runtime is not available.
    """
    # For now we only expose the Ollama-backed model configured in common.
    return [
        {
            "id": "ollama",
            "name": "Ollama (local)",
            "description": "Local Ollama-backed LLM configured via OLLAMA_* env vars.",
            "type": "ollama",
            "base_url": getattr(common, "OLLAMA_BASE_URL", None),
            "default_model": getattr(common, "OLLAMA_CHAT_MODEL", None),
            "embed_model": getattr(common, "OLLAMA_EMBED_MODEL", None),
            "supports_streaming": False,  # set True if your Ollama setup supports token streaming
        }
    ]


def get_default_llm() -> Dict[str, object]:
    """Return metadata for the project's configured default LLM.

    This maps environment-configured values from `common` into a simple
    dict the frontend or other components can consume.
    """
    ll = list_llms()
    # Find the entry that matches the configured default model if possible.
    default_model = getattr(common, "OLLAMA_CHAT_MODEL", None)
    for e in ll:
        if e.get("default_model") == default_model:
            return e
    # Fallback to the first entry
    return ll[0] if ll else {}
