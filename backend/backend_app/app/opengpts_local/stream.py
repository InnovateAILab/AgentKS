import json
from typing import List


# Import the existing run_agent so we can reuse the local agent pipeline.
from ..tools.tools import run_agent


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def stream_local_assistant(assistant_id: str, user_id: str, role: str, messages: List[dict], chunk_size: int = 256):
    """Generator that yields SSE-style chunks for a locally-run assistant.

    This clones the minimal streaming behavior: first emit a role 'assistant'
    followed by content chunks and a final stop event. It calls the local
    `run_agent` (which runs your agent+tools pipeline) to produce a final
    answer, then streams that answer in chunks. This avoids network calls.

    Args:
        assistant_id: opaque identifier (not used for behavior, only for parity)
        user_id: user identifier passed to local run
        role: user role (e.g., "user" or "admin")
        messages: list of OpenGPTs-style messages (we'll map human -> user)
        chunk_size: size in bytes/characters for each streamed chunk
    """

    # Map OpenGPTs-style messages (type=human) to internal ChatMessage-like objects
    internal_msgs = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        t = m.get("type")
        c = m.get("content")
        if t == "human" and c:
            # create a lightweight object with role and content attributes
            internal_msgs.append(type("M", (), {"role": "user", "content": c}))

    # Run the local agent to produce a full answer (synchronous)
    answer = run_agent(user_id, role, internal_msgs)

    # SSE-style: first send assistant role event
    cid = f"chatcmpl-{int(__import__('time').time())}"
    header = {
        "id": cid,
        "object": "chat.completion.chunk",
        "model": "rag-agent",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield _sse_event(header)

    # Stream content in chunks
    if not answer:
        body = {
            "id": cid,
            "object": "chat.completion.chunk",
            "model": "rag-agent",
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        }
        yield _sse_event(body)
    else:
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i : i + chunk_size]
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": "rag-agent",
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            yield _sse_event(payload)

        # final stop
        footer = {
            "id": cid,
            "object": "chat.completion.chunk",
            "model": "rag-agent",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield _sse_event(footer)

    # termination marker
    yield "data: [DONE]\n\n"
