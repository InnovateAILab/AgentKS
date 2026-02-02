"""Simple adapter to call a running OpenGPTs HTTP backend.

This adapter implements a minimal flow:
- Create a thread for the configured assistant
- Post messages (user messages) as the input
- Run the assistant on that thread
- Poll the thread state and return the AI response

Authentication: by default this uses a cookie `opengpts_user_id` to identify the
user (this mirrors the README examples). If `OPENGPTS_JWT` is set in `common`,
the adapter will use it as a Bearer token in the Authorization header instead.
"""

import time
from typing import List, Dict, Any

import requests

from common import OPENGPTS_URL, OPENGPTS_JWT


def _cookies_for_user(user_id: str) -> Dict[str, str]:
    return {"opengpts_user_id": user_id}


def _auth_headers() -> Dict[str, str]:
    if OPENGPTS_JWT:
        return {"Authorization": f"Bearer {OPENGPTS_JWT}"}
    return {}


def create_thread(assistant_id: str, user_id: str, name: str = "auto") -> str:
    if not OPENGPTS_URL:
        raise RuntimeError("OPENGPTS_URL is not configured")
    url = f"{OPENGPTS_URL.rstrip('/')}/threads"
    payload = {"name": name, "assistant_id": assistant_id}
    headers = _auth_headers()
    cookies = _cookies_for_user(user_id) if not OPENGPTS_JWT else None
    r = requests.post(url, json=payload, headers=headers, cookies=cookies, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("thread_id") or data.get("threadId") or data.get("id")


def run_assistant(assistant_id: str, user_id: str, messages: List[Dict[str, Any]], timeout: int = 30) -> str:
    """Run assistant with the given messages and return the AI content.

    messages should be a list of dicts shaped like {"content": str, "type": "human"}
    (we will construct that from the app messages).
    """
    if not OPENGPTS_URL:
        raise RuntimeError("OPENGPTS_URL is not configured")
    thread_id = create_thread(assistant_id, user_id)
    if not thread_id:
        raise RuntimeError("failed to create thread")

    # POST /runs with the input messages
    run_url = f"{OPENGPTS_URL.rstrip('/')}/runs"
    payload = {"assistant_id": assistant_id, "thread_id": thread_id, "input": {"messages": messages}}
    headers = _auth_headers()
    cookies = _cookies_for_user(user_id) if not OPENGPTS_JWT else None
    r = requests.post(run_url, json=payload, headers=headers, cookies=cookies, timeout=10)
    r.raise_for_status()

    # Poll the thread state until we see an AI message or timeout
    state_url = f"{OPENGPTS_URL.rstrip('/')}/threads/{thread_id}/state"
    end = time.time() + timeout
    last_ai = None
    while time.time() < end:
        r = requests.get(state_url, headers=headers, cookies=cookies, timeout=10)
        if r.status_code != 200:
            time.sleep(0.5)
            continue
        data = r.json()
        # API returns {"values": [...]} where ai messages have type 'ai'
        vals = data.get("values") or data.get("values", {}).get("messages") or []
        # normalize to a list of messages
        if isinstance(vals, dict):
            vals = vals.get("messages") or []
        for v in vals:
            if v.get("type") == "ai":
                last_ai = v.get("content")
        if last_ai:
            return last_ai
        time.sleep(0.5)

    raise TimeoutError("Timed out waiting for OpenGPTs response")
