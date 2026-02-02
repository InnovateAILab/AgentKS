"""Adapter to call a running OpenGPTs HTTP backend."""

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
    if not OPENGPTS_URL:
        raise RuntimeError("OPENGPTS_URL is not configured")
    thread_id = create_thread(assistant_id, user_id)
    if not thread_id:
        raise RuntimeError("failed to create thread")

    run_url = f"{OPENGPTS_URL.rstrip('/')}/runs"
    payload = {"assistant_id": assistant_id, "thread_id": thread_id, "input": {"messages": messages}}
    headers = _auth_headers()
    cookies = _cookies_for_user(user_id) if not OPENGPTS_JWT else None
    r = requests.post(run_url, json=payload, headers=headers, cookies=cookies, timeout=10)
    r.raise_for_status()

    state_url = f"{OPENGPTS_URL.rstrip('/')}/threads/{thread_id}/state"
    end = time.time() + timeout
    last_ai = None
    while time.time() < end:
        r = requests.get(state_url, headers=headers, cookies=cookies, timeout=10)
        if r.status_code != 200:
            time.sleep(0.5)
            continue
        data = r.json()
        vals = data.get("values") or data.get("values", {}).get("messages") or []
        if isinstance(vals, dict):
            vals = vals.get("messages") or []
        for v in vals:
            if v.get("type") == "ai":
                last_ai = v.get("content")
        if last_ai:
            return last_ai
        time.sleep(0.5)

    raise TimeoutError("Timed out waiting for OpenGPTs response")
