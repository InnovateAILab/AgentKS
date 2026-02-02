import ast
import asyncio
import time
from typing import Any, Dict, List, Optional, Literal

from common import Document, llm
from langchain_core.tools import tool
from langchain.agents import create_agent

from common import (
    sha256_text,
    tool_run_log,
    tool_get_by_ids,
    tool_vs,
    TOOL_SELECT_TOPK,
    SEARXNG_URL,
    CDS_BASE_URL,
    ARXIV_API_URL,
    INSPIRE_BASE_URL,
)
from rag import add_documents, rag_search, fetch_url_text
from mcp import run_mcp_tool_async
import uuid
import common as _common
import xml.etree.ElementTree as ET
import json
import requests
from typing import Dict as TypingDict


# Calculator
_ALLOWED = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd}


def safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval").body

    def _eval(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED:
            v = _eval(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED:
            a = _eval(n.left)
            b = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow):
                return a**b
            if isinstance(n.op, ast.Mod):
                return a % b
        raise ValueError("Unsupported expression")

    return float(_eval(node))


def looks_like_math(text: str) -> bool:
    allowed = set("0123456789+-*/().% ^\n\t ")
    t = text.strip()
    return t and all(ch in allowed for ch in t)


@tool
def calculator(expression: str) -> str:
    """Do arithmetic exactly. Input like '12*(3+4)'."""
    return str(safe_eval(expression))


# Auto tool selection (Top-K)
def select_tools_for_query(role: str, query: str, k: int) -> List[str]:
    filt = {"enabled": {"$eq": True}} if role == "admin" else {"enabled": {"$eq": True}, "scope": {"$eq": "global"}}
    docs = tool_vs.similarity_search(query, k=k, filter=filt)
    tool_ids: List[str] = []
    for d in docs:
        tid = d.metadata.get("tool_id")
        if tid and tid not in tool_ids:
            tool_ids.append(tid)
    return tool_ids


# Tool implementations
def _remember_url(used_urls: Dict[str, int], url: Optional[str]) -> None:
    if not url:
        return
    if url not in used_urls:
        used_urls[url] = len(used_urls) + 1


def run_searxng(base_url: str, query: str, used_urls: Dict[str, int]) -> Dict[str, Any]:
    r = requests.get(f"{base_url.rstrip('/')}/search", params={"q": query, "format": "json"}, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = []
    for x in (data.get("results") or [])[:5]:
        url = x.get("url")
        _remember_url(used_urls, url)
        results.append({"title": x.get("title"), "url": url, "snippet": x.get("content")})
    return {"results": results}


def run_cds_search(base_url: str, query: str, rg: int, used_urls: Dict[str, int]) -> Dict[str, Any]:
    params = {
        "p": query,
        "of": "recjson",
        "rg": str(rg),
        "jrec": "1",
        "ot": "recid,creation_date,authors[0],number_of_authors,title,abstract",
    }
    r = requests.get(f"{base_url.rstrip('/')}/search", params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    out = []
    for rec in (data or [])[:rg]:
        recid = rec.get("recid")
        title = None
        t = rec.get("title")
        if isinstance(t, dict):
            title = t.get("title") or t.get("subtitle")
        elif isinstance(t, str):
            title = t
        record_url = f"{base_url.rstrip('/')}/record/{recid}" if recid else None
        _remember_url(used_urls, record_url)
        out.append({"recid": recid, "title": title, "creation_date": rec.get("creation_date"), "url": record_url})
    return {"results": out}


def _xml_text(el: Optional[ET.Element]) -> Optional[str]:
    return None if el is None else (el.text or "").strip()


def run_arxiv_search(
    api_url: str, query: str, max_results: int, sortBy: str, sortOrder: str, used_urls: Dict[str, int]
) -> Dict[str, Any]:
    params = {
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }
    r = requests.get(api_url, params=params, timeout=25)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("a:entry", ns):
        title = _xml_text(entry.find("a:title", ns))
        summary = _xml_text(entry.find("a:summary", ns))
        link_el = entry.find("a:link[@rel='alternate']", ns)
        url = link_el.attrib.get("href") if link_el is not None else None
        _remember_url(used_urls, url)
        results.append(
            {
                "title": title,
                "url": url,
                "summary": (summary[:400] + "...") if summary and len(summary) > 400 else summary,
            }
        )
    return {"results": results}


def run_inspirehep_search(base_url: str, query: str, size: int, used_urls: Dict[str, int]) -> Dict[str, Any]:
    r = requests.get(
        f"{base_url.rstrip('/')}/api/literature",
        params={"q": query, "size": str(size)},
        headers={"Accept": "application/json"},
        timeout=25,
    )
    r.raise_for_status()
    data = r.json()
    hits = ((data or {}).get("hits") or {}).get("hits") or []
    out = []
    for h in hits[:size]:
        meta = h.get("metadata") or {}
        titles = meta.get("titles") or []
        title = titles[0].get("title") if titles and isinstance(titles[0], dict) else None
        links = h.get("links") or {}
        url = links.get("html") or links.get("self")
        _remember_url(used_urls, url)
        out.append({"title": title, "url": url})
    return {"results": out}


def run_db_tool(row: tuple, user_id: str, payload: Dict[str, Any], used_urls: Dict[str, int]) -> Any:
    tool_id, name, desc, typ, config_json, enabled, scope, provider, mcp_server, mcp_tool = row
    config = config_json if isinstance(config_json, dict) else json.loads(config_json or "{}")

    t0 = time.time()
    try:
        if not enabled:
            tool_run_log(tool_id, user_id, payload, None, "skipped", "tool disabled", 0)
            return {"error": "tool disabled"}

        if typ == "searxng_search":
            out = run_searxng(config.get("base_url") or SEARXNG_URL, payload.get("query", ""), used_urls)
        elif typ == "cds_search":
            out = run_cds_search(
                config.get("base_url") or CDS_BASE_URL, payload.get("query", ""), int(config.get("rg") or 5), used_urls
            )
        elif typ == "arxiv_search":
            out = run_arxiv_search(
                config.get("api_url") or ARXIV_API_URL,
                payload.get("query", ""),
                int(config.get("max_results") or 5),
                str(config.get("sortBy") or "submittedDate"),
                str(config.get("sortOrder") or "descending"),
                used_urls,
            )
        elif typ == "inspirehep_search":
            out = run_inspirehep_search(
                config.get("base_url") or INSPIRE_BASE_URL,
                payload.get("query", ""),
                int(config.get("size") or 5),
                used_urls,
            )
        elif typ == "mcp_tool":
            mcp_url = config.get("url")
            headers = config.get("headers") or {}
            toolname = config.get("tool") or mcp_tool
            if not mcp_url or not toolname:
                raise ValueError("mcp_tool missing config.url or tool name")
            out = asyncio.run(run_mcp_tool_async(mcp_url, headers, toolname, payload))
        else:
            out = {"error": f"unknown tool type: {typ}"}

        tool_run_log(tool_id, user_id, payload, out, "ok", None, int((time.time() - t0) * 1000))
        return out
    except Exception as e:
        tool_run_log(tool_id, user_id, payload, None, "error", str(e), int((time.time() - t0) * 1000))
        return {"error": str(e)}


def make_agent_tools(user_id: str, role: str, user_query: str, used_urls: Dict[str, int]):
    @tool
    def kb_search(query: str) -> str:
        """Search KB (global + your private memory). Return numbered evidence blocks."""
        docs = rag_search(user_id, query, k=6)
        if not docs:
            return "No relevant passages found."
        blocks = []
        for d in docs:
            url = d.metadata.get("source_url") or d.metadata.get("source") or "unknown"
            _remember_url(used_urls, url)
            idx = used_urls.get(url, 0)
            scope = d.metadata.get("scope", "unknown")
            blocks.append(f"[{idx}] (scope={scope}) URL={url}\n{d.page_content}")
        return "\n\n---\n\n".join(blocks)

    @tool
    def save_memory(text: str) -> str:
        """Save private memory ONLY when user explicitly asks to remember."""
        n = add_documents(
            "private", user_id, [Document(page_content=text, metadata={"source_url": "memory", "type": "note"})]
        )
        return f"Saved to private memory. chunks={n}"

    @tool
    def ingest_urls(urls: List[str], scope: Literal["private", "global"] = "private") -> str:
        """Fetch URLs and ingest into RAG. Only admins may ingest GLOBAL."""
        from common import upsert_source, mark_source_status

        if len(urls) > 30:
            return "Too many URLs. Max is 30."

        if scope == "global":
            if role != "admin":
                return "Denied: only admin can ingest GLOBAL knowledge."
            target_scope = "global"
            target_user = None
        else:
            target_scope = "private"
            target_user = user_id

        ok, fail = [], []
        for u in urls:
            sid = None
            try:
                sid = upsert_source(target_scope, target_user, u)
                mark_source_status(sid, "fetching")
                txt = fetch_url_text(u)
                csha = sha256_text(txt)
                n = add_documents(
                    target_scope,
                    user_id,
                    [Document(page_content=txt, metadata={"source_url": u, "source_id": sid, "type": "url"})],
                )
                mark_source_status(sid, "ingested", content_sha=csha)
                ok.append(f"{u} (chunks={n})")
            except Exception as e:
                if sid:
                    mark_source_status(sid, "failed", error=str(e))
                fail.append(f"{u} (error={e})")

        out = []
        if ok:
            out.append("Ingested:\n" + "\n".join(ok))
        if fail:
            out.append("Failed:\n" + "\n".join(fail))
        return "\n\n".join(out) if out else "Nothing ingested."

    selected_tool_ids = select_tools_for_query(role, user_query, k=TOOL_SELECT_TOPK)
    rows = tool_get_by_ids(selected_tool_ids)

    db_tools = []
    for row in rows:

        def _factory(r):
            rid, rname, rdesc, *_ = r

            @tool(name=rname, description=rdesc)
            def _t(**kwargs):
                return run_db_tool(r, user_id, kwargs, used_urls)

            return _t

        db_tools.append(_factory(row))

    return [kb_search, calculator, save_memory, ingest_urls] + db_tools


def run_agent(user_id: str, role: str, messages: List[Any]):
    user_query = next((m.content for m in reversed(messages) if m.role == "user"), "")

    if looks_like_math(user_query):
        try:
            return str(safe_eval(user_query))
        except Exception:
            pass

    # If configured to use the OpenGPTs-compatible local path, run a local
    # OpenGPTs-like assistant (this clones the minimal OpenGPTs run semantics
    # instead of delegating to a remote server).
    if getattr(_common, "USE_OPENGPTS", False):
        # Convert incoming messages (OpenGPTs-style) to a simple human-only list
        og_msgs = []
        for m in messages:
            try:
                rolev = getattr(m, "role", None)
                content = getattr(m, "content", None)
            except Exception:
                rolev = None
                content = None
            if rolev == "user" and content:
                og_msgs.append({"content": content, "type": "human"})
        assistant_id = getattr(_common, "OPENGPTS_ASSISTANT_ID", None)
        if not assistant_id:
            raise RuntimeError("OPENGPTS_ASSISTANT_ID is not configured")

        # Local OpenGPTs-like runner: create a lightweight thread object and
        # execute the local agent pipeline to produce a reply.
        def _create_thread(assistant_id: str, user_id: str, name: str = "auto"):
            return {"id": str(uuid.uuid4()), "assistant_id": assistant_id, "user_id": user_id, "name": name}

        def _run_local_assistant(
            assistant_id: str, user_id: str, messages: List[TypingDict[str, str]], timeout: int = 30
        ):
            # Map OpenGPTs message list -> internal messages list
            internal_msgs = []
            for m in messages:
                t = m.get("type")
                c = m.get("content")
                if t == "human" and c:
                    internal_msgs.append(type("M", (), {"role": "user", "content": c}))

            # Duplicate the core agent invocation (same logic as below) so this
            # local OpenGPTs path runs fully in-process without remote calls.
            used_urls: Dict[str, int] = {}
            user_q = next((m.content for m in reversed(internal_msgs) if m.role == "user"), "")
            tools = make_agent_tools(user_id, role, user_q, used_urls)

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

            agent = create_agent(model=llm, tools=tools, system_prompt=system)
            result = agent.invoke(
                {"messages": [{"role": "system", "content": system}] + [m.model_dump() for m in internal_msgs]},
                config={"recursion_limit": 25},
            )

            answer = ""
            for m in reversed(result["messages"]):
                if getattr(m, "type", "") == "ai":
                    answer = m.content
                    break
                if isinstance(m, dict) and m.get("role") == "assistant":
                    answer = m.get("content", "")
                    break

            if used_urls:
                refs = [f"- [{idx}]({url})" for url, idx in sorted(used_urls.items(), key=lambda x: x[1]) if url]
                if refs:
                    answer += "\n\n---\n\n**References**\n" + "\n".join(refs)

            return answer or "No response produced."

        # run
        thread = _create_thread(assistant_id, user_id)
        return _run_local_assistant(assistant_id, user_id, og_msgs)

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

    agent = create_agent(model=llm, tools=tools, system_prompt=system)
    result = agent.invoke(
        {"messages": [{"role": "system", "content": system}] + [m.model_dump() for m in messages]},
        config={"recursion_limit": 25},
    )

    answer = ""
    for m in reversed(result["messages"]):
        if getattr(m, "type", "") == "ai":
            answer = m.content
            break
        if isinstance(m, dict) and m.get("role") == "assistant":
            answer = m.get("content", "")
            break

    if used_urls:
        refs = [f"- [{idx}]({url})" for url, idx in sorted(used_urls.items(), key=lambda x: x[1]) if url]
        if refs:
            answer += "\n\n---\n\n**References**\n" + "\n".join(refs)

    return answer or "No response produced."
