from typing import List, Literal
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from common import doc_vs, splitter, Document, MAX_CHARS_PER_DOC, FETCH_TIMEOUT_SECONDS


def add_documents(scope: Literal["global", "private"], user_id: str, docs: List[Document]) -> int:
    chunks = splitter.split_documents(docs)
    for d in chunks:
        d.metadata["scope"] = scope
        d.metadata["user_id"] = user_id if scope == "private" else None
    doc_vs.add_documents(chunks)
    return len(chunks)


def rag_search(user_id: str, query: str, k: int = 6) -> List[Document]:
    filt = {
        "$or": [{"scope": {"$eq": "global"}}, {"$and": [{"scope": {"$eq": "private"}}, {"user_id": {"$eq": user_id}}]}]
    }
    return doc_vs.similarity_search(query, k=k, filter=filt)


def fetch_url_text(url: str) -> str:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs allowed")
    r = requests.get(url, timeout=FETCH_TIMEOUT_SECONDS, headers={"User-Agent": "Mozilla/5.0 (RAGBot)"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join([ln for ln in lines if ln])
    return text[:MAX_CHARS_PER_DOC]
