"""RAG helper functions: ingestion, retrieval, and URL fetching.

This module intentionally performs imports of runtime objects (like `doc_vs`)
inside functions to avoid circular imports with `app.main`. Callers can import
these helpers directly from `app.rag`.
"""
from typing import List, Optional, Literal
import hashlib
import os
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Config defaults (can be overridden by env in app.main)
FETCH_TIMEOUT_SECONDS = int(os.getenv("FETCH_TIMEOUT_SECONDS", "25"))
MAX_CHARS_PER_DOC = int(os.getenv("MAX_CHARS_PER_DOC", "250000"))

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def add_documents(scope: Literal["global", "private"], user_id: Optional[str], docs: List[Document]) -> int:
    """Split documents and add them to the vector store (doc_vs).

    This function imports `doc_vs` lazily from `app.main` to avoid
    circular import issues.
    """
    # Lazy import to avoid circular dependency on app.main
    from app.main import doc_vs  # type: ignore

    chunks = splitter.split_documents(docs)
    for d in chunks:
        d.metadata["scope"] = scope
        d.metadata["user_id"] = user_id if scope == "private" else None
    doc_vs.add_documents(chunks)
    return len(chunks)


def rag_search(user_id: str, query: str, k: int = 6) -> List[Document]:
    """Perform similarity search against the RAG store.

    Uses `doc_vs.similarity_search` with a filter that returns global docs
    and private docs owned by the given user.
    """
    from app.main import doc_vs  # type: ignore

    filt = {
        "$or": [
            {"scope": {"$eq": "global"}},
            {"$and": [{"scope": {"$eq": "private"}}, {"user_id": {"$eq": user_id}}]},
        ]
    }
    return doc_vs.similarity_search(query, k=k, filter=filt)


def fetch_url_text(url: str) -> str:
    """Fetch a URL and return cleaned plaintext.

    Raises ValueError for non-http(s) URLs. Truncates content to
    MAX_CHARS_PER_DOC.
    """
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
