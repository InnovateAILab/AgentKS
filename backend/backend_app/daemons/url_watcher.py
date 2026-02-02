"""Daemon: URL watcher and ingester

Checks the `urls` table for rows with status IN ('queued','refresh').
- queued: fetch content and ingest into the RAG store
- refresh: delete existing rag_documents for that url, then fetch & re-ingest

This script intentionally re-uses helper functions from `app.main` to
avoid duplicating ingestion/vectorization logic.

Run as: python -m backend_app.daemons.url_watcher  (when PYTHONPATH includes backend/backend_app)
Or from repo root:
    PYTHONPATH=backend/backend_app python backend/backend_app/daemons/url_watcher.py

Configuration (env):
- SLEEP_SECONDS: how long to sleep between polling (default 5)
- BATCH_SIZE: how many URLs to process per loop (default 10)
"""

import os
import time
import traceback
from typing import List, Tuple, Optional

import psycopg

# Import helpers from the app package. This assumes the script is executed
# with Python path such that `app` is importable (e.g. when run from
# backend/backend_app or with PYTHONPATH set accordingly).
try:
    from app.rag import fetch_url_text, add_documents, sha256_text
    from app.main import mark_source_status, COLLECTION_DOCS
    from langchain_core.documents import Document
except Exception:  # pragma: no cover - import-time fallback
    # If imports fail, surface a helpful error when running the daemon.
    raise

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required to run the URL watcher daemon")

PG_DSN = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
# How long between periodic content checks for already-ingested URLs
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", str(60 * 60)))  # default 1h
# Only check ingested URLs whose last_fetched_at is older than this
STALE_AFTER_SECONDS = int(os.getenv("STALE_AFTER_SECONDS", str(60 * 60 * 6)))  # default 6h


def claim_urls(conn, batch_size: int) -> List[Tuple[str, str, str]]:
    """Select a batch of URLs with status queued/refresh and lock them.

    Returns list of (id, url, status).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, url, status
            FROM urls
            WHERE status IN ('queued','refresh')
            ORDER BY created_at NULLS LAST
            FOR UPDATE SKIP LOCKED
            LIMIT %s
            """,
            (batch_size,),
        )
        rows = cur.fetchall()
        return [(r[0], r[1], r[2]) for r in rows]


def get_latest_rag_content_hash(conn, url_id: str) -> Optional[str]:
    """Return the most recent content_hash for documents associated with url_id.

    Uses rag_documents.content_hash and returns None if no document exists.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content_hash FROM rag_documents WHERE url_id = %s ORDER BY updated_at DESC NULLS LAST LIMIT 1",
            (url_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def schedule_refresh_if_changed(conn, url_id: str, url: str) -> bool:
    """Fetch URL, compute sha, compare to latest rag content hash, and mark for refresh if changed.

    Returns True if a change was detected and the url was marked for refresh.
    """
    try:
        text = fetch_url_text(url)
    except Exception:
        # On fetch error, don't change status here; let other workers retry based on last_fetched_at
        return False
    csha = sha256_text(text)
    latest = get_latest_rag_content_hash(conn, url_id)
    if latest != csha:
        with conn.cursor() as cur:
            cur.execute("UPDATE urls SET status = 'refresh' WHERE id = %s", (url_id,))
        conn.commit()
        print(f"Change detected for {url} (id={url_id}), scheduled refresh")
        return True
    return False


def claim_ingested_urls_for_check(conn, batch_size: int) -> List[Tuple[str, str, Optional[str]]]:
    """Select a batch of ingested URLs that haven't been checked recently.

    Returns list of (id, url, last_fetched_at).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, url, last_fetched_at
            FROM urls
            WHERE status = 'ingested' AND (last_fetched_at IS NULL OR last_fetched_at < now() - (%s || ' seconds')::interval)
            ORDER BY last_fetched_at NULLS FIRST
            LIMIT %s
            """,
            (STALE_AFTER_SECONDS, batch_size),
        )
        rows = cur.fetchall()
        return [(r[0], r[1], r[2]) for r in rows]


def delete_rag_documents(conn, url_id: str) -> int:
    """Remove rag_documents rows for the given url_id and adjust rag_groups.doc_count.

    Returns number of deleted rows.
    """
    # Count per rag_group
    with conn.cursor() as cur:
        cur.execute(
            "SELECT rag_group_id, COUNT(*) FROM rag_documents WHERE url_id = %s GROUP BY rag_group_id",
            (url_id,),
        )
        counts = cur.fetchall()
        # Delete docs
        cur.execute("DELETE FROM rag_documents WHERE url_id = %s RETURNING id", (url_id,))
        deleted = cur.rowcount if cur.rowcount is not None else 0
        # Update rag_groups.doc_count
        for rgid, cnt in counts:
            cur.execute(
                "UPDATE rag_groups SET doc_count = GREATEST(coalesce(doc_count,0) - %s, 0) WHERE id = %s",
                (cnt, rgid),
            )
    return deleted


def delete_vectors_for_source(conn, source_id: str, collection_table: str) -> int:
    """Delete vectors/documents from the PGVector collection table where
    metadata->>'source_id' matches the given source_id.

    Returns number of rows deleted (best-effort; may be None depending on cursor).
    """
    # Prefer using the PGVector API (doc_vs) if available.
    try:
        from app.main import doc_vs  # type: ignore

        if doc_vs is not None:
            # Try a few common deletion APIs on vector stores. These may vary by
            # implementation; try a safe series of attempts and return if any
            # succeed.
            for method_name in ("delete_documents", "delete", "delete_by_filter", "delete_documents_by_filter"):
                method = getattr(doc_vs, method_name, None)
                if not method:
                    continue
                try:
                    # Prefer a filter-based call when possible
                    try:
                        res = method(filter={"metadata": {"source_id": source_id}})
                    except TypeError:
                        # Fallback to a looser calling convention
                        res = method({"source_id": source_id})

                    # Interpret result: if numeric, return it; if iterable, return len(); else 0
                    if isinstance(res, int):
                        return res
                    if hasattr(res, "__len__"):
                        return len(res)
                    return 0
                except Exception:
                    # try next method
                    continue
    except Exception:
        # non-fatal; fall back to SQL deletion below
        pass

    # Fallback: raw SQL deletion against the PGVector collection table
    with conn.cursor() as cur:
        # Use jsonb ->> operator to compare text field
        cur.execute(
            f"DELETE FROM {collection_table} WHERE (metadata->>%s) = %s RETURNING id",
            ("source_id", source_id),
        )
        # rowcount may be None for some drivers; use returned rows length when available
        try:
            rows = cur.fetchall()
            return len(rows)
        except Exception:
            return cur.rowcount or 0


def process_url(conn, url_id: str, url: str, status: str):
    """Fetch and ingest a single URL according to its status.

    - If status == 'refresh': delete existing rag_documents for url_id first.
    - Fetch content, create Document, call add_documents to insert into vector store.
    - Update urls.status and last_fetched_at/last_error via mark_source_status.
    """
    try:
        if status == "refresh":
            deleted = delete_rag_documents(conn, url_id)
            # commit the deletion before ingesting new content (so other workers see it)
            conn.commit()
            # Also delete vectors/documents from the vector collection that reference this source
            try:
                deleted_vectors = delete_vectors_for_source(conn, url_id, COLLECTION_DOCS)
                conn.commit()
                print(f"Removed {deleted_vectors} vectors for source {url_id}")
            except Exception as e:
                # Don't fail the whole job if vector deletion fails; log and continue
                print(f"Warning: failed to delete vectors for source {url_id}: {e}")
        # Fetch text
        text = fetch_url_text(url)
        csha = sha256_text(text)
        doc = Document(page_content=text, metadata={"source_url": url, "source_id": url_id, "type": "url"})
        # Use 'global' scope by default; ingestion function will accept private/global.
        # We don't have a per-url owner column in the normalized `urls` table, so
        # private ingestion is not supported here unless you extend the schema.
        n = add_documents("global", None, [doc])
        # Update status to 'ingested'
        mark_source_status(url_id, "ingested", error=None, content_sha=csha)
        print(f"Processed URL {url} -> chunks={n}")
    except Exception as e:
        # Mark failed and record the error
        try:
            mark_source_status(url_id, "failed", error=str(e))
        except Exception:
            # swallow secondary errors
            pass
        print(f"Error processing {url}: {e}\n{traceback.format_exc()}")


def main_loop():
    print("Starting URL watcher daemon")
    with psycopg.connect(PG_DSN) as conn:
        while True:
            try:
                # First, claim any queued/refresh work and process it
                rows = claim_urls(conn, BATCH_SIZE)
                if not rows:
                    # If there's no immediate work, also look for ingested URLs to re-check
                    check_rows = claim_ingested_urls_for_check(conn, BATCH_SIZE)
                    if check_rows:
                        for url_id, url, _ in check_rows:
                            try:
                                schedule_refresh_if_changed(conn, url_id, url)
                            except Exception:
                                conn.rollback()
                                continue
                        # short sleep after checks
                        time.sleep(0.5)
                        continue
                    time.sleep(SLEEP_SECONDS)
                    continue

                for url_id, url, status in rows:
                    # process each url in its own transaction to avoid holding locks
                    try:
                        process_url(conn, url_id, url, status)
                    except Exception:
                        # ensure we don't leave the transaction open
                        conn.rollback()
                # small sleep to yield
                time.sleep(0.1)
            except Exception as e:
                print(f"Daemon error: {e}\n{traceback.format_exc()}")
                time.sleep(5)


if __name__ == "__main__":
    main_loop()
