"""
URL Watcher Daemon - RAG Integration

This daemon monitors the `urls` table and automatically processes URLs:
1. Status 'queued': Fetch content and inject into RAG
2. Status 'refresh': Delete existing documents and re-fetch
3. Periodic checking: Monitor 'ingested' URLs for content changes

Integration with RAG:
- Uses rag_common for embeddings and database operations
- Integrates with RAG injection service logic
- Maintains rag_groups.doc_count consistency
- Handles content hashing for change detection

Configuration (env):
- SLEEP_SECONDS: Polling interval (default: 5)
- BATCH_SIZE: URLs to process per loop (default: 10)
- CHECK_INTERVAL_SECONDS: How often to check ingested URLs (default: 3600)
- STALE_AFTER_SECONDS: Consider URL stale after this time (default: 21600)
- DEFAULT_RAG_GROUP: RAG group name for URL documents (default: "web_content")
- DEFAULT_EMBED_MODEL: Embedding model (default: from OLLAMA_EMBED_MODEL)

Run as:
    python -m backend_app.rag.daemons.url_watcher
Or via supervisord (already configured)
"""

import os
import sys
import time
import hashlib
import logging
import traceback
from typing import List, Tuple, Optional
from datetime import datetime
import psycopg

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import RAG common utilities
from rag.rag_common import (
    DATABASE_URL, PG_DSN, OLLAMA_EMBED_MODEL,
    get_rag_group_by_name, get_vector_store_for_model, db_exec
)

# Import URL discovery utilities (same directory)
from url_discovery import discover_urls_quick

# LangChain imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    import requests
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    DEPS_ERROR = str(e)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", str(60 * 60)))  # 1 hour
STALE_AFTER_SECONDS = int(os.getenv("STALE_AFTER_SECONDS", str(60 * 60 * 6)))  # 6 hours
DEFAULT_RAG_GROUP = os.getenv("DEFAULT_RAG_GROUP", "web_content")
DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", OLLAMA_EMBED_MODEL)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required to run URL watcher daemon")

if not DEPS_AVAILABLE:
    logger.error(f"Dependencies not available: {DEPS_ERROR}")
    raise RuntimeError(f"Required dependencies missing: {DEPS_ERROR}")


# =========================
# URL Fetching
# =========================
def fetch_url_content(url: str, timeout: int = 30) -> Tuple[str, str]:
    """
    Fetch content from URL.
    
    Returns:
        Tuple of (content, content_type)
    """
    try:
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'AgentKS-URLWatcher/1.0'
        })
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', 'text/html')
        return response.text, content_type
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        raise


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# =========================
# Database Operations
# =========================
def claim_source_urls_for_discovery(conn, batch_size: int) -> List[Tuple[str, str, bool]]:
    """
    Select and lock source URLs that need discovery.
    
    Returns:
        List of (id, url, is_parent)
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, url, is_parent
            FROM source_urls
            WHERE discovery_status = 'pending'
            ORDER BY created_at
            FOR UPDATE SKIP LOCKED
            LIMIT %s
        """, (batch_size,))
        rows = cur.fetchall()
        return [(r[0], r[1], r[2]) for r in rows]


def claim_discovered_urls(conn, batch_size: int) -> List[Tuple[str, str, str, str]]:
    """
    Select and lock discovered URLs with status queued/refresh.
    
    Returns:
        List of (id, url, status, source_url_id)
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, url, status, source_url_id
            FROM discovered_urls
            WHERE status IN ('queued', 'refresh')
            ORDER BY created_at
            FOR UPDATE SKIP LOCKED
            LIMIT %s
        """, (batch_size,))
        rows = cur.fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]


def claim_ingested_urls_for_check(conn, batch_size: int) -> List[Tuple[str, str, Optional[str]]]:
    """
    Select ingested URLs that haven't been checked recently.
    
    Returns:
        List of (id, url, last_fetched_at)
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, url, last_fetched_at
            FROM urls
            WHERE status = 'ingested' 
              AND (last_fetched_at IS NULL OR last_fetched_at < now() - (%s || ' seconds')::interval)
            ORDER BY last_fetched_at NULLS FIRST
            LIMIT %s
        """, (STALE_AFTER_SECONDS, batch_size))
        rows = cur.fetchall()
        return [(r[0], r[1], r[2]) for r in rows]


def get_latest_rag_content_hash(conn, url_id: str) -> Optional[str]:
    """
    Get the most recent content hash for documents associated with url_id.
    
    Returns:
        Content hash or None if no document exists
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content_hash 
            FROM rag_documents 
            WHERE url_id = %s 
            ORDER BY updated_at DESC NULLS LAST 
            LIMIT 1
        """, (url_id,))
        row = cur.fetchone()
        return row[0] if row else None


def update_url_status(conn, url_id: str, status: str, error: Optional[str] = None, content_hash: Optional[str] = None):
    """Update URL status and metadata."""
    with conn.cursor() as cur:
        if error:
            cur.execute("""
                UPDATE urls 
                SET status = %s, last_error = %s, last_fetched_at = now()
                WHERE id = %s
            """, (status, error, url_id))
        else:
            cur.execute("""
                UPDATE urls 
                SET status = %s, last_error = NULL, last_fetched_at = now()
                WHERE id = %s
            """, (status, url_id))
    conn.commit()


def ensure_rag_group_exists(conn, group_name: str, scope: str = "global") -> str:
    """
    Ensure RAG group exists, create if not.
    
    Returns:
        RAG group ID
    """
    result = get_rag_group_by_name(group_name, scope)
    if result:
        return result[0]  # group_id
    
    # Create new group
    import uuid
    group_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO rag_groups (id, name, scope, description, embed_model, doc_count, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, 0, now(), now())
        """, (
            group_id,
            group_name,
            scope,
            f"Auto-created group for web content",
            DEFAULT_EMBED_MODEL
        ))
    conn.commit()
    logger.info(f"Created RAG group: {group_name} (id: {group_id})")
    return group_id


# =========================
# Document Processing
# =========================
def delete_rag_documents_for_url(conn, url_id: str) -> int:
    """
    Delete all rag_documents for a URL and update group doc counts.
    
    Returns:
        Number of documents deleted
    """
    # Get counts per rag_group
    with conn.cursor() as cur:
        cur.execute("""
            SELECT rag_group_id, COUNT(*) 
            FROM rag_documents 
            WHERE url_id = %s 
            GROUP BY rag_group_id
        """, (url_id,))
        counts = cur.fetchall()
        
        # Delete documents
        cur.execute("DELETE FROM rag_documents WHERE url_id = %s", (url_id,))
        deleted = cur.rowcount if cur.rowcount is not None else 0
        
        # Update group doc counts
        for group_id, count in counts:
            cur.execute("""
                UPDATE rag_groups 
                SET doc_count = GREATEST(COALESCE(doc_count, 0) - %s, 0),
                    updated_at = now()
                WHERE id = %s
            """, (count, group_id))
    
    conn.commit()
    logger.info(f"Deleted {deleted} documents for URL {url_id}")
    return deleted


def inject_url_to_rag(conn, url_id: str, url: str, content: str, content_hash: str, 
                      rag_group_id: str, embed_model: str):
    """
    Process URL content and inject into RAG system.
    
    Steps:
    1. Chunk the content
    2. Generate embeddings
    3. Store in rag_documents and vector store
    4. Update rag_groups.doc_count
    """
    import uuid
    
    # Get vector store for the embedding model
    vector_store = get_vector_store_for_model(embed_model)
    if not vector_store:
        raise RuntimeError(f"Failed to get vector store for model: {embed_model}")
    
    # Split content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_text(content)
    logger.info(f"Split URL {url} into {len(chunks)} chunks")
    
    # Create document ID
    doc_id = str(uuid.uuid4())
    
    # Store in rag_documents table
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO rag_documents (id, rag_group_id, url_id, title, content, content_hash, metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, now(), now())
        """, (
            doc_id,
            rag_group_id,
            url_id,
            url,  # Use URL as title
            content,
            content_hash,
            psycopg.types.json.Json({
                "source": "url_watcher",
                "url": url,
                "content_length": len(content),
                "chunks": len(chunks)
            })
        ))
    
    # Create Document objects for vector store
    documents = []
    for i, chunk in enumerate(chunks):
        doc_metadata = {
            "document_id": doc_id,
            "url_id": url_id,
            "url": url,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "source": "url_watcher"
        }
        documents.append(Document(page_content=chunk, metadata=doc_metadata))
    
    # Add to vector store
    vector_store.add_documents(documents)
    logger.info(f"Added {len(documents)} chunks to vector store")
    
    # Update rag_groups.doc_count
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE rag_groups 
            SET doc_count = doc_count + 1, updated_at = now()
            WHERE id = %s
        """, (rag_group_id,))
    
    conn.commit()
    return len(chunks)


# =========================
# URL Discovery & Hierarchy
# =========================
def discover_and_create_discovered_urls(conn, source_url_id: str, source_url: str, is_parent: bool, max_urls: int = 50):
    """
    Discover URLs from source and create discovered_urls records.
    
    For non-parent URLs: Creates single discovered_url (depth=0) for the source URL itself
    For parent URLs: Discovers sub-URLs and creates discovered_urls for each
    
    Args:
        conn: Database connection
        source_url_id: ID of source URL record
        source_url: URL to discover from
        is_parent: Whether to discover sub-URLs
        max_urls: Maximum URLs to discover (for parent URLs)
    """
    try:
        logger.info(f"Processing source URL: {source_url} (is_parent={is_parent})")
        
        # Update discovery_status to 'discovering'
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE source_urls 
                SET discovery_status = 'discovering'
                WHERE id = %s
            """, (source_url_id,))
        conn.commit()
        
        discovered_urls_list = []
        
        if not is_parent:
            # Non-parent: just add the source URL itself
            discovered_urls_list = [{
                'url': source_url,
                'title': source_url.split('/')[-1] or source_url,
                'depth': 0
            }]
        else:
            # Parent: discover sub-URLs
            discovered = discover_urls_quick(source_url, max_urls=max_urls)
            
            if not discovered:
                logger.warning(f"No URLs discovered from {source_url}")
                # Still create one entry for the parent itself
                discovered_urls_list = [{
                    'url': source_url,
                    'title': source_url.split('/')[-1] or source_url,
                    'depth': 0
                }]
            else:
                discovered_urls_list = discovered
        
        # Create discovered_urls records
        created_count = 0
        with conn.cursor() as cur:
            for item in discovered_urls_list:
                disc_id = f"disc-{source_url_id}-{created_count}"
                cur.execute("""
                    INSERT INTO discovered_urls (
                        id, url, title, source_url_id, depth, status, discovered_at, created_at
                    ) VALUES (%s, %s, %s, %s, %s, 'queued', now(), now())
                    ON CONFLICT (id) DO NOTHING
                """, (
                    disc_id,
                    item['url'],
                    item.get('title', item['url']),
                    source_url_id,
                    item.get('depth', 0)
                ))
                created_count += 1
        
        # Update source_urls with success status
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE source_urls 
                SET discovery_status = 'discovered',
                    discovered_at = now(),
                    discovered_count = %s
                WHERE id = %s
            """, (created_count, source_url_id))
        conn.commit()
        
        logger.info(f"✓ Created {created_count} discovered_urls from source {source_url}")
        
    except Exception as e:
        error_msg = f"Failed to discover URLs from {source_url}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        # Update discovery_status to 'failed'
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE source_urls 
                SET discovery_status = 'failed',
                    discovery_error = %s
                WHERE id = %s
            """, (str(e), source_url_id))
        conn.commit()


def get_latest_rag_content_hash_for_discovered_url(conn, discovered_url_id: str) -> Optional[str]:
    """
    Get the most recent content hash for a discovered URL.
    
    Returns:
        Content hash from discovered_urls.content_hash or None
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content_hash 
            FROM discovered_urls 
            WHERE id = %s
        """, (discovered_url_id,))
        row = cur.fetchone()
        return row[0] if row else None


def update_discovered_url_status(conn, discovered_url_id: str, status: str, error: Optional[str] = None, content_hash: Optional[str] = None):
    """Update discovered URL status and metadata."""
    with conn.cursor() as cur:
        if error:
            cur.execute("""
                UPDATE discovered_urls 
                SET status = %s, last_error = %s, last_fetched_at = now()
                WHERE id = %s
            """, (status, error, discovered_url_id))
        else:
            if content_hash:
                cur.execute("""
                    UPDATE discovered_urls 
                    SET status = %s, content_hash = %s, last_fetched_at = now(), last_error = NULL
                    WHERE id = %s
                """, (status, content_hash, discovered_url_id))
            else:
                cur.execute("""
                    UPDATE discovered_urls 
                    SET status = %s, last_fetched_at = now(), last_error = NULL
                    WHERE id = %s
                """, (status, discovered_url_id))
    conn.commit()


def delete_rag_documents_for_discovered_url(conn, discovered_url_id: str):
    """Delete all RAG documents associated with a discovered URL."""
    with conn.cursor() as cur:
        # Get affected rag_group_ids for doc_count updates
        cur.execute("""
            SELECT DISTINCT rag_group_id 
            FROM rag_documents 
            WHERE url_id = %s
        """, (discovered_url_id,))
        group_ids = [r[0] for r in cur.fetchall()]
        
        # Delete documents
        cur.execute("DELETE FROM rag_documents WHERE url_id = %s", (discovered_url_id,))
        deleted = cur.rowcount
        
        # Update doc_counts
        for group_id in group_ids:
            if group_id:
                cur.execute("""
                    UPDATE rag_groups 
                    SET doc_count = GREATEST(0, doc_count - %s), updated_at = now()
                    WHERE id = %s
                """, (deleted, group_id))
    
    conn.commit()
    logger.info(f"Deleted {deleted} RAG documents for discovered_url {discovered_url_id}")


def check_if_selected(conn, parent_url_id: str, child_url: str) -> bool:
    """
    Check if child URL is selected in parent's discovered_urls.
    
    Args:
        conn: Database connection
        parent_url_id: Parent URL ID
        child_url: Child URL to check
        
    Returns:
        True if selected, False otherwise
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT discovered_urls 
            FROM urls 
            WHERE id = %s
        """, (parent_url_id,))
        row = cur.fetchone()
        
        if not row or not row[0]:
            return False
        
        discovered = row[0] if isinstance(row[0], list) else []
def check_if_selected(conn, parent_url_id: str, child_url: str) -> bool:
    """
    DEPRECATED: Not used in new two-table structure.
    Selection is now handled by creating discovered_urls records directly.
    """
    return False


# =========================
# URL Processing (Discovered URLs)
# =========================
def process_discovered_url(conn, discovered_url_id: str, url: str, status: str):
    """
    Process a single discovered URL according to its status.
    
    - queued: Fetch and inject
    - refresh: Delete old documents, then fetch and inject
    """
    try:
        logger.info(f"Processing discovered URL: {url} (status: {status})")
        
        # Ensure RAG group exists
        rag_group_id = ensure_rag_group_exists(conn, DEFAULT_RAG_GROUP)
        
        # Get embedding model for this group
        embed_model_result = db_exec(
            "SELECT embed_model FROM rag_groups WHERE id = %s",
            (rag_group_id,)
        )
        embed_model = embed_model_result[0][0] if embed_model_result else DEFAULT_EMBED_MODEL
        
        # If refresh, delete existing documents
        if status == 'refresh':
            delete_rag_documents_for_discovered_url(conn, discovered_url_id)
        
        # Fetch content
        content, content_type = fetch_url_content(url)
        content_hash = compute_content_hash(content)
        
        # Check if content has changed
        if status != 'refresh':
            existing_hash = get_latest_rag_content_hash_for_discovered_url(conn, discovered_url_id)
            if existing_hash == content_hash:
                logger.info(f"Content unchanged for {url}, skipping")
                update_discovered_url_status(conn, discovered_url_id, 'ingested', content_hash=content_hash)
                return
        
        # Inject to RAG - use discovered_url_id as url_id in rag_documents
        chunks_created = inject_url_to_rag(
            conn, discovered_url_id, url, content, content_hash,
            rag_group_id, embed_model
        )
        
        # Update discovered URL status and chunks count
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE discovered_urls 
                SET status = 'ingested', chunks_count = %s, rag_group_id = %s
                WHERE id = %s
            """, (chunks_created, rag_group_id, discovered_url_id))
        conn.commit()
        
        logger.info(f"✓ Successfully processed {url}: {chunks_created} chunks created")
        
    except Exception as e:
        error_msg = f"Error processing {url}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        update_discovered_url_status(conn, discovered_url_id, 'failed', error=str(e))
        conn.rollback()


def schedule_refresh_if_changed(conn, discovered_url_id: str, url: str) -> bool:
    """
    Fetch discovered URL, check if content has changed, and schedule refresh if needed.
    
    Returns:
        True if change detected and refresh scheduled
    """
    try:
        content, _ = fetch_url_content(url)
        content_hash = compute_content_hash(content)
        existing_hash = get_latest_rag_content_hash_for_discovered_url(conn, discovered_url_id)
        
        if existing_hash != content_hash:
            with conn.cursor() as cur:
                cur.execute("UPDATE discovered_urls SET status = 'refresh' WHERE id = %s", (discovered_url_id,))
            conn.commit()
            logger.info(f"✓ Change detected for {url}, scheduled for refresh")
            return True
        
        # Update last_fetched_at even if no change
        with conn.cursor() as cur:
            cur.execute("UPDATE discovered_urls SET last_fetched_at = now() WHERE id = %s", (discovered_url_id,))
        conn.commit()
        return False
        
    except Exception as e:
        logger.warning(f"Failed to check {url}: {e}")
        conn.rollback()
        return False


def claim_ingested_discovered_urls_for_check(conn, batch_size: int) -> List[Tuple[str, str, Optional[str]]]:
    """
    Select ingested discovered URLs that haven't been checked recently.
    
    Returns:
        List of (id, url, last_fetched_at)
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, url, last_fetched_at
            FROM discovered_urls
            WHERE status = 'ingested' 
              AND (last_fetched_at IS NULL OR last_fetched_at < now() - (%s || ' seconds')::interval)
            ORDER BY last_fetched_at NULLS FIRST
            LIMIT %s
        """, (STALE_AFTER_SECONDS, batch_size))
        rows = cur.fetchall()
        return [(r[0], r[1], r[2]) for r in rows]


# =========================
# Main Daemon Loop
# =========================
def main_loop():
    """
    Main daemon loop:
    1. Process source_urls for discovery
    2. Process discovered_urls for RAG ingestion
    3. Periodically check ingested discovered_urls for changes
    """
    logger.info("=" * 60)
    logger.info("Starting URL Watcher Daemon (Two-Table Mode)")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  SLEEP_SECONDS: {SLEEP_SECONDS}")
    logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"  CHECK_INTERVAL_SECONDS: {CHECK_INTERVAL_SECONDS}")
    logger.info(f"  STALE_AFTER_SECONDS: {STALE_AFTER_SECONDS}")
    logger.info(f"  DEFAULT_RAG_GROUP: {DEFAULT_RAG_GROUP}")
    logger.info(f"  DEFAULT_EMBED_MODEL: {DEFAULT_EMBED_MODEL}")
    logger.info("=" * 60)
    
    with psycopg.connect(PG_DSN) as conn:
        while True:
            try:
                work_done = False
                
                # Step 1: Process source_urls for discovery
                source_urls = claim_source_urls_for_discovery(conn, BATCH_SIZE)
                
                if source_urls:
                    logger.info(f"Discovering URLs from {len(source_urls)} source URLs")
                    for source_url_id, url, is_parent in source_urls:
                        try:
                            discover_and_create_discovered_urls(conn, source_url_id, url, is_parent)
                        except Exception as e:
                            logger.error(f"Error discovering from source URL {url}: {e}")
                            conn.rollback()
                    work_done = True
                    time.sleep(0.1)
                
                # Step 2: Process discovered_urls for RAG ingestion
                discovered_urls = claim_discovered_urls(conn, BATCH_SIZE)
                
                if discovered_urls:
                    logger.info(f"Processing {len(discovered_urls)} discovered URLs")
                    for discovered_url_id, url, status, source_url_id in discovered_urls:
                        try:
                            process_discovered_url(conn, discovered_url_id, url, status)
                        except Exception as e:
                            logger.error(f"Error processing discovered URL {url}: {e}")
                            conn.rollback()
                    work_done = True
                    time.sleep(0.1)
                
                # Step 3: Check ingested URLs for changes (if no other work)
                if not work_done:
                    check_urls = claim_ingested_discovered_urls_for_check(conn, BATCH_SIZE)
                    
                    if check_urls:
                        logger.info(f"Checking {len(check_urls)} ingested URLs for changes")
                        for discovered_url_id, url, _ in check_urls:
                            try:
                                schedule_refresh_if_changed(conn, discovered_url_id, url)
                            except Exception:
                                conn.rollback()
                        time.sleep(0.5)
                    else:
                        # Nothing to do, sleep
                        time.sleep(SLEEP_SECONDS)
                        
            except Exception as e:
                logger.error(f"Daemon error: {e}\n{traceback.format_exc()}")
                conn.rollback()
                time.sleep(5)


if __name__ == "__main__":
    main_loop()
