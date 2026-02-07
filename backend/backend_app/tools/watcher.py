#!/usr/bin/env python3
"""MCP Watcher Daemon

Monitors the mcps table for new or updated MCP servers, retrieves their tools,
and populates the tools table. This replaces manual tool filling with automatic
discovery and registration.

Functions moved from main.py:
- tool_upsert: Insert/update tools in database
- index_tool_with_mcp_context: Index tools for semantic search with MCP context
"""
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .tool_discovery import index_tool_with_mcp_context, index_tool_simple
from .discovery import discover_mcp_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mcp_watcher")

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "")
MCP_CHECK_INTERVAL = int(os.getenv("MCP_CHECK_INTERVAL", "60"))  # seconds between checks
MCP_CLAIM_LIMIT = int(os.getenv("MCP_CLAIM_LIMIT", "5"))  # max MCPs to process per cycle

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

PG_DSN = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")


# =========================
# Database helpers
# =========================
def db_exec(query: str, params: tuple = ()):
    """Execute a SQL query and return results if any."""
    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            try:
                return cur.fetchall()
            except psycopg.ProgrammingError:
                return None


def db_exec_commit(query: str, params: tuple = ()):
    """Execute a SQL query and commit immediately."""
    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            conn.commit()
            try:
                return cur.fetchall()
            except psycopg.ProgrammingError:
                return None


# =========================
# Tool management functions (moved from main.py)
# =========================
def tool_upsert(
    tool_id: str,
    name: str,
    kind: str,
    mcp_id: Optional[str],
    metadata: Dict[str, Any],
    tags: List[str],
):
    """
    Insert or update a tool in the database.
    
    Args:
        tool_id: Unique tool identifier
        name: Tool name
        kind: Tool type/kind (e.g., 'mcp_tool', 'http_json')
        mcp_id: Reference to parent MCP server
        metadata: Tool metadata including description, enabled, scope, config
        tags: List of tags for categorization
    """
    metadata_json = json.dumps(metadata)
    tags_json = json.dumps(tags)
    
    db_exec_commit(
        """
        INSERT INTO tools (id, name, kind, mcp_id, metadata, tags, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, now(), now())
        ON CONFLICT (id)
        DO UPDATE SET
          name = EXCLUDED.name,
          kind = EXCLUDED.kind,
          mcp_id = EXCLUDED.mcp_id,
          metadata = EXCLUDED.metadata,
          tags = EXCLUDED.tags,
          updated_at = now()
        """,
        (tool_id, name, kind, mcp_id, metadata_json, tags_json),
    )
    logger.info(f"Upserted tool: {name} (id={tool_id}, mcp_id={mcp_id})")


def tool_exists(tool_id: str) -> bool:
    """Check if a tool already exists in the database."""
    rows = db_exec("SELECT id FROM tools WHERE id = %s", (tool_id,))
    return bool(rows)


def get_mcp_by_id(mcp_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve MCP server details by ID."""
    rows = db_exec(
        "SELECT id, name, endpoint, kind, description, resource, context, auth, status FROM mcps WHERE id = %s",
        (mcp_id,),
    )
    if not rows:
        return None
    
    row = rows[0]
    return {
        "id": row[0],
        "name": row[1],
        "endpoint": row[2],
        "kind": row[3],
        "description": row[4],
        "resource": row[5],
        "context": row[6],
        "auth": row[7],
        "status": row[8],
    }


# =========================
# MCP claiming and processing
# =========================
def claim_pending_mcps(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Claim MCP servers that need tool discovery.
    
    MCPs are candidates if:
    - status = 'pending' (newly added, awaiting tool discovery)
    - status = 'enabled' AND last_checked_at IS NULL (never processed)
    - status = 'enabled' AND last_checked_at < (now - 1 day) (periodic refresh)
    
    Returns list of MCP records with all relevant fields.
    """
    query = """
        UPDATE mcps
        SET status = 'processing', updated_at = now()
        WHERE id IN (
            SELECT id FROM mcps
            WHERE status IN ('pending', 'enabled')
              AND (
                last_checked_at IS NULL
                OR last_checked_at < (now() - interval '1 day')
              )
            ORDER BY created_at ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, name, endpoint, kind, description, resource, context, auth, tags
    """
    rows = db_exec_commit(query, (limit,))
    
    if not rows:
        return []
    
    mcps = []
    for row in rows:
        mcps.append({
            "id": row[0],
            "name": row[1],
            "endpoint": row[2],
            "kind": row[3],
            "description": row[4],
            "resource": row[5],
            "context": row[6],
            "auth": row[7],
            "tags": row[8] if row[8] else [],
        })
    
    logger.info(f"Claimed {len(mcps)} MCP(s) for tool discovery")
    return mcps


def mark_mcp_processed(mcp_id: str, success: bool, error: Optional[str] = None, tool_count: int = 0):
    """Mark an MCP as processed (success or failure)."""
    if success:
        db_exec_commit(
            """
            UPDATE mcps
            SET status = 'enabled',
                last_checked_at = now(),
                last_error = NULL,
                updated_at = now(),
                metadata = jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{tool_count}',
                    %s::text::jsonb
                )
            WHERE id = %s
            """,
            (tool_count, mcp_id),
        )
        logger.info(f"MCP {mcp_id} processed successfully ({tool_count} tools)")
    else:
        db_exec_commit(
            """
            UPDATE mcps
            SET status = 'error',
                last_error = %s,
                updated_at = now()
            WHERE id = %s
            """,
            (error, mcp_id),
        )
        logger.error(f"MCP {mcp_id} processing failed: {error}")


# =========================
# Tool registration and indexing
# =========================
def register_tools_for_mcp(mcp: Dict[str, Any], tools: List[Dict[str, Any]]):
    """
    Register discovered tools in the database and index them for semantic search.
    
    Args:
        mcp: MCP server dict with id, name, description, context, resource
        tools: List of tool dicts from MCP discovery
    """
    mcp_id = mcp["id"]
    mcp_name = mcp["name"]
    mcp_description = mcp.get("description", "")
    mcp_context = mcp.get("context", "")
    mcp_resource = mcp.get("resource", "")
    
    registered_count = 0
    
    for tool in tools:
        tool_name = tool.get("name", "")
        tool_description = tool.get("description", "")
        tool_input_schema = tool.get("inputSchema", {})
        
        if not tool_name:
            logger.warning(f"Skipping tool with no name from MCP {mcp_name}")
            continue
        
        # Generate deterministic tool ID: mcp_id + tool_name
        tool_id = f"{mcp_id}_{tool_name}"
        
        # Prepare metadata
        metadata = {
            "description": tool_description,
            "enabled": True,
            "scope": "global",  # Default scope; can be overridden via admin UI
            "config": {},
            "inputSchema": tool_input_schema,
        }
        
        # Prepare tags from MCP tags
        tags = mcp.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        
        # Upsert tool to database
        try:
            tool_upsert(
                tool_id=tool_id,
                name=tool_name,
                kind="mcp_tool",
                mcp_id=mcp_id,
                metadata=metadata,
                tags=tags,
            )
            registered_count += 1
        except Exception as e:
            logger.error(f"Failed to upsert tool {tool_name} for MCP {mcp_name}: {e}")
            continue
        
        # Index tool for semantic search with MCP context
        try:
            index_tool_with_mcp_context(
                tool_id=tool_id,
                name=tool_name,
                description=tool_description,
                enabled=True,
                scope="global",
                mcp_description=mcp_description,
                mcp_context=mcp_context,
                mcp_resource=mcp_resource,
            )
        except Exception as e:
            logger.error(f"Failed to index tool {tool_name} for MCP {mcp_name}: {e}")
            continue
    
    logger.info(f"Registered {registered_count}/{len(tools)} tools for MCP {mcp_name}")


# =========================
# Main processing loop
# =========================
def process_mcp(mcp: Dict[str, Any]):
    """Process a single MCP: discover tools and register them."""
    mcp_id = mcp["id"]
    mcp_name = mcp["name"]
    
    logger.info(f"Processing MCP: {mcp_name} (id={mcp_id}, endpoint={mcp['endpoint']})")
    
    try:
        # Discover tools from MCP
        tools = discover_mcp_tools(mcp)
        
        if not tools:
            logger.warning(f"No tools discovered from MCP {mcp_name}")
            mark_mcp_processed(mcp_id, success=True, tool_count=0)
            return
        
        # Register and index tools
        register_tools_for_mcp(mcp, tools)
        
        # Mark as successfully processed
        mark_mcp_processed(mcp_id, success=True, tool_count=len(tools))
    
    except Exception as e:
        logger.error(f"Error processing MCP {mcp_name}: {e}", exc_info=True)
        mark_mcp_processed(mcp_id, success=False, error=str(e))


def main_loop():
    """Main daemon loop: claim and process pending MCPs."""
    logger.info("MCP Watcher daemon started")
    logger.info(f"Check interval: {MCP_CHECK_INTERVAL}s, Claim limit: {MCP_CLAIM_LIMIT}")
    
    while True:
        try:
            # Claim pending MCPs
            mcps = claim_pending_mcps(limit=MCP_CLAIM_LIMIT)
            
            if not mcps:
                logger.debug("No pending MCPs to process")
            else:
                # Process each MCP
                for mcp in mcps:
                    try:
                        process_mcp(mcp)
                    except Exception as e:
                        logger.error(f"Unexpected error processing MCP {mcp['id']}: {e}", exc_info=True)
            
            # Sleep before next check
            logger.debug(f"Sleeping for {MCP_CHECK_INTERVAL}s")
            time.sleep(MCP_CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(MCP_CHECK_INTERVAL)


if __name__ == "__main__":
    main_loop()
