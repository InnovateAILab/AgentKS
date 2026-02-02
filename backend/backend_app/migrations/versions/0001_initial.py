"""initial migration

Revision ID: 0001_initial
Revises: 
Create Date: 2026-02-02 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Execute the same SQL as in 01_create_admin_tables.sql to keep parity
    op.execute('''
-- Admin UI and RAG schema (improved/normalized)

-- 1) URLs table: canonical list of URLs to be crawled/ingested
CREATE TABLE IF NOT EXISTS urls (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    scope TEXT DEFAULT 'global',
    tags JSONB DEFAULT '[]'::jsonb,
    status TEXT DEFAULT 'queued',
    last_fetched_at TIMESTAMP,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_urls_scope ON urls(scope);

-- 2) RAG groups: logical collections/indices (e.g. 'global', 'private:user123')
CREATE TABLE IF NOT EXISTS rag_groups (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    scope TEXT DEFAULT 'global',
    owner TEXT,
    description TEXT,
    embed_model TEXT,
    doc_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_rag_groups_name_scope ON rag_groups(name, scope);

-- 3) Association table linking rag_groups to urls (many-to-many)
CREATE TABLE IF NOT EXISTS rag_group_urls (
    rag_group_id TEXT NOT NULL REFERENCES rag_groups(id) ON DELETE CASCADE,
    url_id TEXT NOT NULL REFERENCES urls(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'linked',
    added_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (rag_group_id, url_id)
);

-- 4) RAG documents: extracted or injected content items belonging to a rag_group
CREATE TABLE IF NOT EXISTS rag_documents (
    id TEXT PRIMARY KEY,
    rag_group_id TEXT NOT NULL REFERENCES rag_groups(id) ON DELETE CASCADE,
    url_id TEXT REFERENCES urls(id) ON DELETE SET NULL,
    title TEXT,
    content TEXT,
    content_hash TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rag_documents_rag_group_id ON rag_documents(rag_group_id);

-- 5) MCP endpoints (microservice/process endpoints that provide tools)
CREATE TABLE IF NOT EXISTS mcps (
    id TEXT PRIMARY KEY,
    name TEXT,
    endpoint TEXT,
    kind TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    tags JSONB DEFAULT '[]'::jsonb,
    status TEXT DEFAULT 'enabled',
    last_checked_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_mcps_status ON mcps(status);

-- 6) Tools and tool runs (tooling catalog and execution history)
CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT,
    mcp_id TEXT REFERENCES mcps(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_runs (
    id TEXT PRIMARY KEY,
    tool_id TEXT REFERENCES tools(id) ON DELETE SET NULL,
    input JSONB,
    output JSONB,
    status TEXT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name);
''')


def downgrade() -> None:
    # Drop in reverse order to respect fk constraints
    op.execute('''
DROP INDEX IF EXISTS idx_tools_name;
DROP TABLE IF EXISTS tool_runs;
DROP TABLE IF EXISTS tools;
DROP INDEX IF EXISTS idx_mcps_status;
DROP TABLE IF EXISTS mcps;
DROP INDEX IF EXISTS idx_rag_documents_rag_group_id;
DROP TABLE IF EXISTS rag_documents;
DROP TABLE IF EXISTS rag_group_urls;
DROP INDEX IF EXISTS ux_rag_groups_name_scope;
DROP TABLE IF EXISTS rag_groups;
DROP INDEX IF EXISTS idx_urls_scope;
DROP TABLE IF EXISTS urls;
''')
