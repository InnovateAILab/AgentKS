"""Split URLs into source_urls and discovered_urls tables

Revision ID: 0007_split_urls_tables
Revises: 0006_seed_llms
Create Date: 2026-02-05 11:00:00

This migration restructures URL management by splitting into two tables:
1. source_urls: URLs manually added by users
2. discovered_urls: URLs discovered from source_urls (even if just one URL)

Benefits:
- Cleaner separation of concerns
- All RAG operations reference discovered_urls
- Source URLs track discovery status and metadata
- Discovered URLs track processing status independently

Flow:
User adds URL -> source_urls (with is_parent flag)
  -> Discovery runs -> discovered_urls (one or many)
    -> Processing -> rag_documents (linked to discovered_urls)
"""

# Alembic boilerplate
from alembic import op
import sqlalchemy as sa

revision = '0007_split_urls_tables'
down_revision = '0006_seed_llms'
branch_labels = None
depends_on = None


def upgrade():
    """Create source_urls and discovered_urls tables."""
    conn = op.get_bind()
    
    # Create source_urls table - URLs added by users
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS source_urls (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            scope TEXT DEFAULT 'global',
            tags JSONB DEFAULT '[]'::jsonb,
            is_parent BOOLEAN DEFAULT FALSE,
            
            -- Discovery tracking
            discovery_status TEXT DEFAULT 'pending',
            -- Values: 'pending', 'discovering', 'discovered', 'failed'
            discovered_at TIMESTAMP,
            discovery_error TEXT,
            discovered_count INTEGER DEFAULT 0,
            
            -- Metadata
            created_at TIMESTAMP DEFAULT now(),
            created_by TEXT,
            notes TEXT,
            
            -- Constraints
            CONSTRAINT source_urls_scope_check CHECK (scope IN ('global', 'private'))
        );
        
        COMMENT ON TABLE source_urls IS 'User-added source URLs for discovery and processing';
        COMMENT ON COLUMN source_urls.is_parent IS 'If true, discover sub-URLs; if false, treat as single URL';
        COMMENT ON COLUMN source_urls.discovery_status IS 'Status of URL discovery process';
        COMMENT ON COLUMN source_urls.discovered_count IS 'Number of URLs discovered (1 for non-parent URLs)';
    """))
    
    # Create discovered_urls table - URLs ready for processing
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS discovered_urls (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT,
            
            -- Link to source
            source_url_id TEXT NOT NULL REFERENCES source_urls(id) ON DELETE CASCADE,
            depth INTEGER DEFAULT 0,
            
            -- Processing status
            status TEXT DEFAULT 'queued',
            -- Values: 'queued', 'processing', 'ingested', 'failed', 'refresh'
            
            -- Processing metadata
            last_fetched_at TIMESTAMP,
            last_error TEXT,
            content_hash TEXT,
            
            -- RAG association (optional explicit link, or use rag_documents.url_id)
            rag_group_id TEXT,
            chunks_count INTEGER DEFAULT 0,
            
            -- Metadata
            discovered_at TIMESTAMP DEFAULT now(),
            created_at TIMESTAMP DEFAULT now(),
            
            -- Constraints
            CONSTRAINT discovered_urls_status_check CHECK (
                status IN ('queued', 'processing', 'ingested', 'failed', 'refresh')
            ),
            CONSTRAINT discovered_urls_depth_check CHECK (depth >= 0)
        );
        
        COMMENT ON TABLE discovered_urls IS 'URLs discovered from source_urls, ready for RAG processing';
        COMMENT ON COLUMN discovered_urls.source_url_id IS 'Reference to the source URL that led to this discovery';
        COMMENT ON COLUMN discovered_urls.depth IS 'Discovery depth: 0 = source URL itself, 1 = first-level link, etc.';
        COMMENT ON COLUMN discovered_urls.status IS 'Processing status for RAG ingestion';
        COMMENT ON COLUMN discovered_urls.content_hash IS 'SHA256 hash of content for change detection';
    """))
    
    # Create indexes for performance
    conn.execute(sa.text("""
        -- source_urls indexes
        CREATE INDEX IF NOT EXISTS idx_source_urls_discovery_status 
            ON source_urls(discovery_status);
        CREATE INDEX IF NOT EXISTS idx_source_urls_scope 
            ON source_urls(scope);
        CREATE INDEX IF NOT EXISTS idx_source_urls_created_at 
            ON source_urls(created_at DESC);
        
        -- discovered_urls indexes
        CREATE INDEX IF NOT EXISTS idx_discovered_urls_source_url_id 
            ON discovered_urls(source_url_id);
        CREATE INDEX IF NOT EXISTS idx_discovered_urls_status 
            ON discovered_urls(status);
        CREATE INDEX IF NOT EXISTS idx_discovered_urls_url 
            ON discovered_urls(url);
        CREATE INDEX IF NOT EXISTS idx_discovered_urls_rag_group_id 
            ON discovered_urls(rag_group_id);
        CREATE INDEX IF NOT EXISTS idx_discovered_urls_content_hash 
            ON discovered_urls(content_hash);
    """))
    
    # Migrate existing urls data to new tables
    conn.execute(sa.text("""
        -- Check if urls table exists and has data
        DO $$
        DECLARE
            urls_exists BOOLEAN;
            urls_count INTEGER;
        BEGIN
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'urls'
            ) INTO urls_exists;
            
            IF urls_exists THEN
                SELECT COUNT(*) INTO urls_count FROM urls;
                
                IF urls_count > 0 THEN
                    -- Migrate to source_urls
                    INSERT INTO source_urls (
                        id, url, scope, tags, is_parent,
                        discovery_status, created_at
                    )
                    SELECT 
                        id, 
                        url, 
                        COALESCE(scope, 'global'),
                        COALESCE(tags, '[]'::jsonb),
                        COALESCE(is_parent, FALSE),
                        CASE 
                            WHEN status = 'ingested' THEN 'discovered'
                            WHEN status = 'failed' THEN 'failed'
                            ELSE 'pending'
                        END,
                        COALESCE(created_at, now())
                    FROM urls
                    ON CONFLICT (url) DO NOTHING;
                    
                    -- Migrate to discovered_urls (treat all as discovered with depth 0)
                    INSERT INTO discovered_urls (
                        id, url, source_url_id, depth, status,
                        last_fetched_at, last_error, content_hash, created_at
                    )
                    SELECT 
                        'disc-' || id,
                        url,
                        id,
                        0,
                        COALESCE(status, 'queued'),
                        last_fetched_at,
                        last_error,
                        content_hash,
                        COALESCE(created_at, now())
                    FROM urls
                    ON CONFLICT (id) DO NOTHING;
                    
                    RAISE NOTICE 'Migrated % URLs from urls table', urls_count;
                END IF;
            END IF;
        END $$;
    """))
    
    # Update rag_documents to reference discovered_urls
    conn.execute(sa.text("""
        -- Add comment about url_id reference
        COMMENT ON COLUMN rag_documents.url_id IS 'References discovered_urls.id (or old urls.id for backward compatibility)';
    """))


def downgrade():
    """Revert to single urls table."""
    conn = op.get_bind()
    
    # Recreate urls table if needed and merge data back
    conn.execute(sa.text("""
        -- Create urls table if it doesn't exist
        CREATE TABLE IF NOT EXISTS urls (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            scope TEXT DEFAULT 'global',
            tags JSONB DEFAULT '[]'::jsonb,
            status TEXT DEFAULT 'queued',
            is_parent BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT now(),
            last_fetched_at TIMESTAMP,
            last_error TEXT,
            content_hash TEXT
        );
        
        -- Migrate source_urls back to urls
        INSERT INTO urls (id, url, scope, tags, status, is_parent, created_at)
        SELECT 
            id, url, scope, tags,
            CASE 
                WHEN discovery_status = 'discovered' THEN 'ingested'
                WHEN discovery_status = 'failed' THEN 'failed'
                ELSE 'queued'
            END,
            is_parent,
            created_at
        FROM source_urls
        ON CONFLICT (id) DO NOTHING;
        
        -- Drop new tables
        DROP TABLE IF EXISTS discovered_urls CASCADE;
        DROP TABLE IF EXISTS source_urls CASCADE;
    """))
