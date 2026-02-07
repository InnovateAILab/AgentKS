# Migrating Existing Tools to MCP Watcher

## Overview

If you have existing tools in the `tools` table that were manually registered, you can migrate them to be managed by the MCP watcher system.

## Pre-Migration Checklist

Before migrating:
1. ✅ Backup your database
2. ✅ Ensure MCP watcher is running
3. ✅ Document existing tools
4. ✅ Identify which tools come from which MCPs

## Migration Strategies

### Strategy 1: Clean Slate (Recommended for New Systems)

**When to use:** You're just starting and don't have production data yet.

**Steps:**
1. Delete all existing tools
2. Add MCP servers
3. Let MCP watcher discover everything

```sql
-- Backup existing tools
CREATE TABLE tools_backup AS SELECT * FROM tools;

-- Delete all tools
DELETE FROM tools;

-- Add your MCP servers via Admin UI
-- Tools will be auto-discovered within 60 seconds
```

### Strategy 2: Side-by-Side (Recommended for Production)

**When to use:** You have existing tools in production and want to test MCP watcher first.

**Steps:**
1. Add MCP servers with `status='disabled'`
2. Manually trigger discovery for one MCP
3. Compare discovered tools with existing
4. Enable MCP watcher gradually

```sql
-- Step 1: Add MCP but don't auto-discover yet
INSERT INTO mcps (id, name, endpoint, kind, status, created_at)
VALUES ('test-mcp', 'Test MCP', 'http://mcp:5000', 'http', 'disabled', now());

-- Step 2: Manually trigger discovery
UPDATE mcps SET status = 'pending' WHERE id = 'test-mcp';
-- Wait 60 seconds for mcp_watcher to process

-- Step 3: Compare tools
SELECT 
  'Discovered' as source,
  name,
  metadata->>'description' as description
FROM tools 
WHERE mcp_id = 'test-mcp'
UNION ALL
SELECT 
  'Existing' as source,
  name,
  metadata->>'description' as description
FROM tools 
WHERE mcp_id IS NULL;

-- Step 4: If satisfied, enable auto-discovery
UPDATE mcps SET status = 'enabled' WHERE id = 'test-mcp';
```

### Strategy 3: Migration with Mapping

**When to use:** You want to preserve existing tool IDs and gradually migrate.

**Steps:**
1. Create mapping table
2. Add MCPs
3. Update existing tools with mcp_id references
4. Re-index tools with MCP context

```sql
-- Step 1: Create mapping table
CREATE TEMP TABLE tool_mcp_mapping (
    tool_name TEXT,
    mcp_id TEXT
);

-- Step 2: Define mappings (update with your data)
INSERT INTO tool_mcp_mapping VALUES
('arxiv_search', 'basic-tools-mcp'),
('cds_search', 'basic-tools-mcp'),
('inspirehep_search', 'basic-tools-mcp'),
('web_search', 'basic-tools-mcp');

-- Step 3: Add MCPs (if not already added)
INSERT INTO mcps (id, name, endpoint, kind, status, created_at)
VALUES 
('basic-tools-mcp', 'Basic Tools MCP', 'http://basic_tools_mcp_service:5000/mcp', 'http', 'enabled', now())
ON CONFLICT (id) DO NOTHING;

-- Step 4: Update existing tools with MCP references
UPDATE tools t
SET mcp_id = m.mcp_id
FROM tool_mcp_mapping m
WHERE t.name = m.tool_name;

-- Step 5: Re-index with MCP context
-- Run in Python shell:
-- from app.tool_discovery import reindex_all_tools
-- reindex_all_tools()
```

## Example: Migrate Basic Tools

### Before Migration

```sql
SELECT id, name, kind, mcp_id FROM tools;

-- Results:
-- id                | name              | kind           | mcp_id
-- -----------------+-------------------+----------------+--------
-- tool-001         | arxiv_search      | arxiv_search   | NULL
-- tool-002         | cds_search        | cds_search     | NULL
-- tool-003         | web_search        | searxng_search | NULL
```

### Migration Steps

```sql
-- 1. Add Basic Tools MCP
INSERT INTO mcps (id, name, endpoint, kind, description, context, resource, status, created_at)
VALUES (
    'basic-tools-mcp',
    'Basic Tools MCP',
    'http://basic_tools_mcp_service:5000/mcp',
    'http',
    'Core research and search tools',
    'Tools for searching academic databases and the web',
    'arXiv, CDS, INSPIRE-HEP, SearXNG',
    'pending',
    now()
);

-- 2. Wait 60 seconds for mcp_watcher to discover tools
-- New tools will have IDs like: basic-tools-mcp_arxiv_search

-- 3. Verify new tools were created
SELECT id, name, kind, mcp_id 
FROM tools 
WHERE mcp_id = 'basic-tools-mcp';

-- Results:
-- id                           | name              | kind     | mcp_id
-- ----------------------------+-------------------+----------+-----------------
-- basic-tools-mcp_arxiv_search | arxiv_search      | mcp_tool | basic-tools-mcp
-- basic-tools-mcp_cds_search   | cds_search        | mcp_tool | basic-tools-mcp
-- basic-tools-mcp_web_search   | web_search        | mcp_tool | basic-tools-mcp

-- 4. Delete old tools (after verifying new ones work)
DELETE FROM tools WHERE mcp_id IS NULL AND kind IN ('arxiv_search', 'cds_search', 'searxng_search');
```

### After Migration

```sql
SELECT id, name, kind, mcp_id FROM tools;

-- Results:
-- id                           | name              | kind     | mcp_id
-- ----------------------------+-------------------+----------+-----------------
-- basic-tools-mcp_arxiv_search | arxiv_search      | mcp_tool | basic-tools-mcp
-- basic-tools-mcp_cds_search   | cds_search        | mcp_tool | basic-tools-mcp
-- basic-tools-mcp_web_search   | web_search        | mcp_tool | basic-tools-mcp
```

## Removing Hardcoded Tool Creation

If your code has hardcoded tool creation (like in `ensure_default_tools()`), you should remove it after migration:

**Before (in `app/main.py`):**
```python
def ensure_default_tools():
    _ensure("cds_search", "Search CERN Document Server...", "cds_search", {...})
    _ensure("arxiv_search", "Search arXiv...", "arxiv_search", {...})
    _ensure("inspirehep_search", "Search INSPIRE-HEP...", "inspirehep_search", {...})
    _ensure("web_search", "Search the web...", "searxng_search", {...})
```

**After:**
```python
def ensure_default_tools():
    # Tools are now auto-discovered via MCP watcher
    # Remove hardcoded tool creation
    pass
```

**Or better yet, remove the function entirely:**
```python
# Delete ensure_default_tools() and all calls to it
# MCP watcher handles everything now
```

## Handling Custom Tools

If you have custom tools that don't come from an MCP:

```sql
-- Keep custom tools, just mark them clearly
UPDATE tools 
SET kind = 'custom', 
    metadata = jsonb_set(metadata, '{source}', '"manual"'::jsonb)
WHERE mcp_id IS NULL;
```

## Re-indexing After Migration

After migration, re-index all tools to include MCP context:

```python
# In Python shell or script
from app.tool_discovery import reindex_all_tools

# This will:
# 1. Delete old tool embeddings
# 2. Re-create embeddings with MCP context
# 3. Combine tool + MCP descriptions for richer search
reindex_all_tools()
```

```bash
# Or via docker-compose
docker-compose exec backend python -c "
from app.tool_discovery import reindex_all_tools
reindex_all_tools()
print('Re-indexing complete')
"
```

## Validation

After migration, validate that everything works:

```sql
-- 1. Check all tools have MCP references (except custom)
SELECT COUNT(*) as tools_without_mcp
FROM tools 
WHERE mcp_id IS NULL AND kind != 'custom';
-- Should return 0

-- 2. Check tool counts per MCP
SELECT 
    m.name as mcp_name,
    COUNT(t.id) as tool_count,
    m.status,
    m.last_checked_at
FROM mcps m
LEFT JOIN tools t ON t.mcp_id = m.id
GROUP BY m.name, m.status, m.last_checked_at;

-- 3. Verify embeddings exist
SELECT 
    c.name as collection,
    COUNT(e.id) as embedding_count
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'tool_embeddings'
GROUP BY c.name;

-- 4. Test tool discovery
-- In Python:
-- from app.tool_discovery import discover_tools
-- tools = discover_tools("search for papers", user_scope="global", top_k=5)
-- print(tools)
```

## Rollback Plan

If migration fails:

```sql
-- Restore from backup
DROP TABLE tools;
ALTER TABLE tools_backup RENAME TO tools;

-- Disable MCP watcher
UPDATE mcps SET status = 'disabled';

-- Restart backend to stop mcp_watcher daemon
-- docker-compose restart backend
```

## Migration Checklist

- [ ] Backup database
- [ ] Document existing tools
- [ ] Add MCP servers via Admin UI
- [ ] Wait for auto-discovery (60 seconds)
- [ ] Verify new tools created
- [ ] Test tool discovery search
- [ ] Re-index with MCP context
- [ ] Validate embeddings
- [ ] Test LLM with discovered tools
- [ ] Delete old hardcoded tools
- [ ] Remove `ensure_default_tools()` code
- [ ] Monitor logs for errors
- [ ] Update documentation

## Common Issues

### Duplicate Tools

**Problem:** Old and new tools coexist with same name but different IDs

**Solution:**
```sql
-- List duplicates
SELECT name, COUNT(*) 
FROM tools 
GROUP BY name 
HAVING COUNT(*) > 1;

-- Delete old tools (keep MCP-managed ones)
DELETE FROM tools 
WHERE name IN (
    SELECT name 
    FROM tools 
    GROUP BY name 
    HAVING COUNT(*) > 1
)
AND mcp_id IS NULL;
```

### Missing Embeddings

**Problem:** Tools exist but not searchable

**Solution:**
```python
# Re-index all tools
from app.tool_discovery import reindex_all_tools
reindex_all_tools()
```

### Tools Not Updating

**Problem:** MCP server added new tools but not appearing

**Solution:**
```sql
-- Force refresh
UPDATE mcps SET last_checked_at = NULL, status = 'pending' WHERE id = 'mcp-id';
```

## Post-Migration Monitoring

Monitor for 24 hours after migration:

```bash
# Watch daemon logs
docker-compose logs -f backend | grep -E "mcp_watcher|tool"

# Check for errors
docker-compose exec postgres psql -U user -d dbname -c \
  "SELECT id, name, last_error FROM mcps WHERE last_error IS NOT NULL;"

# Monitor tool counts
watch -n 60 'docker-compose exec postgres psql -U user -d dbname -c \
  "SELECT m.name, COUNT(t.id) FROM mcps m LEFT JOIN tools t ON t.mcp_id = m.id GROUP BY m.name;"'
```

## Best Practices

1. **Migrate gradually**: Start with one MCP, validate, then migrate others
2. **Test in staging first**: Don't migrate production directly
3. **Keep backups**: Database backups before each migration step
4. **Monitor actively**: Watch logs and metrics during migration
5. **Document mappings**: Keep record of which tools came from which MCPs
6. **Validate thoroughly**: Test tool discovery and LLM binding after migration
7. **Clean up code**: Remove hardcoded tool creation after successful migration

---

**Migration Support:** If you encounter issues, check logs, database, and documentation. The MCP watcher is designed to be robust and will automatically retry failed operations.
