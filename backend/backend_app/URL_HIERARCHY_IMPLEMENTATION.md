# URL Hierarchy System Implementation

## Overview
Enhanced the URL processing system to support multi-page sites (like ReadTheDocs documentation). The system can now:
1. Accept a parent URL from the admin UI
2. Automatically discover all related sub-URLs
3. Present the list for user selection
4. Process only the selected URLs

## Files Created

### 1. `/backend/backend_app/rag/url_discovery.py`
**Purpose**: URL discovery utilities for finding related URLs from parent pages

**Key Components**:
- `URLDiscoverer` class: Main discovery engine
  - `discover_urls()`: Generic URL discovery with depth control
  - `discover_documentation_urls()`: Specialized for docs sites (ReadTheDocs, MkDocs, Sphinx)
  - Configurable: max_depth, max_urls, timeout
  
- `discover_urls_quick()`: Simple helper function for quick discovery
  
**Features**:
- Sitemap parsing and HTML link extraction
- Same-domain filtering
- Documentation-specific navigation detection
- Automatic title extraction from HTML or URL
- Excludes binary files (PDF, images, archives, etc.)

**Dependencies**: 
- requests
- BeautifulSoup4

### 2. `/backend/backend_app/web/templates/urls_discover.html`
**Purpose**: Admin UI template for viewing and selecting discovered URLs

**Features**:
- Display all discovered URLs with titles
- Checkboxes for selection
- Select All / Deselect All buttons
- Live count of selected URLs
- Shows discovery status (discovering, discovered, failed)
- Responsive design with scrollable list

### 3. `/backend/backend_app/migrations/versions/0007_url_hierarchy.py`
**Purpose**: Database migration for URL hierarchy support

**Schema Changes**:
```sql
ALTER TABLE urls ADD COLUMN parent_url_id TEXT REFERENCES urls(id) ON DELETE CASCADE;
ALTER TABLE urls ADD COLUMN is_parent BOOLEAN DEFAULT FALSE;
ALTER TABLE urls ADD COLUMN discovered_urls JSONB DEFAULT '[]'::jsonb;
ALTER TABLE urls ADD COLUMN discovery_status TEXT DEFAULT 'pending';
ALTER TABLE urls ADD COLUMN depth INTEGER DEFAULT 0;
ALTER TABLE urls ADD COLUMN discovered_at TIMESTAMP;
```

**Indexes**:
- `idx_urls_parent_url_id` - for parent-child queries
- `idx_urls_is_parent` - for filtering parent URLs
- `idx_urls_discovery_status` - for discovery workflow queries

**JSONB Structure** for `discovered_urls`:
```json
[
  {
    "url": "https://example.com/page1",
    "title": "Page 1 Title",
    "selected": true
  },
  {
    "url": "https://example.com/page2",
    "title": "Page 2 Title", 
    "selected": false
  }
]
```

## Files Modified

### 4. `/backend/backend_app/rag/daemons/url_watcher.py`
**Changes**:
- Added import for `url_discovery` module
- New function `discover_and_save_child_urls()`: Performs discovery and saves results to JSONB
- New function `check_if_parent_url()`: Checks if URL is a parent
- New function `check_if_selected()`: Checks if child URL is selected in parent's list
- Modified `process_url()`: Now handles parent URLs differently:
  - Parent URLs: trigger discovery, set status to 'discovered'
  - Regular URLs: fetch and ingest as before

**Workflow**:
1. URL watcher sees `status='queued'` and `is_parent=true`
2. Calls `discover_and_save_child_urls()`
3. Discovery runs via `discover_urls_quick()`
4. Results saved to `discovered_urls` JSONB field
5. Status updated to `discovery_status='discovered'`
6. Admin selects URLs via UI
7. Selected URLs created as child records with `status='queued'`
8. URL watcher processes child URLs normally

### 5. `/backend/backend_app/web/main.py`
**Changes**:

**Handler: `urls_add()` (POST /admin/urls/add)**
- Added `is_parent` form parameter
- Converts checkbox value to boolean
- Includes `is_parent` in INSERT statement

**Handler: `urls_list()` (GET /admin/urls)**
- Extended SELECT to include `is_parent` and `discovery_status` columns
- Added these fields to items dictionary for template

**New Handler: `urls_discover_view()` (GET /admin/urls/{url_id}/discover)**
- Displays discovery UI for parent URL
- Shows discovered URLs with current selection state
- Validates that URL is a parent URL

**New Handler: `urls_discover_save()` (POST /admin/urls/{url_id}/discover)**
- Saves URL selection from checkboxes
- Updates `discovered_urls` JSONB with selection state
- Creates child URL records for selected URLs
- Sets `parent_url_id`, `depth=1`, and `status='queued'` for children

### 6. `/backend/backend_app/web/templates/urls_add.html`
**Changes**:
- Added checkbox for "Multi-page site (discover sub-URLs)"
- Form field name: `is_parent`
- Includes helpful description about when to use this feature

### 7. `/backend/backend_app/web/templates/urls_list.html`
**Changes**:
- Added "Actions" column to table
- Shows "Parent" badge for parent URLs
- Shows "discovered" status badge
- Shows "View URLs" link for parent URLs with `discovery_status='discovered'`
- Shows "Discovering..." text for URLs being discovered
- Updated colspan in empty row message from 6 to 7

## Database Schema Summary

### urls table (after migration 0007)
```
id                  TEXT PRIMARY KEY
url                 TEXT NOT NULL
scope               TEXT
tags                JSONB
status              TEXT (queued, discovered, ingested, failed, refresh)
is_parent           BOOLEAN DEFAULT FALSE
parent_url_id       TEXT REFERENCES urls(id) ON DELETE CASCADE
discovered_urls     JSONB DEFAULT '[]'::jsonb
discovery_status    TEXT DEFAULT 'pending' (pending, discovering, discovered, failed)
depth               INTEGER DEFAULT 0
discovered_at       TIMESTAMP
created_at          TIMESTAMP
last_fetched_at     TIMESTAMP
last_error          TEXT
content_hash        TEXT
```

## User Workflow

### Adding a Multi-Page Site
1. Navigate to "Add URL" page
2. Enter parent URL (e.g., `https://panda-wms.readthedocs.io/en/latest/`)
3. Check "Multi-page site (discover sub-URLs)" checkbox
4. Submit form
5. URL is queued with `is_parent=true`

### Discovery Process
1. URL watcher picks up parent URL
2. Discovers all related URLs automatically
3. Saves discovered URLs to database
4. Status changes to 'discovered'

### Selecting URLs
1. Navigate to URLs list page
2. Find parent URL (has "Parent" badge)
3. Click "View URLs" action link
4. Review discovered URLs with checkboxes
5. Select/deselect URLs to process
6. Click "Process Selected URLs"
7. Child URL records created for selected URLs

### Processing Child URLs
1. Child URLs appear in URLs list with `status='queued'`
2. URL watcher processes them normally
3. Content fetched and injected into RAG
4. Status changes to 'ingested'

## Configuration

### URL Discovery Settings
Set via environment variables in url_discovery.py:
- `max_depth`: How deep to crawl (default: 2)
- `max_urls`: Maximum URLs to discover (default: 100)
- `timeout`: HTTP request timeout (default: 30 seconds)

### URL Watcher Settings
Already configured via environment:
- `SLEEP_SECONDS`: Polling interval (default: 5)
- `BATCH_SIZE`: URLs per batch (default: 10)

## Example Use Cases

### Documentation Sites
- ReadTheDocs (e.g., Panda WMS docs)
- Sphinx-based documentation
- MkDocs sites
- GitBook sites

### Multi-Page Content
- Blog post series
- Tutorial sections
- API reference pages
- Wiki pages

## Technical Notes

### Discovery Strategies
1. **Documentation Detection**: Automatically uses specialized discovery for URLs containing "docs", "documentation", or "readthedocs"
2. **Navigation Parsing**: Looks for common navigation elements (`.toctree`, `.md-nav`, etc.)
3. **Link Extraction**: Falls back to all same-domain links if no navigation found
4. **Title Extraction**: Uses `<title>`, `<h1>`, or URL path as fallback

### Data Flow
```
Admin UI (urls_add.html)
  ↓ is_parent=true
Database (urls table)
  ↓ status='queued'
URL Watcher (url_watcher.py)
  ↓ check_if_parent_url()
URL Discovery (url_discovery.py)
  ↓ discover_urls_quick()
Database (discovered_urls JSONB)
  ↓ discovery_status='discovered'
Admin UI (urls_discover.html)
  ↓ user selection
Database (child URL records)
  ↓ status='queued'
URL Watcher (normal processing)
  ↓ fetch + inject
RAG Documents
```

## Error Handling

### Discovery Failures
- HTTP errors: Logged, status set to 'failed'
- Parse errors: Logged, returns empty list
- Timeout: Configurable per request

### Selection Validation
- Parent URL existence checked
- is_parent flag validated
- Duplicate URL detection (skips existing)

## Future Enhancements

### Potential Improvements
1. **Incremental Discovery**: Discover in batches for very large sites
2. **URL Patterns**: Allow regex patterns for inclusion/exclusion
3. **Priority Scoring**: Rank discovered URLs by importance
4. **Depth Visualization**: Show tree view of URL hierarchy
5. **Automatic Re-discovery**: Periodically check for new pages
6. **Sitemap.xml Support**: Parse XML sitemaps directly
7. **Robots.txt Respect**: Honor crawl rules

### Performance Optimizations
1. **Parallel Discovery**: Fetch multiple pages concurrently
2. **Caching**: Cache discovered URLs for faster re-processing
3. **Rate Limiting**: Respect server load
4. **Incremental Processing**: Process selected URLs immediately

## Testing Checklist

- [ ] Add parent URL via UI
- [ ] Verify discovery runs automatically
- [ ] Check discovered_urls JSONB structure
- [ ] View discovered URLs in UI
- [ ] Select/deselect URLs
- [ ] Save selection and create child URLs
- [ ] Verify child URLs are processed
- [ ] Test with ReadTheDocs site
- [ ] Test with regular multi-page site
- [ ] Test error cases (invalid URL, timeout, etc.)
- [ ] Verify CASCADE delete for parent-child relationships

## Migration Instructions

### To Apply Changes
1. Run database migration: `alembic upgrade head`
2. Restart backend services: `supervisorctl restart all`
3. Install Python dependencies if missing:
   ```bash
   pip install beautifulsoup4 requests
   ```

### Rollback
If needed, run: `alembic downgrade -1`

This will:
- Remove new columns from urls table
- Drop indexes
- Preserve existing URL data (is_parent will be null/false)
