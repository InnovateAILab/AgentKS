# database structure (revised)

This folder contains initialization SQL executed during Postgres first-start. The schema below
is designed for two RAG ingestion flows:

- Admin-driven URL ingestion: admins add entries to `urls`; a worker/daemon fetches the URL
  content, producing `rag_documents` and linking them to `rag_groups`.
- API-driven/injected content: services can create `rag_documents` directly or create/update
  `rag_groups` via API.

Summary of the redesigned tables

- `urls` — canonical list of URLs to fetch and their state (status, last_fetched_at, last_error).
- `rag_groups` — logical collections/indices for RAG (e.g. `global`, or `private:user123`).
- `rag_group_urls` — many-to-many link table between `rag_groups` and `urls`.
- `rag_documents` — extracted or injected content items belonging to a `rag_group` (optionally tied to a URL).
- `mcps` — registered MCP endpoints (microservice/process endpoints that provide tools/data).
- `tools` — catalog of tools discovered from MCPs or registered manually.
- `tool_runs` — execution history for tools (input, output, status, timestamps).

Key design choices and rationale

- Normalized relationships: separating `rag_groups` and `urls` via an association table removes duplication
  and allows the same URL to be indexed by multiple rag collections.
- Use JSONB for flexible fields (`tags`, `metadata`) so the schema can evolve without immediate migrations.
- Add sensible defaults and timestamps to support ordering and simple operational queries.

Verification and example queries

List all tables (inside postgres container):

```bash
docker compose exec postgres psql -U "${AK_POSTGRES_USER}" -d "${AK_POSTGRES_DB}" -c "\dt"
```

Check counts and sample rows:

```bash
docker compose exec postgres psql -U "${AK_POSTGRES_USER}" -d "${AK_POSTGRES_DB}" -c "SELECT count(*) FROM urls;"
docker compose exec postgres psql -U "${AK_POSTGRES_USER}" -d "${AK_POSTGRES_DB}" -c "SELECT * FROM rag_groups LIMIT 5;"
```

Migration notes

- These init SQL files are executed only when Postgres initializes a new data directory. If you already
  have an existing DB volume, the files are not re-run. To apply the schema to an existing DB:
  - Run the SQL manually via `psql` or a migration tool, or
  - Remove the DB volume (warning: this deletes data) and restart the stack so the init scripts run.

Recommended next steps

- Move any demo INSERTs from Python startup code into a `02_seed_admin_data.sql` file in this folder so
  the schema and demo data are provisioned consistently.
- For long-term schema evolution, use a migration tool such as Alembic to create and apply incremental
  migrations instead of editing these init files.

If you'd like, I can generate a `02_seed_admin_data.sql` with example seed rows, and add a short
Alembic scaffold to the repo with an initial migration that mirrors the SQL in `01_create_admin_tables.sql`.
# database structure

## RAG
The rag content can commes from two parts: (1) admin add urls to the urls table, a daemon fetch the url contents and injects/updates the rag contents; (2) a rest api add rag contents 
### 1. rag groups
for url injection
primary key: (name, url), with status (queued, fetched, injected, failed, toremove, removed, updated_at, description)
admin can add/remove urls to this table. a daemon will periodly check the urls and update contents to the rag table. if remove a url, corresponding content in rag will be removed.

for rest api injection
primary key (name, function_name, )

### rag contents
rag information, with a foreign key to the urls table

## 2. tools
### tools table
### tools run table

## 3. MCP
### urls
admin can add/remove mcp server urls. a daemon agent will periodly check the urls and update the mcp tools to tools table.
