CREATE TABLE IF NOT EXISTS tools (
  id UUID PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  description TEXT NOT NULL,

  -- searxng_search / cds_search / arxiv_search / inspirehep_search / http_json / mcp_tool
  type TEXT NOT NULL,

  config JSONB NOT NULL,
  enabled BOOLEAN NOT NULL DEFAULT TRUE,

  scope TEXT NOT NULL CHECK (scope IN ('global','admin_only')),

  provider TEXT NOT NULL DEFAULT 'native',
  mcp_server TEXT NULL,
  mcp_tool TEXT NULL,

  created_by TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_runs (
  id UUID PRIMARY KEY,
  tool_id UUID NOT NULL REFERENCES tools(id),
  user_id TEXT NOT NULL,
  request JSONB NOT NULL,
  response JSONB NULL,
  status TEXT NOT NULL CHECK (status IN ('ok','error','skipped')),
  error TEXT NULL,
  latency_ms INT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_tool_runs_user_time ON tool_runs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_tools_enabled_scope ON tools(enabled, scope);
