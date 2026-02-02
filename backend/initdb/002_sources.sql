CREATE TABLE IF NOT EXISTS sources (
  id UUID PRIMARY KEY,
  scope TEXT NOT NULL CHECK (scope IN ('global','private')),
  user_id TEXT NULL,
  url TEXT NOT NULL,
  url_sha256 TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('queued','fetching','ingested','failed')),
  error TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  fetched_at TIMESTAMPTZ NULL,
  content_sha256 TEXT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sources_scope_user_urlsha
  ON sources(scope, user_id, url_sha256);

CREATE INDEX IF NOT EXISTS ix_sources_scope_user_status
  ON sources(scope, user_id, status, updated_at DESC);
