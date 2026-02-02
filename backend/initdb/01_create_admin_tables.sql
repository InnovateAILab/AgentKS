-- Create admin UI tables. These are executed by Postgres on first init via the docker-compose mount of ./backend/initdb

CREATE TABLE IF NOT EXISTS urls (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    scope TEXT,
    tags JSONB,
    status TEXT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mcps (
    id TEXT PRIMARY KEY,
    name TEXT,
    endpoint TEXT,
    kind TEXT,
    tags JSONB,
    status TEXT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rags (
    id TEXT PRIMARY KEY,
    name TEXT,
    scope TEXT,
    owner TEXT,
    doc_count INTEGER,
    embed_model TEXT,
    updated_at TIMESTAMP
);
