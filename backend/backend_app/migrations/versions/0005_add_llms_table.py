"""add llms table for LLM management

Revision ID: 0005_add_llms_table
Revises: 0004_add_mcp_fields_auth
Create Date: 2026-02-04 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0005_add_llms_table'
down_revision = '0004_add_mcp_fields_auth'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('''
-- LLMs table: registry of available language models
CREATE TABLE IF NOT EXISTS llms (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,  -- openai, anthropic, google, ollama, azure, bedrock, fireworks
    model_name TEXT NOT NULL,  -- gpt-4, claude-3-haiku, gemini-pro, llama2, etc.
    description TEXT,
    auth_meta JSONB DEFAULT '{}'::jsonb,  -- {"api_key": "env:OPENAI_API_KEY", "base_url": "...", etc.}
    config JSONB DEFAULT '{}'::jsonb,  -- {"temperature": 0, "max_tokens": 2000, etc.}
    enabled BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,
    priority INTEGER DEFAULT 100,  -- Lower number = higher priority in try list
    scope TEXT DEFAULT 'global',  -- 'global' or 'private:user_id'
    owner TEXT,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llms_enabled ON llms(enabled);
CREATE INDEX IF NOT EXISTS idx_llms_priority ON llms(priority);
CREATE INDEX IF NOT EXISTS idx_llms_scope ON llms(scope);
CREATE UNIQUE INDEX IF NOT EXISTS ux_llms_default_scope ON llms(is_default, scope) WHERE is_default = true;
    ''')


def downgrade() -> None:
    op.execute('DROP TABLE IF EXISTS llms CASCADE')
