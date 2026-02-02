"""create admin_actions table

Revision ID: 0003_admin_actions
Revises: 0002_seed_data
Create Date: 2026-02-02 00:10:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0003_admin_actions'
down_revision = '0002_seed_data'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('''
CREATE TABLE IF NOT EXISTS admin_actions (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    actor TEXT,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_admin_actions_actor ON admin_actions(actor);
''')


def downgrade() -> None:
    op.execute('''
DROP INDEX IF EXISTS idx_admin_actions_actor;
DROP TABLE IF EXISTS admin_actions;
''')
