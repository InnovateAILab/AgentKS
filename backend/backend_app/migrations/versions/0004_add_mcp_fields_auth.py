"""add mcp description/resource/context/auth

Revision ID: 0004_add_mcp_fields_auth
Revises: 0003_admin_actions
Create Date: 2026-02-03 00:30:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0004_add_mcp_fields_auth'
down_revision = '0003_admin_actions'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add descriptive columns to mcps: description, resource, context
    op.add_column('mcps', sa.Column('description', sa.Text(), nullable=True))
    op.add_column('mcps', sa.Column('resource', sa.Text(), nullable=True))
    op.add_column('mcps', sa.Column('context', sa.Text(), nullable=True))
    # Add a JSONB column to store auth/header configuration for the MCP (optional)
    op.add_column('mcps', sa.Column('auth', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('mcps', 'auth')
    op.drop_column('mcps', 'context')
    op.drop_column('mcps', 'resource')
    op.drop_column('mcps', 'description')
