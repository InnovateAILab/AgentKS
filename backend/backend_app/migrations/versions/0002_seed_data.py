"""seed data migration

Revision ID: 0002_seed_data
Revises: 0001_initial
Create Date: 2026-02-02 00:05:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002_seed_data'
down_revision = '0001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Idempotent seed data for dev / demo purposes
    op.execute('''
INSERT INTO rag_groups (id,name,scope,owner,description,embed_model,doc_count)
VALUES
  ('rg_global','global','global','', 'Global public collection','nomic-embed-text',2)
ON CONFLICT (id) DO NOTHING;

INSERT INTO rag_groups (id,name,scope,owner,description,embed_model,doc_count)
VALUES
  ('rg_private_user123','private:user123','private','user123','Private collection for user123','nomic-embed-text',1)
ON CONFLICT (id) DO NOTHING;

INSERT INTO urls (id,url,scope,tags,status)
VALUES
  ('u1','https://example.com/doc1','global','["physics","note"]','ingested'),
  ('u2','https://example.com/doc2','private','["personal"]','queued')
ON CONFLICT (id) DO NOTHING;

INSERT INTO rag_group_urls (rag_group_id,url_id)
VALUES
  ('rg_global','u1'),
  ('rg_global','u2'),
  ('rg_private_user123','u2')
ON CONFLICT (rag_group_id,url_id) DO NOTHING;

INSERT INTO rag_documents (id,rag_group_id,url_id,title,content,content_hash,metadata)
VALUES
  ('d1','rg_global','u1','Doc 1','This is example content for doc1.','hash1','{"source":"example.com"}'),
  ('d2','rg_global','u2','Doc 2','Example content for doc2.','hash2','{"source":"example.com"}'),
  ('d3','rg_private_user123','u2','Private Doc','Private content sample.','hash3','{"owner":"user123"}')
ON CONFLICT (id) DO NOTHING;

INSERT INTO mcps (id,name,endpoint,kind,metadata,tags,status)
VALUES
  ('m1','mcp-search','http://mcp:8080','http','{}','["tools"]','enabled')
ON CONFLICT (id) DO NOTHING;

INSERT INTO tools (id,name,kind,mcp_id,metadata,tags)
VALUES
  ('t1','search','http','m1','{}','["search"]')
ON CONFLICT (id) DO NOTHING;
''')


def downgrade() -> None:
    # Remove seeded rows (safe for demo data)
    op.execute('''
DELETE FROM tools WHERE id IN ('t1');
DELETE FROM mcps WHERE id IN ('m1');
DELETE FROM rag_documents WHERE id IN ('d1','d2','d3');
DELETE FROM rag_group_urls WHERE (rag_group_id,url_id) IN (('rg_global','u1'),('rg_global','u2'),('rg_private_user123','u2'));
DELETE FROM urls WHERE id IN ('u1','u2');
DELETE FROM rag_groups WHERE id IN ('rg_global','rg_private_user123');
''')
