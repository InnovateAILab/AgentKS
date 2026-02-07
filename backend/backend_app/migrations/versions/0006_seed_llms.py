"""seed llms table with default LLM configurations

Revision ID: 0006_seed_llms
Revises: 0005_add_llms_table
Create Date: 2026-02-04 00:10:00.000000
"""
from alembic import op
import sqlalchemy as sa
import uuid

# revision identifiers, used by Alembic.
revision = '0006_seed_llms'
down_revision = '0005_add_llms_table'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Seed LLMs based on llms_backup.py configurations
    llms_data = [
        # OpenAI
        {
            'id': str(uuid.uuid4()),
            'name': 'OpenAI GPT-4',
            'provider': 'openai',
            'model_name': 'gpt-4',
            'description': 'OpenAI GPT-4 - Most capable model for complex tasks',
            'auth_meta': '{"api_key_env": "OPENAI_API_KEY", "base_url_env": "OPENAI_API_BASE"}',
            'config': '{"temperature": 0, "max_tokens": 2000}',
            'enabled': True,
            'is_default': False,
            'priority': 10,
            'scope': 'global'
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'OpenAI GPT-3.5 Turbo',
            'provider': 'openai',
            'model_name': 'gpt-3.5-turbo',
            'description': 'OpenAI GPT-3.5 Turbo - Fast and cost-effective',
            'auth_meta': '{"api_key_env": "OPENAI_API_KEY", "base_url_env": "OPENAI_API_BASE"}',
            'config': '{"temperature": 0}',
            'enabled': True,
            'is_default': False,
            'priority': 20,
            'scope': 'global'
        },
        # Azure OpenAI
        {
            'id': str(uuid.uuid4()),
            'name': 'Azure OpenAI',
            'provider': 'azure',
            'model_name': 'gpt-4',
            'description': 'Azure OpenAI - Enterprise-grade OpenAI deployment',
            'auth_meta': '{"api_key_env": "AZURE_OPENAI_API_KEY", "endpoint_env": "AZURE_OPENAI_API_BASE", "deployment_name_env": "AZURE_OPENAI_DEPLOYMENT_NAME", "api_version_env": "AZURE_OPENAI_API_VERSION"}',
            'config': '{"temperature": 0}',
            'enabled': False,
            'is_default': False,
            'priority': 30,
            'scope': 'global'
        },
        # Anthropic Claude
        {
            'id': str(uuid.uuid4()),
            'name': 'Claude 3 Haiku',
            'provider': 'anthropic',
            'model_name': 'claude-3-haiku-20240307',
            'description': 'Anthropic Claude 3 Haiku - Fast and efficient',
            'auth_meta': '{"api_key_env": "ANTHROPIC_API_KEY"}',
            'config': '{"temperature": 0, "max_tokens_to_sample": 2000}',
            'enabled': False,
            'is_default': False,
            'priority': 40,
            'scope': 'global'
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Claude 3 Sonnet',
            'provider': 'anthropic',
            'model_name': 'claude-3-sonnet-20240229',
            'description': 'Anthropic Claude 3 Sonnet - Balanced performance',
            'auth_meta': '{"api_key_env": "ANTHROPIC_API_KEY"}',
            'config': '{"temperature": 0, "max_tokens_to_sample": 2000}',
            'enabled': False,
            'is_default': False,
            'priority': 41,
            'scope': 'global'
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Claude 3 Opus',
            'provider': 'anthropic',
            'model_name': 'claude-3-opus-20240229',
            'description': 'Anthropic Claude 3 Opus - Most capable Claude model',
            'auth_meta': '{"api_key_env": "ANTHROPIC_API_KEY"}',
            'config': '{"temperature": 0, "max_tokens_to_sample": 4000}',
            'enabled': False,
            'is_default': False,
            'priority': 42,
            'scope': 'global'
        },
        # Bedrock Claude
        {
            'id': str(uuid.uuid4()),
            'name': 'Bedrock Claude v2',
            'provider': 'bedrock',
            'model_name': 'anthropic.claude-v2',
            'description': 'AWS Bedrock Claude v2',
            'auth_meta': '{"aws_access_key_id_env": "AWS_ACCESS_KEY_ID", "aws_secret_access_key_env": "AWS_SECRET_ACCESS_KEY", "region_env": "AWS_REGION"}',
            'config': '{"temperature": 0}',
            'enabled': False,
            'is_default': False,
            'priority': 50,
            'scope': 'global'
        },
        # Google
        {
            'id': str(uuid.uuid4()),
            'name': 'Google Gemini Pro',
            'provider': 'google',
            'model_name': 'gemini-pro',
            'description': 'Google Gemini Pro - Multimodal capabilities',
            'auth_meta': '{"project_id_env": "GOOGLE_CLOUD_PROJECT", "location": "us-central1"}',
            'config': '{"temperature": 0, "convert_system_message_to_human": true, "streaming": true}',
            'enabled': False,
            'is_default': False,
            'priority': 60,
            'scope': 'global'
        },
        # Fireworks
        {
            'id': str(uuid.uuid4()),
            'name': 'Mixtral 8x7B (Fireworks)',
            'provider': 'fireworks',
            'model_name': 'accounts/fireworks/models/mixtral-8x7b-instruct',
            'description': 'Mixtral 8x7B Instruct via Fireworks AI',
            'auth_meta': '{"api_key_env": "FIREWORKS_API_KEY"}',
            'config': '{"temperature": 0}',
            'enabled': False,
            'is_default': False,
            'priority': 70,
            'scope': 'global'
        },
        # Ollama
        {
            'id': str(uuid.uuid4()),
            'name': 'Ollama Llama2',
            'provider': 'ollama',
            'model_name': 'llama2:7b',
            'description': 'Local Llama2 via Ollama',
            'auth_meta': '{"base_url_env": "OLLAMA_BASE_URL"}',
            'config': '{"temperature": 0}',
            'enabled': True,
            'is_default': True,
            'priority': 80,
            'scope': 'global'
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Ollama Llama3',
            'provider': 'ollama',
            'model_name': 'llama3',
            'description': 'Local Llama3 via Ollama',
            'auth_meta': '{"base_url_env": "OLLAMA_BASE_URL"}',
            'config': '{"temperature": 0}',
            'enabled': False,
            'is_default': False,
            'priority': 81,
            'scope': 'global'
        }
    ]
    
    conn = op.get_bind()
    for llm in llms_data:
        conn.execute(sa.text('''
            INSERT INTO llms (id, name, provider, model_name, description, auth_meta, config, enabled, is_default, priority, scope)
            VALUES (:id, :name, :provider, :model_name, :description, :auth_meta::jsonb, :config::jsonb, :enabled, :is_default, :priority, :scope)
        '''), llm)


def downgrade() -> None:
    op.execute('DELETE FROM llms')
