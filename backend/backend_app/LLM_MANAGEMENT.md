# LLM Management System

## Overview

The LLM Management System provides a comprehensive database-backed solution for registering, configuring, and managing multiple Large Language Models (LLMs) through an admin web interface.

## Features

- ✅ **Database Registry**: Store LLM configurations in PostgreSQL with full CRUD operations
- ✅ **Multiple Providers**: Support for OpenAI, Azure, Anthropic, Bedrock, Google, Fireworks, Ollama
- ✅ **Enable/Disable**: Toggle LLM availability without deleting configurations
- ✅ **Priority System**: Control fallback order with numeric priority (lower = higher priority)
- ✅ **Default LLM**: Mark one LLM as default for each scope
- ✅ **Auth Metadata**: Store authentication configuration as JSON with environment variable references
- ✅ **Model Config**: Store model-specific parameters (temperature, max_tokens, etc.)
- ✅ **Bulk Actions**: Enable, disable, or delete multiple LLMs at once
- ✅ **Search & Filter**: Filter LLMs by provider, status, or search query
- ✅ **Audit Logging**: Track all LLM management actions

## Database Schema

### `llms` Table

```sql
CREATE TABLE llms (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    description TEXT,
    auth_meta JSONB DEFAULT '{}'::jsonb,
    config JSONB DEFAULT '{}'::jsonb,
    enabled BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,
    priority INTEGER DEFAULT 100,
    scope TEXT DEFAULT 'global',
    owner TEXT,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);
```

### Fields

- **id**: UUID primary key
- **name**: Human-readable display name (unique)
- **provider**: LLM provider (openai, anthropic, google, etc.)
- **model_name**: Actual model identifier (gpt-4, claude-3-opus, etc.)
- **description**: Optional description
- **auth_meta**: JSON object with authentication configuration
- **config**: JSON object with model parameters
- **enabled**: Whether the LLM is available for use
- **is_default**: Whether this is the default LLM (only one per scope)
- **priority**: Fallback order (lower number = higher priority)
- **scope**: Scope for multi-tenancy (default: 'global')
- **owner**: Optional owner identifier
- **created_at**: Creation timestamp
- **updated_at**: Last update timestamp

## Migrations

Two migrations are provided:

### 0005_add_llms_table.py
Creates the `llms` table with appropriate indexes and constraints.

### 0006_seed_llms.py
Seeds the database with default LLM configurations from `llms_backup.py`:
- OpenAI GPT-4 (default, priority 10)
- OpenAI GPT-3.5 Turbo (priority 20)
- Azure OpenAI (disabled, priority 30)
- Claude 3 Haiku, Sonnet, Opus (disabled, priorities 40-42)
- Bedrock Claude v2 (disabled, priority 50)
- Google Gemini Pro (disabled, priority 60)
- Mixtral 8x7B via Fireworks (disabled, priority 70)
- Ollama Llama2 (enabled, priority 80)
- Ollama Llama3 (disabled, priority 81)

## Auth Metadata Format

Auth metadata uses JSON with environment variable references. Keys ending with `_env` are resolved at runtime:

```json
{
  "api_key_env": "OPENAI_API_KEY",
  "base_url_env": "OPENAI_API_BASE"
}
```

### Common Examples

**OpenAI:**
```json
{"api_key_env": "OPENAI_API_KEY"}
```

**Azure OpenAI:**
```json
{
  "api_key_env": "AZURE_OPENAI_API_KEY",
  "endpoint_env": "AZURE_OPENAI_API_BASE",
  "deployment_name_env": "AZURE_OPENAI_DEPLOYMENT_NAME",
  "api_version_env": "AZURE_OPENAI_API_VERSION"
}
```

**Anthropic:**
```json
{"api_key_env": "ANTHROPIC_API_KEY"}
```

**Ollama:**
```json
{"base_url_env": "OLLAMA_BASE_URL"}
```

**AWS Bedrock:**
```json
{
  "aws_access_key_id_env": "AWS_ACCESS_KEY_ID",
  "aws_secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
  "region_env": "AWS_REGION"
}
```

## Config Format

Model configuration as JSON:

```json
{
  "temperature": 0,
  "max_tokens": 2000,
  "streaming": true
}
```

## Admin UI

### Routes

- `GET /admin/llms` - List all LLMs with filtering
- `GET /admin/llms/add` - Show form to add new LLM
- `POST /admin/llms/add` - Create new LLM
- `GET /admin/llms/{id}/edit` - Show form to edit LLM
- `POST /admin/llms/{id}/edit` - Update LLM
- `POST /admin/llms/{id}/delete` - Delete LLM
- `POST /admin/llms/{id}/toggle` - Toggle enabled/disabled
- `POST /admin/llms/{id}/set-default` - Set as default LLM
- `POST /admin/llms/bulk` - Bulk actions (enable, disable, delete)

### Features

1. **List View** (`llms_list.html`)
   - Search by name, model, or description
   - Filter by provider and enabled status
   - Sort by priority
   - Shows enabled status (clickable to toggle)
   - Shows default status with button to set as default
   - Bulk selection with checkboxes
   - Inline edit and delete actions

2. **Add Form** (`llms_add.html`)
   - Provider dropdown
   - Model name input
   - Description textarea
   - Auth metadata JSON editor with examples
   - Config JSON editor
   - Priority number input
   - Enabled checkbox
   - Default checkbox
   - Help text and examples

3. **Edit Form** (`llms_edit.html`)
   - All fields from add form, pre-filled
   - Delete button
   - Tips section

## Usage in Code

### Get Default LLM

```python
row = db_exec("SELECT * FROM llms WHERE is_default = true AND scope = 'global' AND enabled = true")
```

### Get Enabled LLMs by Priority

```python
rows = db_exec("SELECT * FROM llms WHERE enabled = true AND scope = 'global' ORDER BY priority ASC")
```

### Resolve Auth from Environment

```python
import os
import json

llm = {...}  # LLM record from database
auth_meta = llm['auth_meta']

resolved_auth = {}
for key, value in auth_meta.items():
    if key.endswith('_env'):
        # Resolve from environment
        env_var = value
        resolved_auth[key[:-4]] = os.getenv(env_var)
    else:
        resolved_auth[key] = value
```

## Priority System

The priority system enables fallback chains:

1. Lower priority number = tried first
2. If an LLM fails, try the next enabled LLM by priority
3. Example priority values:
   - 10: Primary production LLM
   - 20: Secondary fallback
   - 30-50: Tertiary options
   - 60+: Special-purpose or experimental

## Security Considerations

1. **Environment Variables**: Always use `_env` suffix keys to reference secrets
2. **Never Store Secrets**: Don't put API keys directly in auth_meta
3. **Scope Isolation**: Use `scope` field for multi-tenant isolation
4. **Audit Trail**: All actions are logged to `admin_actions` table

## Running Migrations

```bash
cd backend/backend_app
alembic upgrade head
```

This will:
1. Create the `llms` table (migration 0005)
2. Seed default LLMs (migration 0006)

## Testing

Access the admin UI at: `http://localhost:8000/admin/llms`

Default seed data includes:
- OpenAI GPT-4 (enabled)
- Ollama Llama2 (enabled, default)
- Various other providers (disabled by default)

## Integration with Existing Code

The LLM management system is designed to complement `llms_backup.py`. You can:

1. Query the database for enabled LLMs
2. Use priority ordering for fallback logic
3. Dynamically construct LangChain LLM instances from database records
4. Update the `get_*_llm()` functions to check the database first

## Future Enhancements

- [ ] Per-user LLM preferences (scope per user)
- [ ] Usage tracking and cost estimation
- [ ] Rate limiting configuration
- [ ] Model capability tags (vision, function-calling, etc.)
- [ ] A/B testing configuration
- [ ] Health check endpoints per LLM
- [ ] Automatic failover configuration
