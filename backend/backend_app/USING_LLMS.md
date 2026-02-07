# Using the LLM Management System

## Quick Start

The LLM management system provides three main functions for loading LLMs:

### 1. Get Default LLM

```python
from app.llms import get_llm

# Get the default LLM from database
llm = get_llm()

# Use it
response = llm.invoke("Hello, how are you?")
print(response.content)
```

### 2. Get Specific LLM by ID

```python
from app.llms import get_llm

# Get a specific LLM by its database ID
llm = get_llm(llm_id="abc-123-def-456")

# Use it
response = llm.invoke("What is Python?")
print(response.content)
```

### 3. Get LLM with Automatic Fallback (Recommended)

```python
from app.llms import get_llm_with_fallback

# Get LLM with automatic fallback to next priority if one fails
llm = get_llm_with_fallback()

# Use it
response = llm.invoke("Explain machine learning")
print(response.content)
```

## How It Works

### Database-Driven Configuration

All LLM configurations are stored in the `llms` table:

- **Enabled/Disabled**: Toggle LLM availability without deleting configuration
- **Priority**: Lower priority number = tried first in fallback chain
- **Default**: One LLM can be marked as default per scope
- **Auth Metadata**: JSON with environment variable references (keys ending in `_env`)
- **Config**: JSON with model parameters (temperature, max_tokens, etc.)

### Environment Variable Resolution

Auth metadata uses environment variable references for security:

```json
{
  "api_key_env": "OPENAI_API_KEY",
  "base_url_env": "OPENAI_API_BASE"
}
```

At runtime, the system resolves these to:

```json
{
  "api_key": "sk-...",
  "base_url": "https://api.openai.com/v1"
}
```

### Fallback Chain Example

Given these LLMs in the database:

| Name | Priority | Enabled | Is Default |
|------|----------|---------|------------|
| Ollama Llama2 | 80 | ✓ | ✓ |
| OpenAI GPT-4 | 10 | ✓ | ✗ |
| OpenAI GPT-3.5 | 20 | ✓ | ✗ |
| Claude 3 Opus | 42 | ✗ | ✗ |

When you call `get_llm_with_fallback()`:

1. Tries OpenAI GPT-4 (priority 10) first
2. If that fails, tries OpenAI GPT-3.5 (priority 20)
3. If that fails, tries Ollama Llama2 (priority 80)
4. If all fail, raises RuntimeError

Note: Claude 3 Opus is skipped because it's disabled.

## Integration with main.py

The `backend_app/app/main.py` now loads LLMs from database:

```python
from .llms import get_llm, get_llm_with_fallback

# Try to load LLM from database, fallback to environment variables
try:
    llm = get_llm_with_fallback()
    print("✓ Loaded LLM from database with fallback support")
except Exception as e:
    print(f"⚠ Failed to load LLM from database: {e}")
    print(f"⚠ Falling back to environment-configured Ollama: {OLLAMA_CHAT_MODEL}")
    llm = ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
```

This approach ensures:
- ✅ Production uses database-managed LLMs with fallback
- ✅ Development still works if database isn't seeded
- ✅ No breaking changes to existing code

## Admin UI Usage

Manage LLMs through the web UI at `/admin/llms`:

1. **View all LLMs**: Filter by provider, enabled status, search by name
2. **Add new LLM**: Form with provider dropdown, auth metadata examples
3. **Edit LLM**: Update configuration, change priority, enable/disable
4. **Set default**: Mark one LLM as default (others auto-cleared)
5. **Toggle enabled**: Quick enable/disable without editing
6. **Bulk actions**: Enable, disable, or delete multiple LLMs

## Advanced Usage

### List All LLMs

```python
from app.llms import get_llms_from_db

# Get all enabled LLMs, ordered by priority
llms = get_llms_from_db(enabled_only=True, scope="global")

for llm_config in llms:
    print(f"{llm_config['name']}: {llm_config['provider']} - {llm_config['model_name']}")
    print(f"  Priority: {llm_config['priority']}")
    print(f"  Default: {llm_config['is_default']}")
```

### Get Default LLM Config

```python
from app.llms import get_default_llm_config

# Get default LLM configuration (doesn't create instance)
config = get_default_llm_config(scope="global")
print(f"Default LLM: {config['name']}")
print(f"Provider: {config['provider']}")
print(f"Model: {config['model_name']}")
```

### Create LLM Instance from Config

```python
from app.llms import get_llms_from_db, create_llm_instance

# Get all LLMs
llms = get_llms_from_db()

# Create instance for the first one
if llms:
    llm_instance = create_llm_instance(llms[0])
    response = llm_instance.invoke("Test message")
```

### Dynamic LLM Selection

```python
from app.llms import get_llms_from_db, create_llm_instance

def get_llm_by_provider(provider: str):
    """Get first enabled LLM from specific provider."""
    llms = get_llms_from_db(enabled_only=True)
    for llm_config in llms:
        if llm_config['provider'] == provider:
            return create_llm_instance(llm_config)
    raise ValueError(f"No enabled LLM found for provider: {provider}")

# Use it
openai_llm = get_llm_by_provider("openai")
anthropic_llm = get_llm_by_provider("anthropic")
```

## Migrating from Legacy Functions

Old code using legacy functions still works but shows deprecation warnings:

```python
# OLD (deprecated)
from app.llms import get_openai_llm, get_anthropic_llm, get_ollama_llm
llm = get_openai_llm()

# NEW (recommended)
from app.llms import get_llm, get_llm_with_fallback
llm = get_llm_with_fallback()
```

Benefits of new approach:
- ✅ No code changes needed to switch LLMs
- ✅ Automatic fallback on failure
- ✅ Centralized configuration via database
- ✅ Admin UI for non-technical users
- ✅ Audit trail of all changes

## Environment Variables Required

### For Database Connection

```bash
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname
```

### For LLM Authentication

Set the environment variables referenced in auth_meta:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_BASE=https://...
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2023-05-15

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# AWS Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-west-2

# Google Vertex AI
GOOGLE_CLOUD_PROJECT=my-project
# Also needs gcloud auth or service account key

# Fireworks
FIREWORKS_API_KEY=...

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

## Troubleshooting

### "No default LLM configured"

Run migrations to seed default LLMs:

```bash
cd backend/backend_app
alembic upgrade head
```

### "All LLMs failed to initialize"

Check that:
1. Environment variables for auth are set
2. At least one LLM is enabled in the database
3. Network connectivity to LLM providers

Enable debug logging:

```python
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG))
```

### Import errors in development

The lint errors for missing imports are expected in development environments without all dependencies installed. They don't affect runtime in Docker containers.

## Best Practices

1. **Use fallback in production**: `get_llm_with_fallback()` ensures resilience
2. **Set appropriate priorities**: Lower for production, higher for experimental
3. **Use environment variables**: Never hardcode API keys
4. **Test before enabling**: Add LLMs as disabled first, test, then enable
5. **Monitor costs**: Higher priority = used more often
6. **Document changes**: Admin UI includes audit logging

## Example: Multi-LLM Application

```python
from app.llms import get_llm, get_llm_with_fallback, get_llms_from_db

class MultiLLMService:
    def __init__(self):
        # Primary LLM with fallback
        self.primary_llm = get_llm_with_fallback()
        
    def chat(self, message: str) -> str:
        """Standard chat using default LLM."""
        response = self.primary_llm.invoke(message)
        return response.content
    
    def compare_responses(self, message: str, provider: str = None):
        """Get responses from multiple LLMs for comparison."""
        llms = get_llms_from_db(enabled_only=True)
        
        if provider:
            llms = [l for l in llms if l['provider'] == provider]
        
        responses = {}
        for llm_config in llms[:3]:  # Compare up to 3
            try:
                llm_instance = create_llm_instance(llm_config)
                resp = llm_instance.invoke(message)
                responses[llm_config['name']] = resp.content
            except Exception as e:
                responses[llm_config['name']] = f"Error: {e}"
        
        return responses

# Usage
service = MultiLLMService()

# Standard chat
answer = service.chat("What is Python?")

# Compare multiple models
comparisons = service.compare_responses("Explain quantum computing")
for model, response in comparisons.items():
    print(f"\n{model}:\n{response}")
```
