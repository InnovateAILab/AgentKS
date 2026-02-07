"""LLM Management Module

This module provides database-driven LLM management with automatic fallback support.

Main Functions:
    - get_llm(llm_id=None, scope="global"): Get a specific LLM or the default
    - get_llm_with_fallback(scope="global"): Get LLM with automatic fallback on failure
    - get_llms_from_db(enabled_only=True, scope="global"): List all LLMs from database

Usage Examples:
    # Get the default LLM
    llm = get_llm()
    
    # Get a specific LLM by ID
    llm = get_llm(llm_id="abc-123-def-456")
    
    # Get LLM with automatic fallback (recommended for production)
    llm = get_llm_with_fallback()
    
    # List all enabled LLMs
    llms = get_llms_from_db()

Legacy functions (deprecated):
    - get_openai_llm()
    - get_anthropic_llm()
    - get_google_llm()
    - get_mixtral_fireworks()
    - get_ollama_llm()
"""

import os
import json
from functools import lru_cache
from typing import Optional, Any, List, Dict
from urllib.parse import urlparse

import boto3
import httpx
import psycopg
import structlog
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import BedrockChat, ChatFireworks
from langchain_community.chat_models.ollama import ChatOllama
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

logger = structlog.get_logger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required for LLM management")
PG_DSN = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")


# =========================
# Database helpers
# =========================
def db_exec(query: str, params: tuple = ()):
    """Execute a database query and return results."""
    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            try:
                return cur.fetchall()
            except psycopg.ProgrammingError:
                return None


def resolve_auth_meta(auth_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve environment variable references in auth metadata.
    
    Keys ending with '_env' are replaced with their environment variable values.
    Example: {"api_key_env": "OPENAI_API_KEY"} -> {"api_key": "sk-..."}
    """
    resolved = {}
    for key, value in auth_meta.items():
        if key.endswith('_env'):
            # Remove _env suffix and resolve from environment
            env_var_name = value
            resolved_key = key[:-4]  # Remove '_env' suffix
            resolved[resolved_key] = os.getenv(env_var_name, "")
        else:
            resolved[key] = value
    return resolved


def get_llms_from_db(enabled_only: bool = True, scope: str = "global") -> List[Dict[str, Any]]:
    """Load LLM configurations from database.
    
    Args:
        enabled_only: If True, only return enabled LLMs
        scope: Scope filter (default: 'global')
    
    Returns:
        List of LLM configuration dictionaries, ordered by priority
    """
    query = """
        SELECT id, name, provider, model_name, description, 
               auth_meta, config, enabled, is_default, priority, scope
        FROM llms
        WHERE scope = %s
    """
    params = [scope]
    
    if enabled_only:
        query += " AND enabled = true"
    
    query += " ORDER BY priority ASC"
    
    rows = db_exec(query, tuple(params))
    if not rows:
        logger.warning("No LLMs found in database", enabled_only=enabled_only, scope=scope)
        return []
    
    llms = []
    for row in rows:
        llms.append({
            'id': row[0],
            'name': row[1],
            'provider': row[2],
            'model_name': row[3],
            'description': row[4],
            'auth_meta': row[5] if isinstance(row[5], dict) else json.loads(row[5] or '{}'),
            'config': row[6] if isinstance(row[6], dict) else json.loads(row[6] or '{}'),
            'enabled': row[7],
            'is_default': row[8],
            'priority': row[9],
            'scope': row[10]
        })
    
    return llms


def get_default_llm_config(scope: str = "global") -> Optional[Dict[str, Any]]:
    """Get the default LLM configuration from database."""
    query = """
        SELECT id, name, provider, model_name, description, 
               auth_meta, config, enabled, is_default, priority, scope
        FROM llms
        WHERE scope = %s AND is_default = true AND enabled = true
        LIMIT 1
    """
    rows = db_exec(query, (scope,))
    if not rows:
        logger.warning("No default LLM found in database", scope=scope)
        return None
    
    row = rows[0]
    return {
        'id': row[0],
        'name': row[1],
        'provider': row[2],
        'model_name': row[3],
        'description': row[4],
        'auth_meta': row[5] if isinstance(row[5], dict) else json.loads(row[5] or '{}'),
        'config': row[6] if isinstance(row[6], dict) else json.loads(row[6] or '{}'),
        'enabled': row[7],
        'is_default': row[8],
        'priority': row[9],
        'scope': row[10]
    }


# =========================
# LLM instance creation
# =========================
def create_llm_instance(llm_config: Dict[str, Any]) -> Any:
    """Create a LangChain LLM instance from database configuration.
    
    Args:
        llm_config: LLM configuration dictionary from database
        
    Returns:
        LangChain chat model instance
    """
    provider = llm_config['provider']
    model_name = llm_config['model_name']
    auth_meta = llm_config['auth_meta']
    config = llm_config['config']
    
    # Resolve environment variable references
    resolved_auth = resolve_auth_meta(auth_meta)
    
    # Get proxy if configured
    proxy_url = os.getenv("PROXY_URL")
    http_client = _get_httpx_client(proxy_url) if proxy_url else None
    
    try:
        if provider == 'openai':
            return ChatOpenAI(
                http_client=http_client,
                model=model_name,
                api_key=resolved_auth.get('api_key'),
                base_url=resolved_auth.get('base_url'),
                temperature=config.get('temperature', 0),
                max_tokens=config.get('max_tokens'),
            )
        
        elif provider == 'azure':
            return AzureChatOpenAI(
                http_client=http_client,
                deployment_name=resolved_auth.get('deployment_name', model_name),
                azure_endpoint=resolved_auth.get('endpoint'),
                openai_api_version=resolved_auth.get('api_version'),
                openai_api_key=resolved_auth.get('api_key'),
                temperature=config.get('temperature', 0),
            )
        
        elif provider == 'anthropic':
            return ChatAnthropic(
                model_name=model_name,
                api_key=resolved_auth.get('api_key'),
                max_tokens_to_sample=config.get('max_tokens_to_sample', 2000),
                temperature=config.get('temperature', 0),
            )
        
        elif provider == 'bedrock':
            client = boto3.client(
                "bedrock-runtime",
                region_name=resolved_auth.get('region', 'us-west-2'),
                aws_access_key_id=resolved_auth.get('aws_access_key_id'),
                aws_secret_access_key=resolved_auth.get('aws_secret_access_key'),
            )
            return BedrockChat(model_id=model_name, client=client)
        
        elif provider == 'google':
            return ChatVertexAI(
                model_name=model_name,
                project=resolved_auth.get('project_id'),
                location=resolved_auth.get('location', 'us-central1'),
                temperature=config.get('temperature', 0),
                convert_system_message_to_human=config.get('convert_system_message_to_human', True),
                streaming=config.get('streaming', True),
            )
        
        elif provider == 'fireworks':
            return ChatFireworks(
                model=model_name,
                api_key=resolved_auth.get('api_key'),
                temperature=config.get('temperature', 0),
            )
        
        elif provider == 'ollama':
            return ChatOllama(
                model=model_name,
                base_url=resolved_auth.get('base_url', 'http://localhost:11434'),
                temperature=config.get('temperature', 0),
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
    except Exception as e:
        logger.error("Failed to create LLM instance", 
                    provider=provider, model=model_name, error=str(e))
        raise


def get_llm(llm_id: Optional[str] = None, scope: str = "global") -> Any:
    """Get a LangChain LLM instance.
    
    Args:
        llm_id: Specific LLM ID to load. If None, uses default LLM.
        scope: Scope to search in (default: 'global')
        
    Returns:
        LangChain chat model instance
        
    Raises:
        RuntimeError: If no LLM is found or configured
    """
    if llm_id:
        # Load specific LLM by ID
        query = """
            SELECT id, name, provider, model_name, description, 
                   auth_meta, config, enabled, is_default, priority, scope
            FROM llms
            WHERE id = %s AND enabled = true
            LIMIT 1
        """
        rows = db_exec(query, (llm_id,))
        if not rows:
            raise RuntimeError(f"LLM with id '{llm_id}' not found or not enabled")
        
        row = rows[0]
        llm_config = {
            'id': row[0],
            'name': row[1],
            'provider': row[2],
            'model_name': row[3],
            'description': row[4],
            'auth_meta': row[5] if isinstance(row[5], dict) else json.loads(row[5] or '{}'),
            'config': row[6] if isinstance(row[6], dict) else json.loads(row[6] or '{}'),
            'enabled': row[7],
            'is_default': row[8],
            'priority': row[9],
            'scope': row[10]
        }
    else:
        # Use default LLM
        llm_config = get_default_llm_config(scope)
        if not llm_config:
            raise RuntimeError(f"No default LLM configured for scope '{scope}'")
    
    logger.info("Loading LLM", llm_id=llm_config['id'], name=llm_config['name'], 
                provider=llm_config['provider'], model=llm_config['model_name'])
    
    return create_llm_instance(llm_config)


def get_llm_with_fallback(scope: str = "global") -> Any:
    """Get a LangChain LLM instance with automatic fallback.
    
    Tries LLMs in priority order until one succeeds.
    
    Args:
        scope: Scope to search in (default: 'global')
        
    Returns:
        LangChain chat model instance
        
    Raises:
        RuntimeError: If all LLMs fail to initialize
    """
    llms = get_llms_from_db(enabled_only=True, scope=scope)
    if not llms:
        raise RuntimeError(f"No enabled LLMs found for scope '{scope}'")
    
    last_error = None
    for llm_config in llms:
        try:
            logger.info("Attempting to load LLM", 
                       name=llm_config['name'], 
                       provider=llm_config['provider'],
                       priority=llm_config['priority'])
            return create_llm_instance(llm_config)
        except Exception as e:
            logger.warning("Failed to load LLM, trying next", 
                          name=llm_config['name'], 
                          error=str(e))
            last_error = e
            continue
    
    raise RuntimeError(f"All LLMs failed to initialize. Last error: {last_error}")


# =========================
# Legacy functions (for backward compatibility)
# =========================


# =========================
# Legacy functions (for backward compatibility)
# =========================


# Internal helper to create/reuse an httpx.AsyncClient for downstream SDKs.
@lru_cache(maxsize=4)
def _get_httpx_client(proxy_url: Optional[str]) -> Optional[httpx.AsyncClient]:
    if not proxy_url:
        return None
    parsed = urlparse(proxy_url)
    if not (parsed.scheme and parsed.netloc):
        logger.warning("Invalid PROXY_URL; ignoring proxy and continuing without it.")
        return None
    # Use a persistent AsyncClient so SDKs can reuse connections.
    return httpx.AsyncClient(proxies=proxy_url)