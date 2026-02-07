import os
from functools import lru_cache
from typing import Optional, Any
from urllib.parse import urlparse

import boto3
import httpx
import structlog
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import BedrockChat, ChatFireworks
from langchain_community.chat_models.ollama import ChatOllama
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

logger = structlog.get_logger(__name__)


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


@lru_cache(maxsize=4)
def get_openai_llm(model: str = "gpt-3.5-turbo", azure: bool = False) -> Any:
    """Return a configured OpenAI or Azure OpenAI chat model.

    The function prefers the regular OpenAI provider but will fall back to
    Azure if requested or if instantiation fails and Azure credentials are
    available. HTTP proxy (PROXY_URL) is supported via an AsyncClient.
    """
    proxy_url = os.getenv("PROXY_URL")
    http_client = _get_httpx_client(proxy_url)

    if not azure:
        try:
            return ChatOpenAI(http_client=http_client, model=model, temperature=0)
        except Exception as e:
            logger.warning("ChatOpenAI instantiation failed, attempting AzureChatOpenAI", exc_info=e)

    # Azure path (either requested or fallback). Validate required env vars.
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
    azure_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not all([azure_deployment, azure_endpoint, azure_version, azure_key]):
        raise RuntimeError("Azure OpenAI requested but AZURE_* env vars are not fully configured")
    return AzureChatOpenAI(
        http_client=http_client,
        temperature=0,
        deployment_name=azure_deployment,
        azure_endpoint=azure_endpoint,
        openai_api_version=azure_version,
        openai_api_key=azure_key,
    )


@lru_cache(maxsize=2)
def get_anthropic_llm(bedrock: bool = False) -> Any:
    """Return an Anthropic (Claude) LLM, optionally via Bedrock."""
    if bedrock:
        client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-west-2"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        return BedrockChat(model_id="anthropic.claude-v2", client=client)

    model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    return ChatAnthropic(model_name=model_name, max_tokens_to_sample=2000, temperature=0)


@lru_cache(maxsize=1)
def get_google_llm() -> Any:
    return ChatVertexAI(model_name="gemini-pro", convert_system_message_to_human=True, streaming=True)


@lru_cache(maxsize=1)
def get_mixtral_fireworks() -> Any:
    return ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")


@lru_cache(maxsize=1)
def get_ollama_llm() -> Any:
    model_name = os.environ.get("OLLAMA_MODEL", "llama2")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model_name, base_url=ollama_base_url)