"""Factory helpers for Agent Framework clients and MCP connectivity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from maf_graphrag.agents.config import get_agent_config
from maf_graphrag.agents.mcp_compat import ensure_mcp_client_compatibility

if TYPE_CHECKING:  # pragma: no cover - import guard for typing only
    from agent_framework import (
        MCPStreamableHTTPTool,
        SupportsChatGetResponse,
    )

logger = logging.getLogger(__name__)


def create_mcp_tool(mcp_url: str | None = None) -> MCPStreamableHTTPTool:
    """Create an ``MCPStreamableHTTPTool`` connected to the GraphRAG server."""
    from agent_framework import MCPStreamableHTTPTool

    config = get_agent_config()
    url_value = mcp_url or config.mcp_server_url
    url = str(url_value)

    # Normalize FastMCP streamable endpoint to /mcp for Agent Framework clients.
    if url.endswith("/sse"):
        url = url.replace("/sse", "/mcp")
    elif not url.endswith("/mcp"):
        url = url.rstrip("/") + "/mcp"

    ensure_mcp_client_compatibility()

    return MCPStreamableHTTPTool(
        name="graphrag",
        url=url,
        description="Query the GraphRAG knowledge graph for entity and thematic information",
    )


def create_client() -> SupportsChatGetResponse:
    """Create a llama.cpp-backed OpenAI-compatible chat client."""
    config = get_agent_config()

    from agent_framework.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(
        model=config.model,
        base_url=f"{config.router_base_url}/v1",
        api_key="llama.cpp",
    )


__all__ = ["create_mcp_tool", "create_client"]
