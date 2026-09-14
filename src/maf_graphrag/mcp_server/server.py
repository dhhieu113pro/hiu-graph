"""
GraphRAG MCP Server

FastMCP server that exposes retrieval-only GraphRAG tools via Streamable HTTP.
"""

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from maf_graphrag.mcp_server.config import MCPConfig
from maf_graphrag.mcp_server.tools import (
    entity_query_tool,
    get_relationships_tool,
    get_sources_tool,
    search_entities_tool,
    semantic_search_tool,
)
from maf_graphrag.mcp_server.tools.types import (
    EntityQueryResult,
    EntitySearchResult,
    RelationshipResult,
    SemanticSearchResult,
    SourceResult,
    ToolError,
)

config = MCPConfig.from_env()
mcp = FastMCP(name=config.server_name, version=config.server_version)


@mcp.tool()
async def semantic_search(query: str, limit: int = 10) -> SemanticSearchResult | ToolError:
    """Retrieve semantically relevant source text from the indexed knowledge graph."""
    return await semantic_search_tool(query, limit)


@mcp.tool()
async def search_entities(query: str, limit: int = 10) -> EntitySearchResult | ToolError:
    """Retrieve entities whose indexed descriptions are semantically relevant to a query."""
    return await search_entities_tool(query, limit)


@mcp.tool()
async def get_entity(entity_name: str) -> EntityQueryResult | ToolError:
    """Get indexed metadata for one entity by name."""
    return await entity_query_tool(entity_name=entity_name, limit=1)


@mcp.tool()
async def get_relationships(entity_name: str, limit: int = 20) -> RelationshipResult | ToolError:
    """Return graph relationships connected to an indexed entity."""
    return await get_relationships_tool(entity_name, limit)


@mcp.tool()
async def get_sources(source_ids: list[str], limit: int = 20) -> SourceResult | ToolError:
    """Resolve text-unit IDs to indexed source/document evidence."""
    return await get_sources_tool(source_ids, limit)


def create_mcp_server() -> FastMCP:
    """Create and return configured MCP server instance."""
    return mcp


_cors_middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins or ["http://127.0.0.1:8011"],
        allow_methods=config.cors_methods or ["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=(config.cors_headers or ["Content-Type", "Authorization"])
        + ["mcp-protocol-version", "mcp-session-id"],
        expose_headers=["mcp-session-id"],
    )
]

app = mcp.http_app(middleware=_cors_middleware)


if __name__ == "__main__":
    print("🚀 Starting GraphRAG Retrieval MCP Server")
    print(f"   Server: {config.server_name} v{config.server_version}")
    print(f"   URL: {config.server_url}")
    print(f"   GraphRAG Root: {config.graphrag_root}")
    print("\n📋 Available Tools:")
    print("   - semantic_search(query, limit=10)")
    print("   - search_entities(query, limit=10)")
    print("   - get_entity(entity_name)")
    print("   - get_relationships(entity_name, limit=20)")
    print("   - get_sources(source_ids, limit=20)")
    print("\n✨ Retrieval server ready; no completion LLM is used by MCP tools")

    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
