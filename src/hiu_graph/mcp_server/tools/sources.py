"""Direct source lookup for retrieval-only MCP queries."""

from maf_graphrag.mcp_server.tools._data_cache import get_graph_data
from maf_graphrag.mcp_server.tools.source_resolver import resolve_text_unit_ids
from maf_graphrag.mcp_server.tools.types import SourceInfo, SourceResult, ToolError, handle_tool_errors, validate_limit


@handle_tool_errors("Source lookup")
async def get_sources_tool(source_ids: list[str], limit: int = 20) -> SourceResult | ToolError:
    """Resolve text-unit IDs to source/document evidence without an LLM."""
    if not source_ids:
        return ToolError(error="source_ids must not be empty.")
    if err := validate_limit(limit):
        return err

    sources, missing_ids = resolve_text_unit_ids(source_ids, get_graph_data(), limit)
    typed_sources = [SourceInfo(**source) for source in sources]
    return SourceResult(
        sources=typed_sources,
        missing_ids=missing_ids,
        returned=len(typed_sources),
        query_type="source_lookup",
    )
