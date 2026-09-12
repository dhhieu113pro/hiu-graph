"""
Lazy singleton cache for GraphRAG data.

Loads the knowledge graph once on first access and reuses it for all
subsequent MCP tool calls, avoiding redundant Parquet reads on every request.
"""

import logging
from pathlib import Path

from maf_graphrag.core.data_loader import GraphData, load_all
from maf_graphrag.mcp_server.config import MCPConfig

logger = logging.getLogger(__name__)

_cached_data: GraphData | None = None


def get_graph_data(output_dir: Path | None = None) -> GraphData:
    """Return cached GraphData, loading from disk on first call.

    Args:
        output_dir: Optional explicit GraphRAG output directory. MCP callers
            use this to avoid loading completion-model configuration.

    Raises:
        FileNotFoundError: If knowledge graph files are missing.
    """
    global _cached_data  # noqa: PLW0603
    if _cached_data is None:
        logger.info("Loading knowledge graph data (first request)…")
        resolved_output_dir = output_dir or MCPConfig.from_env().output_dir
        _cached_data = load_all(output_dir=resolved_output_dir)
        logger.info("Knowledge graph loaded: %s", _cached_data)
    return _cached_data
