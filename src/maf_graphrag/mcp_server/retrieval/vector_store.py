"""Focused LanceDB access for MCP retrieval."""

from dataclasses import dataclass
from pathlib import Path

import lancedb


@dataclass(frozen=True)
class VectorMatch:
    """Stable MCP-facing vector search result."""

    id: str
    score: float


def _score_from_distance(distance: float) -> float:
    """Normalize LanceDB distance so larger scores always mean more relevant."""
    return 1.0 / (1.0 + max(distance, 0.0))


class LanceDbVectorStore:
    """Read-only vector search adapter for GraphRAG's LanceDB output."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def search(self, table_name: str, vector: list[float], limit: int) -> list[VectorMatch]:
        """Search one vector table and return normalized ranked matches."""
        if not self._db_path.exists():
            raise FileNotFoundError(f"LanceDB database not found: {self._db_path}")

        db = lancedb.connect(str(self._db_path))
        if table_name not in db.table_names():
            raise LookupError(f"LanceDB table not found: {table_name}")

        rows = db.open_table(table_name).search(vector).limit(limit).to_list()
        return [
            VectorMatch(
                id=str(row["id"]),
                score=_score_from_distance(float(row.get("_distance", 0.0))),
            )
            for row in rows
        ]
