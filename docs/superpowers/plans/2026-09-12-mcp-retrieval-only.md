# MCP Retrieval-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Hiu Graph's generative MCP query surface with five retrieval-only tools that use FastEmbed, LanceDB, and Parquet data and continue to work after indexing when llama.cpp is unavailable.

**Architecture:** Keep GraphRAG indexing and `src/maf_graphrag/core/search.py` unchanged for non-MCP generative flows. Add an MCP retrieval layer containing a cached FastEmbed query encoder and a focused LanceDB adapter, then expose exactly `semantic_search`, `search_entities`, `get_entity`, `get_relationships`, and `get_sources`. Decouple MCP index-data loading from completion-model configuration so querying an existing index never requires `LLAMA_CPP_*` settings.

**Tech Stack:** Python 3.11/3.12, FastMCP 4.x, FastEmbed 0.8.x, LanceDB, pandas/Parquet, GraphRAG 3.0.9, pytest 9.x, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-12-mcp-retrieval-only-design.md`

## Global Constraints

- Gemma/llama.cpp remains required for initial indexing and re-indexing; do not change graph extraction, summarization, or community-report generation.
- MCP query-time code must never construct or call a completion model.
- MCP must advertise exactly five tools: `semantic_search`, `search_entities`, `get_entity`, `get_relationships`, `get_sources`.
- Remove `search_knowledge_graph`, `local_search`, `global_search`, and `list_entities` from the public MCP surface.
- Keep `src/maf_graphrag/core/search.py` generative helpers available for chat and explicit non-MCP flows.
- `FASTEMBED_MODEL_NAME` defaults to `BAAI/bge-small-en-v1.5`.
- `FASTEMBED_CACHE_DIR` defaults to `.cache/fastembed`; Docker uses `/data/fastembed`.
- Query-time and index-time embeddings must use the same environment-backed FastEmbed model name.
- With a valid index, MCP must start and all five tools must work when `LLAMA_CPP_BASE_URL` and `LLAMA_CPP_MODEL` are absent or unreachable.
- Retrieval responses return structured evidence only; no MCP retrieval response contains an LLM-generated `answer` field.
- Use TDD for every production change.

---

## File Structure

**Create**
- `src/maf_graphrag/mcp_server/retrieval/__init__.py` — retrieval-layer exports.
- `src/maf_graphrag/mcp_server/retrieval/query_encoder.py` — cached FastEmbed query encoder.
- `src/maf_graphrag/mcp_server/retrieval/vector_store.py` — raw LanceDB access and score normalization.
- `src/maf_graphrag/mcp_server/tools/retrieval_search.py` — semantic text/entity retrieval tools.
- `src/maf_graphrag/mcp_server/tools/relationships.py` — direct graph-edge traversal.
- `src/maf_graphrag/mcp_server/tools/sources.py` — direct text-unit/source lookup.
- `tests/mcp_server/retrieval/test_query_encoder.py`.
- `tests/mcp_server/retrieval/test_vector_store.py`.
- `tests/mcp_server/tools/test_retrieval_search.py`.
- `tests/mcp_server/tools/test_relationships.py`.
- `tests/mcp_server/tools/test_sources.py`.
- `tests/mcp_server/test_no_llm_runtime.py`.

**Modify**
- `settings.yaml`, `.env.example`, `Dockerfile` — shared FastEmbed configuration.
- `src/maf_graphrag/core/config.py`, `src/maf_graphrag/core/data_loader.py` — explicit output-path validation without completion config.
- `src/maf_graphrag/mcp_server/config.py`, `src/maf_graphrag/mcp_server/tools/_data_cache.py` — MCP-owned output/LanceDB paths.
- `src/maf_graphrag/mcp_server/tools/types.py`, `src/maf_graphrag/mcp_server/tools/__init__.py` — retrieval response contracts and exports.
- `src/maf_graphrag/mcp_server/server.py` — exact five-tool MCP surface.
- `src/maf_graphrag/agents/factories.py` — MCP description reflects evidence retrieval.
- `tests/mcp_server/test_config.py`, `tests/mcp_server/test_server.py`, `tests/mcp_server/tools/test_data_cache.py`, `tests/mcp_server/tools/test_entity_query.py`.
- `README.md`, `src/maf_graphrag/mcp_server/README.md`.
- `pyproject.toml`, `uv.lock` — direct LanceDB dependency.

**Delete**
- `src/maf_graphrag/mcp_server/tools/local_search.py`.
- `src/maf_graphrag/mcp_server/tools/global_search.py`.
- `tests/mcp_server/tools/test_local_search.py`.
- `tests/mcp_server/tools/test_global_search.py`.

---

### Task 1: Decouple MCP Index Loading From Completion Configuration

**Files:**
- Modify: `src/maf_graphrag/core/config.py`
- Modify: `src/maf_graphrag/core/data_loader.py`
- Modify: `src/maf_graphrag/mcp_server/config.py`
- Modify: `src/maf_graphrag/mcp_server/tools/_data_cache.py`
- Modify: `tests/mcp_server/test_config.py`
- Modify: `tests/mcp_server/tools/test_data_cache.py`

**Interfaces:**
- Produces `get_fastembed_model_name() -> str`.
- Produces `get_fastembed_cache_dir() -> str`.
- Changes `validate_output_files(required: list[str] | None = None, output_dir: Path | None = None) -> bool`.
- Changes `get_graph_data(output_dir: Path | None = None) -> GraphData`.
- Produces `MCPConfig.lancedb_dir: Path`.

- [ ] **Step 1: Write failing tests proving an explicit MCP output path does not require llama.cpp configuration**

```python
from pathlib import Path

import pandas as pd


def _write_required_parquet(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"id": "e1", "title": "Alpha", "type": "project"}]).to_parquet(output_dir / "entities.parquet")
    pd.DataFrame([{"source": "Alpha", "target": "Beta"}]).to_parquet(output_dir / "relationships.parquet")
    pd.DataFrame([{"id": "c1"}]).to_parquet(output_dir / "communities.parquet")
    pd.DataFrame([{"id": "r1"}]).to_parquet(output_dir / "community_reports.parquet")
    pd.DataFrame([{"id": "t1", "text": "Alpha text"}]).to_parquet(output_dir / "text_units.parquet")


def test_get_graph_data_uses_explicit_output_without_llama_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    monkeypatch.delenv("LLAMA_CPP_MODEL", raising=False)
    output_dir = tmp_path / "output"
    _write_required_parquet(output_dir)

    from maf_graphrag.mcp_server.tools import _data_cache

    _data_cache._cached_data = None
    data = _data_cache.get_graph_data(output_dir)

    assert data.entities.iloc[0]["title"] == "Alpha"


def test_mcp_config_exposes_lancedb_dir(tmp_path):
    from maf_graphrag.mcp_server.config import MCPConfig

    config = MCPConfig(graphrag_root=tmp_path, output_dir=Path("output"))
    assert config.lancedb_dir == (tmp_path / "output" / "lancedb").resolve()
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
uv run pytest tests/mcp_server/test_config.py tests/mcp_server/tools/test_data_cache.py -q
```

Expected: the new tests fail because validation still falls through `get_output_dir()`/`get_config()` and `lancedb_dir` does not exist.

- [ ] **Step 3: Implement explicit output validation and shared FastEmbed defaults**

In `core/config.py`:

```python
DEFAULT_FASTEMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_FASTEMBED_CACHE_DIR = ".cache/fastembed"


def get_fastembed_model_name() -> str:
    value = os.getenv("FASTEMBED_MODEL_NAME", DEFAULT_FASTEMBED_MODEL_NAME).strip()
    return value or DEFAULT_FASTEMBED_MODEL_NAME


def get_fastembed_cache_dir() -> str:
    value = os.getenv("FASTEMBED_CACHE_DIR", DEFAULT_FASTEMBED_CACHE_DIR).strip()
    return value or DEFAULT_FASTEMBED_CACHE_DIR
```

At the start of `get_config()`:

```python
os.environ.setdefault("FASTEMBED_MODEL_NAME", get_fastembed_model_name())
os.environ.setdefault("FASTEMBED_CACHE_DIR", get_fastembed_cache_dir())
```

Change validation to:

```python
def validate_output_files(required: list[str] | None = None, output_dir: Path | None = None) -> bool:
    required_files = required or [
        "entities.parquet",
        "relationships.parquet",
        "communities.parquet",
        "community_reports.parquet",
        "text_units.parquet",
    ]
    resolved = output_dir if output_dir is not None else get_output_dir()
    missing = [name for name in required_files if not (resolved / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required output files: {', '.join(missing)}\n"
            "Please run indexing first: uv run python -m maf_graphrag.core.index"
        )
    return True
```

In `load_all()`:

```python
if validate:
    validate_output_files(output_dir=output_dir)
```

- [ ] **Step 4: Make MCP resolve its own data paths**

In `_data_cache.py`:

```python
from pathlib import Path

from maf_graphrag.mcp_server.config import MCPConfig


def get_graph_data(output_dir: Path | None = None) -> GraphData:
    global _cached_data
    if _cached_data is None:
        resolved_output_dir = output_dir or MCPConfig.from_env().output_dir
        _cached_data = load_all(output_dir=resolved_output_dir)
    return _cached_data
```

In `MCPConfig`:

```python
@property
def lancedb_dir(self) -> Path:
    return self.output_dir / "lancedb"
```

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
uv run pytest tests/mcp_server/test_config.py tests/mcp_server/tools/test_data_cache.py -q
```

Expected: all focused tests pass with `LLAMA_CPP_BASE_URL` and `LLAMA_CPP_MODEL` absent.

- [ ] **Step 6: Commit**

```bash
git add src/maf_graphrag/core/config.py src/maf_graphrag/core/data_loader.py src/maf_graphrag/mcp_server/config.py src/maf_graphrag/mcp_server/tools/_data_cache.py tests/mcp_server/test_config.py tests/mcp_server/tools/test_data_cache.py
git commit -m "refactor: decouple MCP data loading from llama config"
```

---

### Task 2: Share FastEmbed Configuration and Add the Query Encoder

**Files:**
- Modify: `settings.yaml`
- Modify: `.env.example`
- Modify: `Dockerfile`
- Create: `src/maf_graphrag/mcp_server/retrieval/__init__.py`
- Create: `src/maf_graphrag/mcp_server/retrieval/query_encoder.py`
- Create: `tests/mcp_server/retrieval/test_query_encoder.py`

**Interfaces:**
- Consumes `get_fastembed_model_name()` and `get_fastembed_cache_dir()`.
- Produces `FastEmbedQueryEncoder.encode(text: str) -> list[float]`.
- Produces cached `get_query_encoder() -> FastEmbedQueryEncoder`.

- [ ] **Step 1: Write failing query-encoder tests**

```python
from unittest.mock import MagicMock, patch


def test_query_encoder_uses_shared_fastembed_settings(monkeypatch):
    monkeypatch.setenv("FASTEMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("FASTEMBED_CACHE_DIR", "/tmp/fastembed-test")
    fake_vector = MagicMock()
    fake_vector.tolist.return_value = [0.1, 0.2, 0.3]
    fake_model = MagicMock()
    fake_model.embed.return_value = [fake_vector]

    with patch("maf_graphrag.mcp_server.retrieval.query_encoder.TextEmbedding", return_value=fake_model) as ctor:
        from maf_graphrag.mcp_server.retrieval.query_encoder import FastEmbedQueryEncoder

        vector = FastEmbedQueryEncoder().encode("hello")

    ctor.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5", cache_dir="/tmp/fastembed-test")
    fake_model.embed.assert_called_once_with(["hello"])
    assert vector == [0.1, 0.2, 0.3]


def test_get_query_encoder_is_cached():
    from maf_graphrag.mcp_server.retrieval import query_encoder

    fake_model = MagicMock()
    query_encoder.get_query_encoder.cache_clear()
    with patch("maf_graphrag.mcp_server.retrieval.query_encoder.TextEmbedding", return_value=fake_model) as ctor:
        first = query_encoder.get_query_encoder()
        second = query_encoder.get_query_encoder()

    assert first is second
    ctor.assert_called_once()
```

- [ ] **Step 2: Run tests and verify RED**

```bash
uv run pytest tests/mcp_server/retrieval/test_query_encoder.py -q
```

Expected: import failure because the retrieval package does not exist.

- [ ] **Step 3: Implement the cached encoder**

```python
from functools import lru_cache

from fastembed import TextEmbedding

from maf_graphrag.core.config import get_fastembed_cache_dir, get_fastembed_model_name


class FastEmbedQueryEncoder:
    def __init__(self) -> None:
        self._model = TextEmbedding(
            model_name=get_fastembed_model_name(),
            cache_dir=get_fastembed_cache_dir(),
        )

    def encode(self, text: str) -> list[float]:
        vector = next(iter(self._model.embed([text])))
        return vector.tolist()


@lru_cache(maxsize=1)
def get_query_encoder() -> FastEmbedQueryEncoder:
    return FastEmbedQueryEncoder()
```

Export both symbols from `retrieval/__init__.py`.

- [ ] **Step 4: Parameterize index-time FastEmbed settings with the same variables**

In `settings.yaml`:

```yaml
embedding_models:
  default_embedding_model:
    type: fastembed
    model_provider: fastembed
    model: ${FASTEMBED_MODEL_NAME}
    cache_dir: ${FASTEMBED_CACHE_DIR}
    batch_size: 64
```

In `.env.example`:

```dotenv
FASTEMBED_MODEL_NAME=BAAI/bge-small-en-v1.5
FASTEMBED_CACHE_DIR=.cache/fastembed
```

In Docker `ENV`:

```dockerfile
FASTEMBED_MODEL_NAME=BAAI/bge-small-en-v1.5 \
FASTEMBED_CACHE_DIR=/data/fastembed
```

- [ ] **Step 5: Run tests and smoke-check the default**

```bash
uv run pytest tests/mcp_server/retrieval/test_query_encoder.py tests/core/test_config.py -q
uv run python -c "from maf_graphrag.core.config import get_fastembed_model_name; assert get_fastembed_model_name() == 'BAAI/bge-small-en-v1.5'"
```

Expected: all tests pass and the smoke command exits 0.

- [ ] **Step 6: Commit**

```bash
git add settings.yaml .env.example Dockerfile src/maf_graphrag/mcp_server/retrieval tests/mcp_server/retrieval/test_query_encoder.py
git commit -m "feat: share FastEmbed config with MCP retrieval"
```

---

### Task 3: Add the LanceDB Retrieval Adapter

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/maf_graphrag/mcp_server/retrieval/vector_store.py`
- Create: `tests/mcp_server/retrieval/test_vector_store.py`

**Interfaces:**
- Produces `VectorMatch(id: str, score: float)`.
- Produces `LanceDbVectorStore.search(table_name: str, vector: list[float], limit: int) -> list[VectorMatch]`.
- Normalizes `_distance` as `1.0 / (1.0 + max(distance, 0.0))` so larger means more relevant.

- [ ] **Step 1: Add LanceDB as a direct dependency**

```bash
uv add lancedb
```

Expected: `pyproject.toml` and `uv.lock` record LanceDB directly because MCP imports it directly.

- [ ] **Step 2: Write failing adapter tests using a real temporary LanceDB**

```python
import lancedb
import pyarrow as pa
import pytest


def _create_db(path):
    db = lancedb.connect(str(path))
    db.create_table(
        "text_unit_text",
        data=pa.table({
            "id": ["t1", "t2"],
            "vector": [[1.0, 0.0], [0.0, 1.0]],
            "text": ["alpha", "beta"],
        }),
    )
    return db


def test_search_returns_ranked_ids_with_normalized_scores(tmp_path):
    _create_db(tmp_path / "lancedb")
    from maf_graphrag.mcp_server.retrieval.vector_store import LanceDbVectorStore

    results = LanceDbVectorStore(tmp_path / "lancedb").search("text_unit_text", [1.0, 0.0], 2)

    assert [match.id for match in results] == ["t1", "t2"]
    assert results[0].score > results[1].score


def test_search_rejects_missing_database(tmp_path):
    from maf_graphrag.mcp_server.retrieval.vector_store import LanceDbVectorStore

    with pytest.raises(FileNotFoundError, match="LanceDB database not found"):
        LanceDbVectorStore(tmp_path / "missing").search("text_unit_text", [1.0, 0.0], 1)


def test_search_rejects_missing_table(tmp_path):
    _create_db(tmp_path / "lancedb")
    from maf_graphrag.mcp_server.retrieval.vector_store import LanceDbVectorStore

    with pytest.raises(LookupError, match="LanceDB table not found: entity_description"):
        LanceDbVectorStore(tmp_path / "lancedb").search("entity_description", [1.0, 0.0], 1)
```

- [ ] **Step 3: Run tests and verify RED**

```bash
uv run pytest tests/mcp_server/retrieval/test_vector_store.py -q
```

Expected: import failure because `vector_store.py` does not exist.

- [ ] **Step 4: Implement the adapter**

```python
from dataclasses import dataclass
from pathlib import Path

import lancedb


@dataclass(frozen=True)
class VectorMatch:
    id: str
    score: float


def _score_from_distance(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


class LanceDbVectorStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def search(self, table_name: str, vector: list[float], limit: int) -> list[VectorMatch]:
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
```

- [ ] **Step 5: Run tests and verify GREEN**

```bash
uv run pytest tests/mcp_server/retrieval/test_vector_store.py -q
```

Expected: all adapter tests pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/maf_graphrag/mcp_server/retrieval/vector_store.py tests/mcp_server/retrieval/test_vector_store.py
git commit -m "feat: add LanceDB MCP retrieval adapter"
```

---

### Task 4: Add Semantic Text and Entity Retrieval

**Files:**
- Modify: `src/maf_graphrag/mcp_server/tools/types.py`
- Create: `src/maf_graphrag/mcp_server/tools/retrieval_search.py`
- Modify: `src/maf_graphrag/mcp_server/tools/__init__.py`
- Create: `tests/mcp_server/tools/test_retrieval_search.py`
- Modify: `tests/mcp_server/tools/test_types.py`

**Interfaces:**
- Produces `SemanticMatch`, `SemanticSearchResult`, `EntitySearchMatch`, `EntitySearchResult`.
- Produces `semantic_search_tool(query: str, limit: int = 10) -> SemanticSearchResult | ToolError`.
- Produces `search_entities_tool(query: str, limit: int = 10) -> EntitySearchResult | ToolError`.

- [ ] **Step 1: Write failing retrieval tests with complete fixtures**

```python
import pandas as pd

from maf_graphrag.core.data_loader import GraphData
from maf_graphrag.mcp_server.retrieval.vector_store import VectorMatch


class FakeEncoder:
    def encode(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeStore:
    def search(self, table_name: str, vector: list[float], limit: int) -> list[VectorMatch]:
        if table_name == "text_unit_text":
            return [VectorMatch("t1", 0.90)]
        if table_name == "entity_description":
            return [VectorMatch("e1", 0.88)]
        return []


def _data() -> GraphData:
    return GraphData(
        entities=pd.DataFrame([
            {
                "id": "e1",
                "title": "Project Alpha",
                "type": "project",
                "description": "Main project",
                "community_ids": [1],
            }
        ]),
        relationships=pd.DataFrame(),
        communities=pd.DataFrame(),
        community_reports=pd.DataFrame(),
        text_units=pd.DataFrame([
            {"id": "t1", "text": "Project Alpha uses PostgreSQL", "document_id": "d1"}
        ]),
    )


async def test_semantic_search_returns_hydrated_text_unit(monkeypatch):
    monkeypatch.setattr("maf_graphrag.mcp_server.tools.retrieval_search.get_graph_data", _data)
    monkeypatch.setattr("maf_graphrag.mcp_server.tools.retrieval_search.get_query_encoder", lambda: FakeEncoder())
    monkeypatch.setattr("maf_graphrag.mcp_server.tools.retrieval_search._get_vector_store", lambda: FakeStore())

    from maf_graphrag.mcp_server.tools.retrieval_search import semantic_search_tool

    result = await semantic_search_tool("database", limit=5)

    assert result == {
        "matches": [{
            "text_unit_id": "t1",
            "text": "Project Alpha uses PostgreSQL",
            "score": 0.90,
            "document_ids": ["d1"],
        }],
        "returned": 1,
        "query_type": "semantic_text",
    }


async def test_search_entities_returns_hydrated_entity(monkeypatch):
    monkeypatch.setattr("maf_graphrag.mcp_server.tools.retrieval_search.get_graph_data", _data)
    monkeypatch.setattr("maf_graphrag.mcp_server.tools.retrieval_search.get_query_encoder", lambda: FakeEncoder())
    monkeypatch.setattr("maf_graphrag.mcp_server.tools.retrieval_search._get_vector_store", lambda: FakeStore())

    from maf_graphrag.mcp_server.tools.retrieval_search import search_entities_tool

    result = await search_entities_tool("project", limit=5)

    assert result["matches"] == [{
        "entity_id": "e1",
        "name": "Project Alpha",
        "type": "project",
        "description": "Main project",
        "community_ids": [1],
        "score": 0.88,
    }]


async def test_semantic_search_validates_query_and_limit():
    from maf_graphrag.mcp_server.tools.retrieval_search import semantic_search_tool

    assert "error" in await semantic_search_tool("", limit=5)
    assert "error" in await semantic_search_tool("query", limit=0)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
uv run pytest tests/mcp_server/tools/test_retrieval_search.py tests/mcp_server/tools/test_types.py -q
```

Expected: missing retrieval types and tool module.

- [ ] **Step 3: Add retrieval response types**

```python
class SemanticMatch(TypedDict):
    text_unit_id: str
    text: str
    score: float
    document_ids: NotRequired[list[str]]


class SemanticSearchResult(TypedDict):
    matches: list[SemanticMatch]
    returned: int
    query_type: str


class EntitySearchMatch(TypedDict):
    entity_id: str
    name: str
    type: str
    description: str
    community_ids: list[Any]
    score: float


class EntitySearchResult(TypedDict):
    matches: list[EntitySearchMatch]
    returned: int
    query_type: str
```

Keep `SearchResult` temporarily until Task 7 removes the old MCP wrappers.

- [ ] **Step 4: Implement the two retrieval tools**

In `retrieval_search.py`:

```python
TEXT_UNIT_TABLE = "text_unit_text"
ENTITY_TABLE = "entity_description"


def _get_vector_store() -> LanceDbVectorStore:
    return LanceDbVectorStore(MCPConfig.from_env().lancedb_dir)
```

`semantic_search_tool` must validate query/limit, encode the query, search `TEXT_UNIT_TABLE`, hydrate exact `text_units["id"]` matches, normalize `document_id` into a list of strings, skip stale vector IDs, and return `query_type="semantic_text"`.

`search_entities_tool` must perform the same sequence against `ENTITY_TABLE`, hydrate exact `entities["id"]` matches, skip stale IDs, and return `query_type="semantic_entity"`.

Decorate both with `@handle_tool_errors(...)`.

- [ ] **Step 5: Run tests and verify GREEN**

```bash
uv run pytest tests/mcp_server/tools/test_retrieval_search.py tests/mcp_server/tools/test_types.py -q
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/maf_graphrag/mcp_server/tools/types.py src/maf_graphrag/mcp_server/tools/retrieval_search.py src/maf_graphrag/mcp_server/tools/__init__.py tests/mcp_server/tools/test_retrieval_search.py tests/mcp_server/tools/test_types.py
git commit -m "feat: add semantic MCP retrieval tools"
```

---

### Task 5: Add Relationship and Source Retrieval

**Files:**
- Create: `src/maf_graphrag/mcp_server/tools/relationships.py`
- Create: `src/maf_graphrag/mcp_server/tools/sources.py`
- Modify: `src/maf_graphrag/mcp_server/tools/source_resolver.py`
- Modify: `src/maf_graphrag/mcp_server/tools/types.py`
- Modify: `src/maf_graphrag/mcp_server/tools/__init__.py`
- Create: `tests/mcp_server/tools/test_relationships.py`
- Create: `tests/mcp_server/tools/test_sources.py`
- Modify: `tests/mcp_server/tools/test_source_resolver.py`

**Interfaces:**
- Produces `get_relationships_tool(entity_name: str, limit: int = 20) -> RelationshipResult | ToolError`.
- Produces `get_sources_tool(source_ids: list[str], limit: int = 20) -> SourceResult | ToolError`.
- Produces `resolve_text_unit_ids(source_ids: list[str], data: GraphData, limit: int) -> tuple[list[SourceInfo], list[str]]`.

- [ ] **Step 1: Write failing relationship tests**

```python
import pandas as pd

from maf_graphrag.core.data_loader import GraphData


def _relationship_data() -> GraphData:
    return GraphData(
        entities=pd.DataFrame([
            {"id": "e1", "title": "Project Alpha"},
            {"id": "e2", "title": "Sarah Chen"},
            {"id": "e3", "title": "PostgreSQL"},
        ]),
        relationships=pd.DataFrame([
            {"source": "Project Alpha", "target": "Sarah Chen", "description": "led by", "weight": 2.0, "rank": 1},
            {"source": "PostgreSQL", "target": "Project Alpha", "description": "used by", "weight": 1.0, "rank": 2},
        ]),
        communities=pd.DataFrame(),
        community_reports=pd.DataFrame(),
        text_units=pd.DataFrame(),
    )


async def test_relationships_return_both_directions(monkeypatch):
    monkeypatch.setattr("maf_graphrag.mcp_server.tools.relationships.get_graph_data", _relationship_data)
    from maf_graphrag.mcp_server.tools.relationships import get_relationships_tool

    result = await get_relationships_tool("project alpha", limit=20)

    assert [edge["counterpart"] for edge in result["relationships"]] == ["Sarah Chen", "PostgreSQL"]
    assert [edge["direction"] for edge in result["relationships"]] == ["outgoing", "incoming"]


async def test_relationships_return_error_for_unknown_entity(monkeypatch):
    monkeypatch.setattr("maf_graphrag.mcp_server.tools.relationships.get_graph_data", _relationship_data)
    from maf_graphrag.mcp_server.tools.relationships import get_relationships_tool

    result = await get_relationships_tool("Missing", limit=20)
    assert result["error"] == "Entity not found: Missing"
```

- [ ] **Step 2: Write failing source-resolution tests**

```python
import pandas as pd

from maf_graphrag.core.data_loader import GraphData


def _source_data() -> GraphData:
    return GraphData(
        entities=pd.DataFrame(),
        relationships=pd.DataFrame(),
        communities=pd.DataFrame(),
        community_reports=pd.DataFrame(),
        text_units=pd.DataFrame([
            {"id": "tu-a", "document_id": "doc-a", "text": "Alpha source text"},
            {"id": "tu-b", "document_id": "doc-b", "text": "Beta source text"},
        ]),
        documents=pd.DataFrame([
            {"id": "doc-a", "title": "alpha.md", "text": "Full alpha document"},
            {"id": "doc-b", "title": "beta.md", "text": "Full beta document"},
        ]),
    )


async def test_sources_preserve_order_deduplicate_and_report_missing(monkeypatch):
    monkeypatch.setattr("maf_graphrag.mcp_server.tools.sources.get_graph_data", _source_data)
    from maf_graphrag.mcp_server.tools.sources import get_sources_tool

    result = await get_sources_tool(["tu-b", "tu-a", "tu-b", "missing"], limit=3)

    assert [item["text_unit_id"] for item in result["sources"]] == ["tu-b", "tu-a"]
    assert result["missing_ids"] == ["missing"]
    assert result["sources"][0]["document_title"] == "beta.md"
    assert result["sources"][0]["text"] == "Beta source text"
```

- [ ] **Step 3: Run tests and verify RED**

```bash
uv run pytest tests/mcp_server/tools/test_relationships.py tests/mcp_server/tools/test_sources.py tests/mcp_server/tools/test_source_resolver.py -q
```

Expected: new modules/types are missing.

- [ ] **Step 4: Add response types**

```python
class RelationshipInfo(TypedDict):
    source: str
    target: str
    counterpart: str
    direction: str
    description: NotRequired[str]
    weight: NotRequired[float]
    rank: NotRequired[float]


class RelationshipResult(TypedDict):
    entity: str
    relationships: list[RelationshipInfo]
    returned: int
    query_type: str


class SourceInfo(TypedDict):
    text_unit_id: str
    text: str
    document_id: NotRequired[str]
    document_title: NotRequired[str]
    text_preview: NotRequired[str]


class SourceResult(TypedDict):
    sources: list[SourceInfo]
    missing_ids: list[str]
    returned: int
    query_type: str
```

- [ ] **Step 5: Implement direct relationship traversal**

`get_relationships_tool` must validate entity name and limit, resolve canonical entity title by case-insensitive equality first and case-insensitive contains second, filter rows where the title is source or target, preserve DataFrame order, include `counterpart` and `direction`, include non-null description/weight/rank, and return `ToolError(error=f"Entity not found: {entity_name}")` when resolution fails.

- [ ] **Step 6: Implement concrete text-unit source resolution**

In `source_resolver.py` add:

```python
def resolve_text_unit_ids(source_ids: list[str], data: GraphData, limit: int) -> tuple[list[SourceInfo], list[str]]:
    unique_ids = list(dict.fromkeys(str(source_id) for source_id in source_ids))
    text_rows = {
        str(row["id"]): row
        for _, row in data.text_units.iterrows()
        if row.get("id") is not None
    }
    doc_titles = {}
    if data.documents is not None:
        doc_titles = {
            str(row["id"]): str(row.get("title", ""))
            for _, row in data.documents.iterrows()
            if row.get("id") is not None
        }

    resolved: list[SourceInfo] = []
    missing: list[str] = []
    for source_id in unique_ids:
        row = text_rows.get(source_id)
        if row is None:
            missing.append(source_id)
            continue
        if len(resolved) >= limit:
            break
        text = str(row.get("text", ""))
        item: SourceInfo = {
            "text_unit_id": source_id,
            "text": text,
            "text_preview": _make_text_preview(text),
        }
        document_id = row.get("document_id")
        if document_id is not None:
            document_key = str(document_id)
            item["document_id"] = document_key
            title = doc_titles.get(document_key)
            if title:
                item["document_title"] = title
        resolved.append(item)
    return resolved, missing
```

`get_sources_tool` validates `limit`, returns `ToolError(error="source_ids must not be empty.")` for an empty list, calls this helper, and returns `query_type="source_lookup"`.

- [ ] **Step 7: Run tests and verify GREEN**

```bash
uv run pytest tests/mcp_server/tools/test_relationships.py tests/mcp_server/tools/test_sources.py tests/mcp_server/tools/test_source_resolver.py -q
```

Expected: all focused tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/maf_graphrag/mcp_server/tools/relationships.py src/maf_graphrag/mcp_server/tools/sources.py src/maf_graphrag/mcp_server/tools/source_resolver.py src/maf_graphrag/mcp_server/tools/types.py src/maf_graphrag/mcp_server/tools/__init__.py tests/mcp_server/tools/test_relationships.py tests/mcp_server/tools/test_sources.py tests/mcp_server/tools/test_source_resolver.py
git commit -m "feat: add graph and source MCP retrieval"
```

---

### Task 6: Replace the Public MCP Surface

**Files:**
- Modify: `src/maf_graphrag/mcp_server/server.py`
- Modify: `src/maf_graphrag/mcp_server/tools/__init__.py`
- Modify: `src/maf_graphrag/mcp_server/tools/types.py`
- Modify: `src/maf_graphrag/agents/factories.py`
- Modify: `tests/mcp_server/test_server.py`
- Modify: `tests/mcp_server/tools/test_entity_query.py`
- Delete: `src/maf_graphrag/mcp_server/tools/local_search.py`
- Delete: `src/maf_graphrag/mcp_server/tools/global_search.py`
- Delete: `tests/mcp_server/tools/test_local_search.py`
- Delete: `tests/mcp_server/tools/test_global_search.py`

**Interfaces:**
- Public MCP functions become exactly `semantic_search`, `search_entities`, `get_entity`, `get_relationships`, `get_sources`.

- [ ] **Step 1: Replace server tests with failing retrieval-only dispatch/discovery tests**

```python
from unittest.mock import AsyncMock, patch

from fastmcp import Client


async def test_semantic_search_forwards_to_retrieval_tool():
    from maf_graphrag.mcp_server.server import semantic_search

    expected = {"matches": [], "returned": 0, "query_type": "semantic_text"}
    with patch("maf_graphrag.mcp_server.server.semantic_search_tool", AsyncMock(return_value=expected)) as tool:
        result = await semantic_search("database", limit=7)

    tool.assert_awaited_once_with("database", 7)
    assert result is expected


async def test_advertised_tools_are_retrieval_only():
    from maf_graphrag.mcp_server.server import mcp

    async with Client(mcp) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools} == {
        "semantic_search",
        "search_entities",
        "get_entity",
        "get_relationships",
        "get_sources",
    }
```

Add direct dispatch assertions with these exact expectations:

```python
search_entities_tool.assert_awaited_once_with("project", 4)
entity_query_tool.assert_awaited_once_with(entity_name="Project Alpha", limit=1)
get_relationships_tool.assert_awaited_once_with("Project Alpha", 5)
get_sources_tool.assert_awaited_once_with(["tu-a"], 6)
```

- [ ] **Step 2: Run server tests and verify RED**

```bash
uv run pytest tests/mcp_server/test_server.py -q
```

Expected: current server still advertises generative tools and lacks the new public functions.

- [ ] **Step 3: Rewire `server.py` to exactly five retrieval tools**

Use only these tool imports:

```python
from maf_graphrag.mcp_server.tools import (
    entity_query_tool,
    get_relationships_tool,
    get_sources_tool,
    search_entities_tool,
    semantic_search_tool,
)
```

Register functions with these signatures:

```python
@mcp.tool()
async def semantic_search(query: str, limit: int = 10):
    return await semantic_search_tool(query, limit)


@mcp.tool()
async def search_entities(query: str, limit: int = 10):
    return await search_entities_tool(query, limit)


@mcp.tool()
async def get_entity(entity_name: str):
    return await entity_query_tool(entity_name=entity_name, limit=1)


@mcp.tool()
async def get_relationships(entity_name: str, limit: int = 20):
    return await get_relationships_tool(entity_name, limit)


@mcp.tool()
async def get_sources(source_ids: list[str], limit: int = 20):
    return await get_sources_tool(source_ids, limit)
```

Do not register `list_entities`, `search_knowledge_graph`, `local_search`, or `global_search`.

- [ ] **Step 4: Remove obsolete MCP generative wrappers and answer-oriented MCP types**

Delete the two wrapper modules and tests listed above. Remove their exports from `tools/__init__.py`. Remove `SearchContext` and `SearchResult` from `tools/types.py` once no MCP imports need them. Do not modify `src/maf_graphrag/core/search.py`.

- [ ] **Step 5: Update the chat agent's MCP description**

In `agents/factories.py` use:

```python
description="Retrieve structured evidence, entities, relationships, and sources from the GraphRAG knowledge graph"
```

- [ ] **Step 6: Run MCP tests and verify GREEN**

```bash
uv run pytest tests/mcp_server/test_server.py tests/mcp_server/tools -q
```

Expected: all MCP tests pass and discovery returns exactly five retrieval tools.

- [ ] **Step 7: Commit**

```bash
git add -A src/maf_graphrag/mcp_server src/maf_graphrag/agents/factories.py tests/mcp_server
git commit -m "feat: replace MCP search with retrieval-only tools"
```

---

### Task 7: Add the Hard No-LLM Runtime Regression Test

**Files:**
- Create: `tests/mcp_server/test_no_llm_runtime.py`

**Interfaces:**
- Verifies the approved runtime contract; no production API.

- [ ] **Step 1: Write a complete no-LLM regression test**

```python
import pandas as pd

from maf_graphrag.core.data_loader import GraphData
from maf_graphrag.mcp_server.retrieval.vector_store import VectorMatch


class FakeEncoder:
    def encode(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeVectorStore:
    def search(self, table_name: str, vector: list[float], limit: int) -> list[VectorMatch]:
        if table_name == "text_unit_text":
            return [VectorMatch("t1", 0.9)]
        if table_name == "entity_description":
            return [VectorMatch("e1", 0.8)]
        return []


def _graph_data() -> GraphData:
    return GraphData(
        entities=pd.DataFrame([{
            "id": "e1",
            "title": "Project Alpha",
            "type": "project",
            "description": "Main project",
            "community_ids": [1],
        }]),
        relationships=pd.DataFrame([{
            "source": "Project Alpha",
            "target": "Sarah Chen",
            "description": "led by",
        }]),
        communities=pd.DataFrame(),
        community_reports=pd.DataFrame(),
        text_units=pd.DataFrame([{
            "id": "t1",
            "text": "Project Alpha is led by Sarah Chen",
            "document_id": "d1",
        }]),
        documents=pd.DataFrame([{
            "id": "d1",
            "title": "project_alpha.md",
            "text": "Project Alpha is led by Sarah Chen",
        }]),
    )


async def test_all_retrieval_tools_work_without_completion_model(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    monkeypatch.delenv("LLAMA_CPP_MODEL", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("MCP retrieval attempted a generative GraphRAG call")

    monkeypatch.setattr("graphrag.api.local_search", forbidden)
    monkeypatch.setattr("graphrag.api.global_search", forbidden)
    monkeypatch.setattr("graphrag.api.basic_search", forbidden)
    monkeypatch.setattr("graphrag.api.drift_search", forbidden)

    from maf_graphrag.mcp_server.tools import _data_cache
    _data_cache._cached_data = _graph_data()

    monkeypatch.setattr(
        "maf_graphrag.mcp_server.tools.retrieval_search.get_query_encoder",
        lambda: FakeEncoder(),
    )
    monkeypatch.setattr(
        "maf_graphrag.mcp_server.tools.retrieval_search._get_vector_store",
        lambda: FakeVectorStore(),
    )

    from maf_graphrag.mcp_server.tools.entity_query import entity_query_tool
    from maf_graphrag.mcp_server.tools.relationships import get_relationships_tool
    from maf_graphrag.mcp_server.tools.retrieval_search import search_entities_tool, semantic_search_tool
    from maf_graphrag.mcp_server.tools.sources import get_sources_tool

    assert "error" not in await semantic_search_tool("Project Alpha", 3)
    assert "error" not in await search_entities_tool("Project Alpha", 3)
    assert "error" not in await entity_query_tool(entity_name="Project Alpha", limit=1)
    assert "error" not in await get_relationships_tool("Project Alpha", 5)
    assert "error" not in await get_sources_tool(["t1"], 5)
```

- [ ] **Step 2: Run the regression test**

```bash
uv run pytest tests/mcp_server/test_no_llm_runtime.py -q
```

Expected: PASS only when retrieval no longer reaches GraphRAG's generative APIs or completion configuration.

- [ ] **Step 3: Run a static import guard**

```bash
python - <<'PY'
from pathlib import Path

paths = [
    Path("src/maf_graphrag/mcp_server/server.py"),
    *Path("src/maf_graphrag/mcp_server/tools").glob("*.py"),
    *Path("src/maf_graphrag/mcp_server/retrieval").glob("*.py"),
]
for path in paths:
    text = path.read_text(encoding="utf-8")
    assert "graphrag.api.local_search" not in text, path
    assert "graphrag.api.global_search" not in text, path
    assert "litellm" not in text.lower(), path
print("MCP import guard passed")
PY
```

Expected: `MCP import guard passed`.

- [ ] **Step 4: Commit**

```bash
git add tests/mcp_server/test_no_llm_runtime.py
git commit -m "test: prove MCP retrieval needs no completion LLM"
```

---

### Task 8: Documentation, Full Verification, and Container Acceptance

**Files:**
- Modify: `README.md`
- Modify: `src/maf_graphrag/mcp_server/README.md`
- No other production changes unless verification reveals a defect.

**Interfaces:**
- Documents and verifies the exact five-tool runtime contract.

- [ ] **Step 1: Update lifecycle and MCP migration documentation**

Document this exact split:

```text
Indexing:
  Documents -> Gemma/llama.cpp (graph extraction/summaries)
            -> FastEmbed (vectors)
            -> Parquet + LanceDB

MCP runtime after indexing:
  MCP client -> FastEmbed query embedding -> LanceDB/Parquet -> structured evidence
  No completion LLM or llama.cpp call is required.
```

Document exactly these tools:

```text
semantic_search(query, limit=10)
search_entities(query, limit=10)
get_entity(entity_name)
get_relationships(entity_name, limit=20)
get_sources(source_ids, limit=20)
```

State that `search_knowledge_graph`, `local_search`, `global_search`, and `list_entities` are removed from MCP and that callers must migrate to retrieval primitives.

- [ ] **Step 2: Run stale-reference checks**

```bash
rg -n "search_knowledge_graph|local_search\(|global_search\(|list_entities\(" README.md src/maf_graphrag/mcp_server src/maf_graphrag/agents tests/mcp_server
```

Expected: no stale public-MCP references. Any hit in `src/maf_graphrag/core/search.py` is outside this command's paths and intentionally remains.

- [ ] **Step 3: Run formatting, linting, typing, and the full suite**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Expected: every command exits 0 and pytest reports zero failures.

- [ ] **Step 4: Build the Docker image**

```bash
docker build -t hiu-graph:mcp-retrieval-only .
```

Expected: build exits 0.

- [ ] **Step 5: Prove an existing-index container starts with llama.cpp unreachable**

```bash
docker run --rm \
  -p 8011:8011 \
  -e LLAMA_CPP_BASE_URL=http://127.0.0.1:9 \
  -e LLAMA_CPP_MODEL= \
  -v "$PWD/output:/app/output:ro" \
  hiu-graph:mcp-retrieval-only
```

Expected: entrypoint logs `GraphRAG index is ready; skipping indexing.` and MCP starts without a llama.cpp health check.

- [ ] **Step 6: Invoke the five tools and inspect logs**

With MCP Inspector or the repository FastMCP client, call `semantic_search("Project Alpha", 3)`, copy the first returned `text_unit_id` into a local variable named `source_id`, then call `search_entities("Project Alpha", 3)`, `get_entity("Project Alpha")`, `get_relationships("Project Alpha", 5)`, and `get_sources([source_id], 5)`.

Expected: each call returns structured evidence and container logs contain no `LiteLLM completion()` lines.

- [ ] **Step 7: Verify first-run auto-indexing was not regressed**

```bash
uv run pytest tests/test_docker_entrypoint.py -q
```

Expected: all entrypoint tests pass, preserving `index missing -> wait for llama.cpp -> index -> verify -> start MCP`.

- [ ] **Step 8: Review the branch diff against the approved spec**

```bash
git diff master...HEAD --stat
git diff master...HEAD -- src/maf_graphrag/mcp_server settings.yaml Dockerfile README.md
```

Confirm all eight spec success criteria are represented in code/tests and `src/maf_graphrag/core/search.py` still exists unchanged.

- [ ] **Step 9: Commit documentation or verification fixes**

```bash
git add README.md src/maf_graphrag/mcp_server/README.md
git commit -m "docs: document retrieval-only MCP runtime"
```

If verification required a production fix, rerun Steps 2-7 before committing that fix with its own descriptive commit message.
