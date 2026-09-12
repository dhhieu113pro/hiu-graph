# MCP Retrieval-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Hiu Graph's generative MCP query surface with five retrieval-only tools that use FastEmbed, LanceDB, and Parquet data and work after indexing with llama.cpp completely unavailable.

**Architecture:** Keep GraphRAG indexing and `src/maf_graphrag/core/search.py` unchanged for non-MCP generative flows. Add a small MCP retrieval layer with a cached FastEmbed query encoder and a focused LanceDB adapter, then expose exactly `semantic_search`, `search_entities`, `get_entity`, `get_relationships`, and `get_sources`. Decouple MCP index-data loading from completion-model configuration so existing indexes can be queried without `LLAMA_CPP_*` settings.

**Tech Stack:** Python 3.11/3.12, FastMCP 4.x, FastEmbed 0.8.x, LanceDB, pandas/Parquet, GraphRAG 3.0.9, pytest 9.x, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-12-mcp-retrieval-only-design.md`

## Global Constraints

- Gemma/llama.cpp remains required for initial indexing and re-indexing; do not change graph extraction, summarization, or community-report generation.
- MCP query-time code must never construct or call a completion model.
- MCP must advertise exactly five tools: `semantic_search`, `search_entities`, `get_entity`, `get_relationships`, `get_sources`.
- Remove `search_knowledge_graph`, `local_search`, `global_search`, and `list_entities` from the public MCP surface.
- Keep `src/maf_graphrag/core/search.py` generative helpers available for chat and explicit non-MCP flows.
- `FASTEMBED_MODEL_NAME` defaults to `BAAI/bge-small-en-v1.5`.
- `FASTEMBED_CACHE_DIR` defaults to `.cache/fastembed`; Docker may continue using `/data/fastembed`.
- Query-time embeddings and index-time embeddings must use the same environment-backed FastEmbed model name.
- With a valid index, MCP must start and all five tools must work when `LLAMA_CPP_BASE_URL` and `LLAMA_CPP_MODEL` are absent or point to an unreachable server.
- Retrieval responses return structured evidence only; no response type contains an LLM-generated `answer` field.
- Use TDD: add a failing test, run it, implement the smallest production change, then rerun the focused tests before each commit.

---

## File Structure

**Create:**
- `src/maf_graphrag/mcp_server/retrieval/__init__.py` — retrieval-layer exports.
- `src/maf_graphrag/mcp_server/retrieval/query_encoder.py` — cached FastEmbed query encoder.
- `src/maf_graphrag/mcp_server/retrieval/vector_store.py` — all raw LanceDB access and score normalization.
- `src/maf_graphrag/mcp_server/tools/retrieval_search.py` — `semantic_search_tool` and `search_entities_tool`.
- `src/maf_graphrag/mcp_server/tools/relationships.py` — direct graph-edge traversal.
- `src/maf_graphrag/mcp_server/tools/sources.py` — direct source/text-unit resolution by IDs.
- `tests/mcp_server/retrieval/test_query_encoder.py`.
- `tests/mcp_server/retrieval/test_vector_store.py`.
- `tests/mcp_server/tools/test_retrieval_search.py`.
- `tests/mcp_server/tools/test_relationships.py`.
- `tests/mcp_server/tools/test_sources.py`.
- `tests/mcp_server/test_no_llm_runtime.py`.

**Modify:**
- `settings.yaml` — read FastEmbed model/cache from shared environment variables.
- `.env.example` — document FastEmbed model/cache variables.
- `Dockerfile` — set the default FastEmbed model name alongside the existing Docker cache location.
- `src/maf_graphrag/core/config.py` — provide shared FastEmbed defaults/helpers and allow output validation without loading completion config.
- `src/maf_graphrag/core/data_loader.py` — validate the explicit `output_dir` passed by MCP instead of falling back through `get_config()`.
- `src/maf_graphrag/mcp_server/config.py` — expose the resolved MCP output/LanceDB paths without completion-model settings.
- `src/maf_graphrag/mcp_server/tools/_data_cache.py` — load cached Parquet data from MCP's explicit output directory.
- `src/maf_graphrag/mcp_server/tools/types.py` — retrieval response TypedDicts.
- `src/maf_graphrag/mcp_server/tools/__init__.py` — export only current tool helpers.
- `src/maf_graphrag/mcp_server/server.py` — register exactly the five retrieval-only MCP tools.
- `src/maf_graphrag/agents/factories.py` — update MCP tool description so the chat agent knows MCP returns evidence rather than generated answers.
- `tests/mcp_server/test_config.py` — runtime-path and no-LLM-env coverage.
- `tests/mcp_server/test_server.py` — new tool dispatch and advertised-tool coverage.
- `tests/mcp_server/tools/test_data_cache.py` — explicit-output-dir/no-completion-config coverage.
- `tests/mcp_server/tools/test_entity_query.py` — retain `get_entity` helper coverage; remove assumptions that `list_entities` is a public MCP tool.
- `README.md` and `src/maf_graphrag/mcp_server/README.md` — migration and runtime docs.
- `pyproject.toml` and `uv.lock` — add LanceDB as a direct dependency because MCP code imports it directly.

**Delete:**
- `src/maf_graphrag/mcp_server/tools/local_search.py`.
- `src/maf_graphrag/mcp_server/tools/global_search.py`.
- `tests/mcp_server/tools/test_local_search.py`.
- `tests/mcp_server/tools/test_global_search.py`.

---

### Task 1: Decouple MCP Index Loading From Completion-Model Configuration

**Files:**
- Modify: `src/maf_graphrag/core/config.py`
- Modify: `src/maf_graphrag/core/data_loader.py`
- Modify: `src/maf_graphrag/mcp_server/config.py`
- Modify: `src/maf_graphrag/mcp_server/tools/_data_cache.py`
- Modify: `tests/mcp_server/test_config.py`
- Modify: `tests/mcp_server/tools/test_data_cache.py`

**Interfaces:**
- Produces: `get_fastembed_model_name() -> str`
- Produces: `get_fastembed_cache_dir() -> str`
- Changes: `validate_output_files(required: list[str] | None = None, output_dir: Path | None = None) -> bool`
- Changes: `get_graph_data(output_dir: Path | None = None) -> GraphData`
- Produces: `MCPConfig.lancedb_dir: Path` property returning `output_dir / "lancedb"`

- [ ] **Step 1: Write failing tests for explicit output paths and missing llama.cpp environment variables**

Add tests that remove completion settings and prove MCP data loading can use a supplied output directory:

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
```

Also test:

```python
def test_mcp_config_exposes_lancedb_dir(tmp_path):
    from maf_graphrag.mcp_server.config import MCPConfig

    config = MCPConfig(graphrag_root=tmp_path, output_dir=Path("output"))

    assert config.lancedb_dir == (tmp_path / "output" / "lancedb").resolve()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest tests/mcp_server/test_config.py tests/mcp_server/tools/test_data_cache.py -q
```

Expected: at least the new explicit-output/no-LLM test fails because `load_all(validate=True)` ultimately calls `validate_output_files()` without the supplied path, and `MCPConfig.lancedb_dir` does not exist yet.

- [ ] **Step 3: Add shared FastEmbed defaults and explicit output validation**

In `core/config.py`, add:

```python
DEFAULT_FASTEMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_FASTEMBED_CACHE_DIR = ".cache/fastembed"


def get_fastembed_model_name() -> str:
    return os.getenv("FASTEMBED_MODEL_NAME", DEFAULT_FASTEMBED_MODEL_NAME).strip() or DEFAULT_FASTEMBED_MODEL_NAME


def get_fastembed_cache_dir() -> str:
    return os.getenv("FASTEMBED_CACHE_DIR", DEFAULT_FASTEMBED_CACHE_DIR).strip() or DEFAULT_FASTEMBED_CACHE_DIR
```

At the beginning of `get_config()`, make the defaults available for `settings.yaml` interpolation without requiring callers to set them:

```python
os.environ.setdefault("FASTEMBED_MODEL_NAME", get_fastembed_model_name())
os.environ.setdefault("FASTEMBED_CACHE_DIR", get_fastembed_cache_dir())
```

Change validation to respect an explicit path:

```python
def validate_output_files(required: list[str] | None = None, output_dir: Path | None = None) -> bool:
    if required is None:
        required = [
            "entities.parquet",
            "relationships.parquet",
            "communities.parquet",
            "community_reports.parquet",
            "text_units.parquet",
        ]
    resolved_output_dir = output_dir if output_dir is not None else get_output_dir()
    missing = [name for name in required if not (resolved_output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required output files: {', '.join(missing)}\n"
            "Please run indexing first: uv run python -m maf_graphrag.core.index"
        )
    return True
```

In `data_loader.load_all`, use the already-resolved directory:

```python
if validate:
    validate_output_files(output_dir=output_dir)
```

- [ ] **Step 4: Make MCP data cache resolve its own output path**

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

In `MCPConfig` add:

```python
@property
def lancedb_dir(self) -> Path:
    return self.output_dir / "lancedb"
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/mcp_server/test_config.py tests/mcp_server/tools/test_data_cache.py -q
```

Expected: all tests pass with `LLAMA_CPP_BASE_URL` and `LLAMA_CPP_MODEL` absent in the new regression case.

- [ ] **Step 6: Commit the runtime-decoupling change**

```bash
git add src/maf_graphrag/core/config.py src/maf_graphrag/core/data_loader.py src/maf_graphrag/mcp_server/config.py src/maf_graphrag/mcp_server/tools/_data_cache.py tests/mcp_server/test_config.py tests/mcp_server/tools/test_data_cache.py
git commit -m "refactor: decouple MCP data loading from llama config"
```

---

### Task 2: Share FastEmbed Configuration Between Indexing and MCP

**Files:**
- Modify: `settings.yaml`
- Modify: `.env.example`
- Modify: `Dockerfile`
- Create: `src/maf_graphrag/mcp_server/retrieval/__init__.py`
- Create: `src/maf_graphrag/mcp_server/retrieval/query_encoder.py`
- Create: `tests/mcp_server/retrieval/test_query_encoder.py`

**Interfaces:**
- Consumes: `get_fastembed_model_name() -> str`, `get_fastembed_cache_dir() -> str`
- Produces: `FastEmbedQueryEncoder.encode(text: str) -> list[float]`
- Produces: `get_query_encoder() -> FastEmbedQueryEncoder`, cached once per process

- [ ] **Step 1: Write failing query-encoder tests**

```python
from unittest.mock import MagicMock, patch


def test_query_encoder_uses_shared_fastembed_settings(monkeypatch):
    monkeypatch.setenv("FASTEMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("FASTEMBED_CACHE_DIR", "/tmp/fastembed-test")

    fake_model = MagicMock()
    fake_model.embed.return_value = [MagicMock(tolist=lambda: [0.1, 0.2, 0.3])]

    with patch("maf_graphrag.mcp_server.retrieval.query_encoder.TextEmbedding", return_value=fake_model) as ctor:
        from maf_graphrag.mcp_server.retrieval.query_encoder import FastEmbedQueryEncoder

        vector = FastEmbedQueryEncoder().encode("hello")

    ctor.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5", cache_dir="/tmp/fastembed-test")
    fake_model.embed.assert_called_once_with(["hello"])
    assert vector == [0.1, 0.2, 0.3]
```

Add a singleton test:

```python
def test_get_query_encoder_is_cached():
    from maf_graphrag.mcp_server.retrieval import query_encoder

    query_encoder.get_query_encoder.cache_clear()
    first = query_encoder.get_query_encoder()
    second = query_encoder.get_query_encoder()

    assert first is second
```

- [ ] **Step 2: Run query-encoder tests and verify RED**

```bash
uv run pytest tests/mcp_server/retrieval/test_query_encoder.py -q
```

Expected: import failure because the retrieval/query-encoder module does not exist.

- [ ] **Step 3: Implement the minimal cached query encoder**

Create `query_encoder.py`:

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

Export `FastEmbedQueryEncoder` and `get_query_encoder` from `retrieval/__init__.py`.

- [ ] **Step 4: Parameterize the indexing config with the same environment values**

Change `settings.yaml` embedding section from hard-coded values to:

```yaml
embedding_models:
  default_embedding_model:
    type: fastembed
    model_provider: fastembed
    model: ${FASTEMBED_MODEL_NAME}
    cache_dir: ${FASTEMBED_CACHE_DIR}
    batch_size: 64
```

Add to `.env.example`:

```dotenv
FASTEMBED_MODEL_NAME=BAAI/bge-small-en-v1.5
FASTEMBED_CACHE_DIR=.cache/fastembed
```

Add to Docker `ENV`:

```dockerfile
FASTEMBED_MODEL_NAME=BAAI/bge-small-en-v1.5 \
FASTEMBED_CACHE_DIR=/data/fastembed
```

- [ ] **Step 5: Run tests and configuration smoke test**

```bash
uv run pytest tests/mcp_server/retrieval/test_query_encoder.py tests/core/test_config.py -q
uv run python -c "from maf_graphrag.core.config import get_fastembed_model_name; assert get_fastembed_model_name() == 'BAAI/bge-small-en-v1.5'"
```

Expected: tests pass; command exits 0.

- [ ] **Step 6: Commit shared embedding configuration**

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
- Produces: `VectorMatch(id: str, score: float)` dataclass
- Produces: `LanceDbVectorStore(db_path: Path)`
- Produces: `LanceDbVectorStore.search(table_name: str, vector: list[float], limit: int) -> list[VectorMatch]`
- Score contract: `score = 1.0 / (1.0 + max(distance, 0.0))`, so larger is always better.

- [ ] **Step 1: Add LanceDB as a direct dependency**

Run:

```bash
uv add lancedb
```

Expected: `pyproject.toml` gains a direct `lancedb` dependency and `uv.lock` is updated without changing unrelated package constraints.

- [ ] **Step 2: Write failing adapter tests against a temporary LanceDB**

Use a real temporary database rather than mocking the query chain:

```python
import lancedb
import pyarrow as pa


def test_search_returns_ranked_ids_with_normalized_scores(tmp_path):
    db = lancedb.connect(tmp_path / "lancedb")
    db.create_table(
        "text_unit_text",
        data=pa.table(
            {
                "id": ["t1", "t2"],
                "vector": [[1.0, 0.0], [0.0, 1.0]],
                "text": ["alpha", "beta"],
            }
        ),
    )

    from maf_graphrag.mcp_server.retrieval.vector_store import LanceDbVectorStore

    results = LanceDbVectorStore(tmp_path / "lancedb").search("text_unit_text", [1.0, 0.0], 2)

    assert [match.id for match in results] == ["t1", "t2"]
    assert results[0].score > results[1].score
```

Add tests for a missing database/table that assert a clear `FileNotFoundError` or `LookupError` rather than exposing a low-level LanceDB traceback.

- [ ] **Step 3: Run adapter tests and verify RED**

```bash
uv run pytest tests/mcp_server/retrieval/test_vector_store.py -q
```

Expected: import failure because `vector_store.py` does not exist.

- [ ] **Step 4: Implement the adapter**

Use one raw LanceDB boundary:

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

- [ ] **Step 5: Run adapter tests and verify GREEN**

```bash
uv run pytest tests/mcp_server/retrieval/test_vector_store.py -q
```

Expected: all vector-store tests pass.

- [ ] **Step 6: Commit the vector adapter**

```bash
git add pyproject.toml uv.lock src/maf_graphrag/mcp_server/retrieval/vector_store.py tests/mcp_server/retrieval/test_vector_store.py
git commit -m "feat: add LanceDB MCP retrieval adapter"
```

---

### Task 4: Add Retrieval Response Types and Semantic Search Tools

**Files:**
- Modify: `src/maf_graphrag/mcp_server/tools/types.py`
- Create: `src/maf_graphrag/mcp_server/tools/retrieval_search.py`
- Modify: `src/maf_graphrag/mcp_server/tools/__init__.py`
- Create: `tests/mcp_server/tools/test_retrieval_search.py`
- Modify: `tests/mcp_server/tools/test_types.py`

**Interfaces:**
- Produces: `SemanticMatch`, `SemanticSearchResult`, `EntitySearchMatch`, `EntitySearchResult` TypedDicts.
- Produces: `semantic_search_tool(query: str, limit: int = 10) -> SemanticSearchResult | ToolError`
- Produces: `search_entities_tool(query: str, limit: int = 10) -> EntitySearchResult | ToolError`

- [ ] **Step 1: Write failing tests for semantic text-unit retrieval**

Test behavior through injected/mocked retrieval boundaries, not GraphRAG APIs:

```python
async def test_semantic_search_returns_hydrated_text_units(monkeypatch):
    import pandas as pd
    from maf_graphrag.core.data_loader import GraphData
    from maf_graphrag.mcp_server.retrieval.vector_store import VectorMatch

    data = GraphData(
        entities=pd.DataFrame(),
        relationships=pd.DataFrame(),
        communities=pd.DataFrame(),
        community_reports=pd.DataFrame(),
        text_units=pd.DataFrame([
            {"id": "t1", "text": "Project Alpha uses PostgreSQL", "document_id": "d1"},
        ]),
    )

    monkeypatch.setattr("maf_graphrag.mcp_server.tools.retrieval_search.get_graph_data", lambda: data)
    monkeypatch.setattr(
        "maf_graphrag.mcp_server.tools.retrieval_search.get_query_encoder",
        lambda: type("Encoder", (), {"encode": lambda self, text: [0.1, 0.2]})(),
    )
    monkeypatch.setattr(
        "maf_graphrag.mcp_server.tools.retrieval_search._get_vector_store",
        lambda: type("Store", (), {"search": lambda self, table, vector, limit: [VectorMatch("t1", 0.9)]})(),
    )

    from maf_graphrag.mcp_server.tools.retrieval_search import semantic_search_tool

    result = await semantic_search_tool("database", limit=5)

    assert result["matches"][0]["text_unit_id"] == "t1"
    assert result["matches"][0]["score"] == 0.9
    assert result["matches"][0]["text"] == "Project Alpha uses PostgreSQL"
```

Add validation tests for empty query and invalid limits.

- [ ] **Step 2: Write failing tests for semantic entity retrieval**

```python
async def test_search_entities_hydrates_entity_metadata(monkeypatch):
    # GraphData.entities contains id=e1/title=Project Alpha/type=project/description=...
    # vector store returns VectorMatch("e1", 0.88)
    # assert entity_id/name/type/description/community_ids/score are returned.
```

The concrete assertions must be:

```python
assert result["matches"] == [
    {
        "entity_id": "e1",
        "name": "Project Alpha",
        "type": "project",
        "description": "Main project",
        "community_ids": [1],
        "score": 0.88,
    }
]
```

- [ ] **Step 3: Run retrieval-tool tests and verify RED**

```bash
uv run pytest tests/mcp_server/tools/test_retrieval_search.py tests/mcp_server/tools/test_types.py -q
```

Expected: missing retrieval types/tool module.

- [ ] **Step 4: Add definitive retrieval TypedDicts**

In `types.py` add:

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

Keep `EntityInfo`, `EntityQueryResult`, and `ToolError`. Remove `SearchResult` only after server/wrapper migration in Task 7 so intermediate commits remain importable.

- [ ] **Step 5: Implement retrieval search tools**

Implement `retrieval_search.py` with these rules:

```python
TEXT_UNIT_TABLE = "text_unit_text"
ENTITY_TABLE = "entity_description"


def _get_vector_store() -> LanceDbVectorStore:
    return LanceDbVectorStore(MCPConfig.from_env().lancedb_dir)
```

`semantic_search_tool`:
1. call `validate_query(query)` and `validate_limit(limit)`;
2. `vector = get_query_encoder().encode(query)`;
3. `matches = _get_vector_store().search(TEXT_UNIT_TABLE, vector, limit)`;
4. hydrate rows by exact `text_units["id"]` match;
5. return `{"matches": ..., "returned": len(...), "query_type": "semantic_text"}`.

`search_entities_tool` follows the same flow against `ENTITY_TABLE`, hydrating `entities["id"]` and returning `query_type="semantic_entity"`.

If a vector-store ID no longer exists in the corresponding Parquet file, skip that stale vector row rather than fabricate evidence.

Decorate both with `@handle_tool_errors(...)` so missing index/vector errors become `ToolError`.

- [ ] **Step 6: Run focused retrieval tests and verify GREEN**

```bash
uv run pytest tests/mcp_server/tools/test_retrieval_search.py tests/mcp_server/tools/test_types.py -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit semantic retrieval tools**

```bash
git add src/maf_graphrag/mcp_server/tools/types.py src/maf_graphrag/mcp_server/tools/retrieval_search.py src/maf_graphrag/mcp_server/tools/__init__.py tests/mcp_server/tools/test_retrieval_search.py tests/mcp_server/tools/test_types.py
git commit -m "feat: add semantic MCP retrieval tools"
```

---

### Task 5: Add Direct Relationship Traversal

**Files:**
- Create: `src/maf_graphrag/mcp_server/tools/relationships.py`
- Modify: `src/maf_graphrag/mcp_server/tools/types.py`
- Modify: `src/maf_graphrag/mcp_server/tools/__init__.py`
- Create: `tests/mcp_server/tools/test_relationships.py`

**Interfaces:**
- Produces: `RelationshipInfo`, `RelationshipResult`.
- Produces: `get_relationships_tool(entity_name: str, limit: int = 20) -> RelationshipResult | ToolError`.

- [ ] **Step 1: Write failing traversal tests for both edge directions**

Use a graph fixture:

```python
entities = pd.DataFrame([
    {"id": "e1", "title": "Project Alpha"},
    {"id": "e2", "title": "Sarah Chen"},
    {"id": "e3", "title": "PostgreSQL"},
])
relationships = pd.DataFrame([
    {"source": "Project Alpha", "target": "Sarah Chen", "description": "led by", "weight": 2.0, "rank": 1},
    {"source": "PostgreSQL", "target": "Project Alpha", "description": "used by", "weight": 1.0, "rank": 2},
])
```

Assert:

```python
result = await get_relationships_tool("project alpha", limit=20)
assert [edge["counterpart"] for edge in result["relationships"]] == ["Sarah Chen", "PostgreSQL"]
assert result["relationships"][0]["direction"] == "outgoing"
assert result["relationships"][1]["direction"] == "incoming"
```

Also test unknown entity and limit validation.

- [ ] **Step 2: Run relationship tests and verify RED**

```bash
uv run pytest tests/mcp_server/tools/test_relationships.py -q
```

Expected: module/type import failure.

- [ ] **Step 3: Add relationship response types**

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
```

- [ ] **Step 4: Implement direct Parquet traversal**

`get_relationships_tool` must:
- validate `entity_name` and `limit`;
- resolve the canonical entity title by case-insensitive equality first, then case-insensitive contains only if exact equality finds nothing;
- filter relationship rows where `source == canonical_title` or `target == canonical_title`;
- preserve DataFrame order and stop at `limit`;
- populate `counterpart` and `direction` deterministically;
- include optional description/weight/rank only when the source row has a non-null value;
- return `ToolError(error=f"Entity not found: {entity_name}")` when no entity resolves.

- [ ] **Step 5: Run relationship tests and verify GREEN**

```bash
uv run pytest tests/mcp_server/tools/test_relationships.py -q
```

- [ ] **Step 6: Commit relationship traversal**

```bash
git add src/maf_graphrag/mcp_server/tools/relationships.py src/maf_graphrag/mcp_server/tools/types.py src/maf_graphrag/mcp_server/tools/__init__.py tests/mcp_server/tools/test_relationships.py
git commit -m "feat: add MCP relationship traversal"
```

---

### Task 6: Add Direct Source Resolution

**Files:**
- Create: `src/maf_graphrag/mcp_server/tools/sources.py`
- Modify: `src/maf_graphrag/mcp_server/tools/types.py`
- Modify: `src/maf_graphrag/mcp_server/tools/__init__.py`
- Modify: `src/maf_graphrag/mcp_server/tools/source_resolver.py`
- Create: `tests/mcp_server/tools/test_sources.py`
- Modify: `tests/mcp_server/tools/test_source_resolver.py`

**Interfaces:**
- Produces: `SourceInfo`, `SourceResult`.
- Produces: `get_sources_tool(source_ids: list[str], limit: int = 20) -> SourceResult | ToolError`.

- [ ] **Step 1: Write failing tests for caller order, de-duplication, missing IDs, and limit**

Fixture rows:

```python
text_units = pd.DataFrame([
    {"id": "tu-a", "human_readable_id": 0, "document_id": "doc-a", "text": "Alpha source text"},
    {"id": "tu-b", "human_readable_id": 1, "document_id": "doc-b", "text": "Beta source text"},
])
documents = pd.DataFrame([
    {"id": "doc-a", "title": "alpha.md", "text": "Full alpha document"},
    {"id": "doc-b", "title": "beta.md", "text": "Full beta document"},
])
```

Assert:

```python
result = await get_sources_tool(["tu-b", "tu-a", "tu-b", "missing"], limit=3)
assert [item["text_unit_id"] for item in result["sources"]] == ["tu-b", "tu-a"]
assert result["missing_ids"] == ["missing"]
assert result["sources"][0]["document_title"] == "beta.md"
```

- [ ] **Step 2: Run source tests and verify RED**

```bash
uv run pytest tests/mcp_server/tools/test_sources.py tests/mcp_server/tools/test_source_resolver.py -q
```

Expected: missing source-tool/types implementation.

- [ ] **Step 3: Add source response types**

```python
class SourceInfo(TypedDict):
    text_unit_id: str
    document_id: NotRequired[str]
    document_title: NotRequired[str]
    text_preview: NotRequired[str]


class SourceResult(TypedDict):
    sources: list[SourceInfo]
    missing_ids: list[str]
    returned: int
    query_type: str
```

- [ ] **Step 4: Implement ID-based source resolution**

Add a focused helper to `source_resolver.py` that resolves actual `text_units["id"]` values, rather than the old GraphRAG context `human_readable_id` path:

```python
def resolve_text_unit_ids(source_ids: list[str], data: GraphData, limit: int) -> tuple[list[dict], list[str]]:
    # de-duplicate while preserving order
    # index text_units by string id
    # map document_id to documents.title
    # build at most limit results
    # collect unresolved IDs in caller order
```

`get_sources_tool` validates `1 <= limit <= MAX_LIMIT`, rejects an empty `source_ids` list with `ToolError(error="source_ids must not be empty.")`, calls the helper, and returns `query_type="source_lookup"`.

- [ ] **Step 5: Run source tests and verify GREEN**

```bash
uv run pytest tests/mcp_server/tools/test_sources.py tests/mcp_server/tools/test_source_resolver.py -q
```

- [ ] **Step 6: Commit source lookup**

```bash
git add src/maf_graphrag/mcp_server/tools/sources.py src/maf_graphrag/mcp_server/tools/source_resolver.py src/maf_graphrag/mcp_server/tools/types.py src/maf_graphrag/mcp_server/tools/__init__.py tests/mcp_server/tools/test_sources.py tests/mcp_server/tools/test_source_resolver.py
git commit -m "feat: add MCP source retrieval"
```

---

### Task 7: Replace the Public MCP Surface

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
- Public MCP functions become exactly:
  - `semantic_search(query: str, limit: int = 10)`
  - `search_entities(query: str, limit: int = 10)`
  - `get_entity(entity_name: str)`
  - `get_relationships(entity_name: str, limit: int = 20)`
  - `get_sources(source_ids: list[str], limit: int = 20)`

- [ ] **Step 1: Replace server tests with failing retrieval-only dispatch tests**

Use direct function dispatch assertions:

```python
async def test_semantic_search_forwards_to_retrieval_tool():
    from maf_graphrag.mcp_server.server import semantic_search

    expected = {"matches": [], "returned": 0, "query_type": "semantic_text"}
    with patch(
        "maf_graphrag.mcp_server.server.semantic_search_tool",
        AsyncMock(return_value=expected),
    ) as tool:
        result = await semantic_search("database", limit=7)

    tool.assert_awaited_once_with("database", 7)
    assert result is expected
```

Add equivalent dispatch tests for `search_entities`, `get_entity`, `get_relationships`, and `get_sources`.

Add a public MCP discovery test through FastMCP's client API:

```python
from fastmcp import Client


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

- [ ] **Step 2: Run server tests and verify RED**

```bash
uv run pytest tests/mcp_server/test_server.py -q
```

Expected: current server still advertises generative tools and does not expose the new retrieval functions.

- [ ] **Step 3: Rewire `server.py` to the five approved tools**

Imports must be retrieval-only:

```python
from maf_graphrag.mcp_server.tools import (
    entity_query_tool,
    get_relationships_tool,
    get_sources_tool,
    search_entities_tool,
    semantic_search_tool,
)
```

Register only the five functions named in the interface block. Remove imports of `DEFAULT_RESPONSE_TYPE`, `local_search_tool`, and `global_search_tool` from `server.py`.

`get_entity` continues using:

```python
return await entity_query_tool(entity_name=entity_name, limit=1)
```

Do not register `list_entities`.

- [ ] **Step 4: Remove obsolete MCP generative wrappers and answer-oriented type**

Delete the local/global MCP modules and tests. Remove their exports from `tools/__init__.py`. Remove `SearchContext` and `SearchResult` from `tools/types.py` after verifying no remaining MCP import needs them.

Do not delete or alter `src/maf_graphrag/core/search.py`.

- [ ] **Step 5: Update agent-side MCP description**

In `agents/factories.py`, replace the MCP description with:

```python
description="Retrieve structured evidence, entities, relationships, and sources from the GraphRAG knowledge graph"
```

This tells the chat/router LLM to reason over MCP evidence rather than expecting MCP to synthesize the final answer.

- [ ] **Step 6: Run server and MCP-tool tests and verify GREEN**

```bash
uv run pytest tests/mcp_server/test_server.py tests/mcp_server/tools -q
```

Expected: all MCP tests pass; discovery returns exactly the five retrieval tools.

- [ ] **Step 7: Commit the MCP API replacement**

```bash
git add -A src/maf_graphrag/mcp_server src/maf_graphrag/agents/factories.py tests/mcp_server
git commit -m "feat: replace MCP search with retrieval-only tools"
```

---

### Task 8: Add a Hard No-LLM Runtime Regression Test

**Files:**
- Create: `tests/mcp_server/test_no_llm_runtime.py`

**Interfaces:**
- Verifies the public runtime contract only; produces no production API.

- [ ] **Step 1: Write a regression test that makes completion use impossible**

The test must explicitly remove llama.cpp environment variables and make generative GraphRAG entry points fatal if touched:

```python
async def test_all_mcp_retrieval_tools_work_without_completion_model(monkeypatch, retrieval_fixture):
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    monkeypatch.delenv("LLAMA_CPP_MODEL", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("MCP retrieval attempted to call a generative GraphRAG API")

    monkeypatch.setattr("graphrag.api.local_search", forbidden)
    monkeypatch.setattr("graphrag.api.global_search", forbidden)
    monkeypatch.setattr("graphrag.api.basic_search", forbidden)
    monkeypatch.setattr("graphrag.api.drift_search", forbidden)
```

Use the test fixture to patch only the physical embedding/vector-store boundaries so the test does not download a model:

```python
monkeypatch.setattr(
    "maf_graphrag.mcp_server.tools.retrieval_search.get_query_encoder",
    lambda: FakeEncoder([0.1, 0.2]),
)
monkeypatch.setattr(
    "maf_graphrag.mcp_server.tools.retrieval_search._get_vector_store",
    lambda: FakeVectorStore(...),
)
```

Then call all five server functions and assert none return a `ToolError`.

- [ ] **Step 2: Run the regression test**

```bash
uv run pytest tests/mcp_server/test_no_llm_runtime.py -q
```

Expected: PASS only when no MCP path reaches generative GraphRAG/completion configuration.

- [ ] **Step 3: Run a static import guard**

Run:

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

- [ ] **Step 4: Commit the no-LLM regression guard**

```bash
git add tests/mcp_server/test_no_llm_runtime.py
git commit -m "test: prove MCP retrieval needs no completion LLM"
```

---

### Task 9: Update MCP and Container Documentation

**Files:**
- Modify: `README.md`
- Modify: `src/maf_graphrag/mcp_server/README.md`

**Interfaces:**
- Documentation contract mirrors the exact five-tool MCP API.

- [ ] **Step 1: Update top-level architecture documentation**

Add an explicit lifecycle section:

```text
Indexing:
  Documents -> Gemma/llama.cpp (graph extraction/summaries)
            -> FastEmbed (vectors)
            -> Parquet + LanceDB

MCP runtime after indexing:
  MCP client -> FastEmbed query embedding -> LanceDB/Parquet -> structured evidence
  No completion LLM or llama.cpp call is required.
```

Document that the Docker entrypoint still waits for llama.cpp only when the index is missing.

- [ ] **Step 2: Replace the MCP tool table/examples**

Document exactly:

```text
semantic_search(query, limit=10)
search_entities(query, limit=10)
get_entity(entity_name)
get_relationships(entity_name, limit=20)
get_sources(source_ids, limit=20)
```

Include one example sequence for an MCP host:

```text
1. semantic_search("Who leads Project Alpha?")
2. get_entity("Project Alpha")
3. get_relationships("Project Alpha")
4. get_sources(["<text-unit-id>"])
5. Host LLM synthesizes the answer from returned evidence.
```

- [ ] **Step 3: Add migration warning**

State that this release intentionally removes the MCP tools `search_knowledge_graph`, `local_search`, `global_search`, and `list_entities`; callers should migrate to the retrieval primitives.

- [ ] **Step 4: Run documentation/reference grep**

```bash
rg -n "search_knowledge_graph|local_search\(|global_search\(|list_entities\(" README.md src/maf_graphrag/mcp_server src/maf_graphrag/agents tests/mcp_server
```

Expected: no stale MCP registration/documentation references. References inside `core/search.py`, its tests, or historical Superpowers docs are allowed and should not be changed.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md src/maf_graphrag/mcp_server/README.md
git commit -m "docs: document retrieval-only MCP runtime"
```

---

### Task 10: Full Verification and Container Runtime Check

**Files:**
- No production changes expected; fix only defects revealed by verification.

**Interfaces:**
- Final acceptance against the approved design spec.

- [ ] **Step 1: Run formatting and lint checks**

```bash
uv run ruff format --check .
uv run ruff check .
```

Expected: both exit 0.

- [ ] **Step 2: Run static typing**

```bash
uv run mypy src
```

Expected: exit 0.

- [ ] **Step 3: Run the full test suite**

```bash
uv run pytest
```

Expected: all tests pass with zero failures.

- [ ] **Step 4: Build the Docker image from the feature branch**

```bash
docker build -t hiu-graph:mcp-retrieval-only .
```

Expected: image build exits 0.

- [ ] **Step 5: Verify an existing-index container starts with llama.cpp unreachable**

Use a complete indexed `output/` directory and deliberately point the runtime at an unreachable endpoint:

```bash
docker run --rm \
  -p 8011:8011 \
  -e LLAMA_CPP_BASE_URL=http://127.0.0.1:9 \
  -e LLAMA_CPP_MODEL= \
  -v "$PWD/output:/app/output:ro" \
  hiu-graph:mcp-retrieval-only
```

Expected: entrypoint prints `GraphRAG index is ready; skipping indexing.` and MCP starts on port 8011 without attempting llama.cpp health checks.

- [ ] **Step 6: Invoke all five MCP tools against the running container**

Use MCP Inspector or the repository's FastMCP client test harness to invoke:

```text
semantic_search("Project Alpha", 3)
search_entities("Project Alpha", 3)
get_entity("Project Alpha")
get_relationships("Project Alpha", 5)
get_sources(["<id returned by semantic_search>"], 5)
```

Expected: all return structured retrieval data; container logs contain no `LiteLLM completion()` lines.

- [ ] **Step 7: Verify first-run indexing behavior was not regressed**

Run the existing entrypoint tests:

```bash
uv run pytest tests/test_docker_entrypoint.py -q
```

Expected: all auto-index tests pass, preserving `index missing -> wait for llama.cpp -> build index -> start MCP`.

- [ ] **Step 8: Review branch diff against the approved spec**

```bash
git diff master...HEAD --stat
git diff master...HEAD -- src/maf_graphrag/mcp_server settings.yaml Dockerfile README.md
```

Checklist:
- public MCP surface is exactly five retrieval tools;
- no MCP completion-model calls remain;
- Gemma indexing configuration remains intact;
- FastEmbed model is shared by indexing and MCP retrieval;
- generative `core/search.py` remains available;
- existing-index MCP runtime does not require llama.cpp.

- [ ] **Step 9: Commit any verification-only fixes, if required**

If verification required a code fix, rerun Steps 1-7 and then commit the specific corrected files with a descriptive message. If no fix was required, do not create an empty commit.
