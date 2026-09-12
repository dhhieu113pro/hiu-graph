# Docker Auto-Index Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Hiu Graph Docker image automatically build a missing GraphRAG index before starting MCP, while preserving fast startup when a complete index already exists.

**Architecture:** Add a small, testable Python entrypoint responsible only for container startup preparation. It checks the required index artifacts, waits for the external llama.cpp health endpoint only when indexing is necessary, invokes the existing `maf_graphrag.core.index` module, revalidates output, and `exec`s the existing MCP runner. Docker then copies source input and launches this entrypoint.

**Tech Stack:** Python 3.12 stdlib, pytest, Docker, uv, GraphRAG, llama.cpp.

**Spec:** `docs/superpowers/specs/2026-09-12-docker-auto-index-design.md`

## Global Constraints

- A complete existing index must start MCP without waiting for llama.cpp readiness.
- Missing indexes must be built before MCP starts.
- Required artifacts are `entities.parquet`, `relationships.parquet`, `communities.parquet`, `community_reports.parquet`, `text_units.parquet`, and `lancedb` under `output/`.
- First-run indexing uses the configured external `LLAMA_CPP_BASE_URL`; no model is bundled into the container.
- The llama.cpp wait timeout defaults to 300 seconds and is configurable with `HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS`.
- After successful preparation, MCP must replace the entrypoint process via `os.execv`.
- The image must contain `input/` so it can index when `output/` is absent.

---

### Task 1: Add a tested Docker startup entrypoint

**Files:**
- Create: `docker_entrypoint.py`
- Create: `tests/test_docker_entrypoint.py`

**Interfaces:**
- Produces: `index_is_ready(root: Path) -> bool`
- Produces: `wait_for_llama_cpp(base_url: str, timeout_seconds: float) -> None`
- Produces: `run_indexing(root: Path) -> None`
- Produces: `prepare_index(root: Path) -> None`
- Produces: `main() -> None`

- [ ] **Step 1: Write failing readiness tests**

Create `tests/test_docker_entrypoint.py` with tests that build the six required artifacts in a temporary directory and assert `index_is_ready(tmp_path)` is true, then remove one artifact and assert it is false.

```python
from pathlib import Path

from docker_entrypoint import REQUIRED_INDEX_PATHS, index_is_ready


def create_complete_index(root: Path) -> None:
    for relative_path in REQUIRED_INDEX_PATHS:
        path = root / relative_path
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ready", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)


def test_index_is_ready_when_all_required_artifacts_exist(tmp_path: Path) -> None:
    create_complete_index(tmp_path)
    assert index_is_ready(tmp_path)


def test_index_is_not_ready_when_one_artifact_is_missing(tmp_path: Path) -> None:
    create_complete_index(tmp_path)
    (tmp_path / "output/entities.parquet").unlink()
    assert not index_is_ready(tmp_path)
```

- [ ] **Step 2: Run the readiness tests and verify they fail**

Run: `uv run pytest tests/test_docker_entrypoint.py -v`

Expected: FAIL because `docker_entrypoint` does not exist.

- [ ] **Step 3: Implement readiness detection**

Create `docker_entrypoint.py` with:

```python
from pathlib import Path

REQUIRED_INDEX_PATHS = (
    Path("output/entities.parquet"),
    Path("output/relationships.parquet"),
    Path("output/communities.parquet"),
    Path("output/community_reports.parquet"),
    Path("output/text_units.parquet"),
    Path("output/lancedb"),
)


def index_is_ready(root: Path) -> bool:
    return all((root / relative_path).exists() for relative_path in REQUIRED_INDEX_PATHS)
```

- [ ] **Step 4: Run the readiness tests and verify they pass**

Run: `uv run pytest tests/test_docker_entrypoint.py -v`

Expected: the two readiness tests PASS.

- [ ] **Step 5: Add failing startup preparation tests**

Extend `tests/test_docker_entrypoint.py` using `unittest.mock.patch` and `monkeypatch` to prove:

```python
def test_prepare_index_skips_wait_and_index_when_ready(tmp_path: Path) -> None:
    create_complete_index(tmp_path)
    with patch("docker_entrypoint.wait_for_llama_cpp") as wait_mock, patch(
        "docker_entrypoint.run_indexing"
    ) as index_mock:
        prepare_index(tmp_path)
    wait_mock.assert_not_called()
    index_mock.assert_not_called()


def test_prepare_index_waits_and_indexes_when_missing(tmp_path: Path) -> None:
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://llama:8080")
    monkeypatch.setenv("HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS", "12")
    with patch("docker_entrypoint.wait_for_llama_cpp") as wait_mock, patch(
        "docker_entrypoint.run_indexing"
    ) as index_mock, patch("docker_entrypoint.index_is_ready", side_effect=[False, True]):
        prepare_index(tmp_path)
    wait_mock.assert_called_once_with("http://llama:8080", 12.0)
    index_mock.assert_called_once_with(tmp_path)


def test_prepare_index_fails_if_output_is_still_incomplete(tmp_path: Path) -> None:
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://llama:8080")
    with patch("docker_entrypoint.wait_for_llama_cpp"), patch("docker_entrypoint.run_indexing"), patch(
        "docker_entrypoint.index_is_ready", side_effect=[False, False]
    ):
        with pytest.raises(RuntimeError, match="GraphRAG index is incomplete"):
            prepare_index(tmp_path)
```

Also test `wait_for_llama_cpp` with mocked `urlopen` and `time.sleep`, test `run_indexing` calls `subprocess.run([sys.executable, "-m", "maf_graphrag.core.index"], cwd=root, check=True)`, and test `main()` calls `os.execv` only after `prepare_index` returns successfully.

- [ ] **Step 6: Run the startup preparation tests and verify they fail**

Run: `uv run pytest tests/test_docker_entrypoint.py -v`

Expected: FAIL because startup functions are not implemented yet.

- [ ] **Step 7: Implement the minimal startup lifecycle**

Add stdlib implementations using `urllib.request.urlopen`, `time.monotonic`, `subprocess.run`, and `os.execv`. Use `LLAMA_CPP_BASE_URL` default `http://127.0.0.1:8080`, `HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS` default `300`, and `GRAPHRAG_ROOT` default `.`. Log concise status messages to stdout with `print(..., flush=True)`.

- [ ] **Step 8: Run the entrypoint tests and verify they pass**

Run: `uv run pytest tests/test_docker_entrypoint.py -v`

Expected: all entrypoint tests PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add docker_entrypoint.py tests/test_docker_entrypoint.py
git commit -m "feat: auto-index Docker startup"
```

### Task 2: Wire auto-index startup into the image

**Files:**
- Modify: `Dockerfile`
- Modify: `.env.example`

**Interfaces:**
- Consumes: `docker_entrypoint.py` from Task 1.
- Produces: container startup through `docker_entrypoint.py`.

- [ ] **Step 1: Update Docker image contents and command**

Change the Dockerfile copy section so it includes both startup scripts and source input:

```dockerfile
COPY settings.yaml .env.example run_mcp_server.py docker_entrypoint.py ./
COPY prompts ./prompts
COPY input ./input
COPY src ./src
COPY output ./output
```

Replace the existing CMD with:

```dockerfile
CMD ["uv", "run", "--no-sync", "python", "docker_entrypoint.py"]
```

Keep port `8011` and the existing MCP health check unchanged; MCP will not listen until indexing completes.

- [ ] **Step 2: Document the timeout environment variable**

Add to `.env.example` next to the llama.cpp configuration:

```dotenv
# Maximum first-run wait for llama.cpp when the GraphRAG index is missing.
HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS=300
```

- [ ] **Step 3: Run focused validation**

Run:

```bash
uv run pytest tests/test_docker_entrypoint.py -v
uv run ruff check docker_entrypoint.py tests/test_docker_entrypoint.py
```

Expected: both commands succeed.

- [ ] **Step 4: Commit Task 2**

```bash
git add Dockerfile .env.example
git commit -m "feat: make container first-run ready"
```

### Task 3: Document ready-to-use container behavior and verify regressions

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents the behavior implemented by Tasks 1 and 2.

- [ ] **Step 1: Update the container documentation**

Replace the statement that the image always contains an index with wording that covers both startup paths:

```markdown
On startup the container checks the GraphRAG index. If the required index artifacts are already present, MCP starts immediately. If the index is missing or an empty `/app/output` volume hides the baked index, Hiu Graph waits for `LLAMA_CPP_BASE_URL`, builds the index from `/app/input`, verifies it, and then starts MCP.

For first-run indexing, llama.cpp must already be reachable from the container. `HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS` controls how long startup waits for it (default `300`).
```

Add an optional persistence example using named volumes for `/app/output`, `/app/cache`, and `/data/fastembed`, while retaining the simple `docker run` example.

- [ ] **Step 2: Run the full project test suite**

Run: `uv run pytest`

Expected: all tests pass.

- [ ] **Step 3: Run lint for changed Python files**

Run: `uv run ruff check docker_entrypoint.py tests/test_docker_entrypoint.py`

Expected: no lint errors.

- [ ] **Step 4: Commit Task 3**

```bash
git add README.md
git commit -m "docs: explain Docker auto-index startup"
```

### Task 4: Review the complete branch

**Files:**
- Review all files changed relative to `master`.

**Interfaces:**
- Produces a PR-ready branch with implementation, tests, and documentation.

- [ ] **Step 1: Inspect the branch diff**

Run: `git diff master...HEAD --check` and `git diff master...HEAD`.

Expected: no whitespace errors; changes are limited to the Docker auto-index feature.

- [ ] **Step 2: Re-run verification**

Run:

```bash
uv run pytest
uv run ruff check docker_entrypoint.py tests/test_docker_entrypoint.py
```

Expected: all commands succeed.

- [ ] **Step 3: Open a pull request**

Open a PR from `feat/docker-auto-index` to `master` summarizing first-run indexing, fast-path startup for existing indexes, the external llama.cpp dependency, and test coverage.