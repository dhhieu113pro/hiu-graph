# Docker Auto-Index Startup Design

## Goal

Make the published Hiu Graph container ready to use on first launch even when its GraphRAG `output/` index is missing or hidden by an empty volume.

## Current behavior

The image starts `run_mcp_server.py` immediately. It copies a prebuilt `output/` directory into the image, but it does not copy `input/`. If the index is missing or an empty output volume masks the baked index, MCP starts without first building the required GraphRAG artifacts.

The local PowerShell bootstrap already has the desired lifecycle: detect missing index artifacts, wait for llama.cpp, build the index, then start MCP.

## Proposed behavior

Add a Python Docker entrypoint that runs before MCP:

1. Resolve the GraphRAG root from `GRAPHRAG_ROOT` (default `/app` in the image).
2. Check for all required runtime index artifacts:
   - `output/entities.parquet`
   - `output/relationships.parquet`
   - `output/communities.parquet`
   - `output/community_reports.parquet`
   - `output/text_units.parquet`
   - `output/lancedb`
3. If every artifact exists, skip indexing and start MCP immediately.
4. If any artifact is missing, wait for `${LLAMA_CPP_BASE_URL}/health` to return HTTP 200. The wait timeout is controlled by `HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS` and defaults to 300 seconds.
5. Run `python -m maf_graphrag.core.index` using the same Python interpreter as the entrypoint.
6. Verify the required artifacts again. If the index is still incomplete, exit non-zero instead of starting an unusable MCP server.
7. Replace the entrypoint process with `run_mcp_server.py` using `os.execv` so the MCP server becomes PID 1 and receives container signals directly.

## Image contents

The Docker image must copy `input/` as well as `output/`. This ensures first-run indexing has source documents when the baked index is absent. Users may still mount their own `/app/input` and `/app/output` volumes.

No llama.cpp model is bundled in this image. First-run indexing uses the configured external llama.cpp endpoint through `LLAMA_CPP_BASE_URL` and `LLAMA_CPP_MODEL_NAME`; FastEmbed continues to provide embeddings according to `settings.yaml`.

## Failure behavior

- Existing complete index: no llama.cpp readiness wait is added at startup; MCP starts immediately.
- Missing index and llama.cpp unavailable: log the wait state and exit after the configured timeout.
- Indexing command fails: propagate the failure and do not start MCP.
- Indexing exits successfully but required artifacts are still missing: raise an explicit error and do not start MCP.

## Testing

Add unit tests for the entrypoint with filesystem, HTTP, subprocess, sleep, and `os.execv` operations mocked as needed. Tests must prove:

- complete indexes are detected;
- missing artifacts make an index incomplete;
- complete indexes skip llama.cpp waiting and indexing;
- missing indexes wait for llama.cpp and invoke the indexer;
- the index is revalidated after indexing;
- MCP is exec'd only after startup preparation succeeds.

Existing GraphRAG and MCP tests remain unchanged and must still pass.

## Documentation

Update `.env.example` with `HIU_GRAPH_LLM_WAIT_TIMEOUT_SECONDS=300` and update the container README to explain first-run automatic indexing, external llama.cpp dependency, and optional persistent `/app/output` and `/app/cache` volumes.