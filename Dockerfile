FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8011 \
    GRAPHRAG_ROOT=/app \
    FASTEMBED_CACHE_DIR=/data/fastembed

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY settings.yaml .env.example run_mcp_server.py docker_entrypoint.py ./
COPY prompts ./prompts
COPY input ./input
COPY src ./src
COPY output ./output

RUN mkdir -p /app/output /app/cache /data/fastembed

EXPOSE 8011

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import socket; s=socket.create_connection(('127.0.0.1', 8011), 3); s.close()" || exit 1

CMD ["uv", "run", "--no-sync", "python", "docker_entrypoint.py"]
