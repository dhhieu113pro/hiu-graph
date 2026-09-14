"""Agent configuration for the local llama.cpp runtime."""

from __future__ import annotations

import os

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    ValidationError,
    field_validator,
)


class AgentConfig(BaseModel):
    """Agent configuration backed by Pydantic validation."""

    llama_cpp_base_url: AnyHttpUrl = Field(..., description="Local llama.cpp server URL")
    model: str = Field(..., description="GGUF model served by llama.cpp")
    mcp_server_url: str = Field(default="http://127.0.0.1:8011/mcp", description="GraphRAG MCP server URL")
    router_model: str | None = Field(default=None, description="Optional model override for routing")

    @field_validator("model")
    @classmethod
    def _ensure_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("MODEL must be provided")
        return value.strip()

    @property
    def router_base_url(self) -> str:
        """Return the local llama.cpp URL used for router traffic."""

        return str(self.llama_cpp_base_url).rstrip("/")

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Create configuration from environment variables with strict validation."""

        from dotenv import load_dotenv

        load_dotenv()
        data = {
            "llama_cpp_base_url": os.getenv("LLAMA_CPP_BASE_URL"),
            "model": os.getenv("LLAMA_CPP_MODEL_NAME"),
            "mcp_server_url": os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8011/mcp"),
            "router_model": os.getenv("LLAMA_CPP_ROUTER_MODEL") or None,
        }

        try:
            return cls.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - defensive guard
            raise ValueError(str(exc)) from exc

    @property
    def router_model_name(self) -> str:
        """Return the model used for routing."""

        return self.router_model or self.model

    def validate_mcp_server(self) -> bool:
        """Return True when the MCP server URL looks valid."""

        return self.mcp_server_url.startswith("http")


class SessionConfig(BaseModel):
    """Runtime session settings for chatbot multi-turn memory behavior."""

    ttl_seconds: int = Field(default=1800, ge=60, le=86400)
    max_count: int = Field(default=1000, ge=1, le=50000)
    cleanup_interval_seconds: int = Field(default=60, ge=1, le=3600)
    max_history_groups: int = Field(default=12, ge=1, le=200)

    @classmethod
    def from_env(cls) -> SessionConfig:
        """Create validated session settings from environment variables."""

        data = {
            "ttl_seconds": os.getenv("SESSION_TTL_SECONDS", "1800"),
            "max_count": os.getenv("SESSION_MAX_COUNT", "1000"),
            "cleanup_interval_seconds": os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", "60"),
            "max_history_groups": os.getenv("SESSION_MAX_HISTORY_GROUPS", "12"),
        }

        try:
            return cls.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - defensive guard
            raise ValueError(str(exc)) from exc


def get_agent_config() -> AgentConfig:
    """Get validated agent configuration from environment."""

    return AgentConfig.from_env()


def get_session_config() -> SessionConfig:
    """Get validated runtime session settings from environment."""

    return SessionConfig.from_env()
