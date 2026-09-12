"""Configuration loader for GraphRAG with Pydantic-backed validation."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator

DEFAULT_FASTEMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_FASTEMBED_CACHE_DIR = ".cache/fastembed"


def get_root_dir() -> Path:
    """Get the project root directory (where settings.yaml is located)."""
    # Start from this file and go up to find settings.yaml
    current = Path(__file__).parent.parent

    if (current / "settings.yaml").exists():
        return current

    # Fallback: try current working directory
    cwd = Path.cwd()
    if (cwd / "settings.yaml").exists():
        return cwd

    raise FileNotFoundError("Could not find settings.yaml. Make sure you're running from the project root directory.")


def get_fastembed_model_name() -> str:
    """Return the shared FastEmbed model used for indexing and MCP retrieval."""
    return os.getenv("FASTEMBED_MODEL_NAME", DEFAULT_FASTEMBED_MODEL_NAME).strip() or DEFAULT_FASTEMBED_MODEL_NAME


def get_fastembed_cache_dir() -> str:
    """Return the shared FastEmbed cache directory."""
    return os.getenv("FASTEMBED_CACHE_DIR", DEFAULT_FASTEMBED_CACHE_DIR).strip() or DEFAULT_FASTEMBED_CACHE_DIR


class CoreEnvConfig(BaseModel):
    """Environment-backed configuration for the local llama.cpp runtime."""

    llama_cpp_base_url: str = Field(..., description="llama.cpp server URL")
    llama_cpp_model: str = Field(..., description="Local GGUF model path")

    @field_validator("llama_cpp_base_url", "llama_cpp_model")
    @classmethod
    def _ensure_non_empty(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        if not value or not value.strip():
            raise ValueError(f"{field_name.replace('_', ' ').upper()} must be set in the environment")
        return value.strip()

    @classmethod
    def from_env(cls) -> "CoreEnvConfig":
        """Load configuration from environment variables."""

        load_dotenv()
        data = {
            "llama_cpp_base_url": os.getenv("LLAMA_CPP_BASE_URL"),
            "llama_cpp_model": os.getenv("LLAMA_CPP_MODEL"),
        }
        try:
            return cls.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - defensive guard
            raise OSError(f"Invalid environment configuration: {exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> GraphRagConfig:
    """Load GraphRAG configuration from settings.yaml after validating environment variables."""

    CoreEnvConfig.from_env()
    os.environ.setdefault("FASTEMBED_MODEL_NAME", get_fastembed_model_name())
    os.environ.setdefault("FASTEMBED_CACHE_DIR", get_fastembed_cache_dir())
    root_dir = get_root_dir()
    return load_config(root_dir=root_dir)


def get_output_dir() -> Path:
    """Get the output directory where Parquet files are stored."""
    config = get_config()
    root = get_root_dir()

    # GraphRAG 3.x uses output_storage instead of storage
    output_base = getattr(config.output_storage, "base_dir", "output")
    return root / output_base


def validate_output_files(required: list[str] | None = None, output_dir: Path | None = None) -> bool:
    """
    Check if required output files exist.

    Args:
        required: List of required file names (without path).
                  Defaults to core files needed for search.
        output_dir: Explicit output directory. When omitted, uses GraphRAG config.

    Returns:
        True if all files exist, False otherwise.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    if required is None:
        # GraphRAG 3.x output file names (no create_final_ prefix)
        required = [
            "entities.parquet",
            "relationships.parquet",
            "communities.parquet",
            "community_reports.parquet",
            "text_units.parquet",
        ]

    resolved_output_dir = output_dir if output_dir is not None else get_output_dir()
    missing = [f for f in required if not (resolved_output_dir / f).exists()]

    if missing:
        raise FileNotFoundError(
            f"Missing required output files: {', '.join(missing)}\n"
            f"Please run indexing first: uv run python -m maf_graphrag.core.index"
        )

    return True
