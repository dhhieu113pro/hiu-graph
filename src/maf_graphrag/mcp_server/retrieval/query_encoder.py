"""FastEmbed query encoder for MCP retrieval."""

from functools import lru_cache

from fastembed import TextEmbedding

from maf_graphrag.core.config import get_fastembed_cache_dir, get_fastembed_model_name


class FastEmbedQueryEncoder:
    """Encode MCP search queries with the same FastEmbed model used at index time."""

    def __init__(self) -> None:
        self._model = TextEmbedding(
            model_name=get_fastembed_model_name(),
            cache_dir=get_fastembed_cache_dir(),
        )

    def encode(self, text: str) -> list[float]:
        """Encode one query string into a dense vector."""
        vector = next(iter(self._model.embed([text])))
        return vector.tolist()


@lru_cache(maxsize=1)
def get_query_encoder() -> FastEmbedQueryEncoder:
    """Return the process-wide query encoder singleton."""
    return FastEmbedQueryEncoder()
