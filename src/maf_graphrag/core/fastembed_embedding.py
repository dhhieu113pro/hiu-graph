"""Local FastEmbed adapter for GraphRAG's embedding factory."""

import asyncio
from typing import TYPE_CHECKING, Any, Unpack

from fastembed import TextEmbedding
from graphrag_llm.embedding import LLMEmbedding as BaseLLMEmbedding
from graphrag_llm.embedding import register_embedding
from graphrag_llm.types import LLMEmbeddingResponse, LLMEmbeddingUsage
from openai.types.embedding import Embedding

if TYPE_CHECKING:
    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsStore
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import LLMEmbeddingArgs


class FastEmbedEmbedding(BaseLLMEmbedding):
    """Synchronous local embeddings with an async compatibility wrapper."""

    def __init__(
        self,
        *,
        model_config: "ModelConfig",
        tokenizer: "Tokenizer",
        metrics_store: "MetricsStore",
        **kwargs: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        extra = model_config.model_extra or {}
        self._model_name = model_config.model
        self._model = TextEmbedding(
            model_name=self._model_name,
            cache_dir=extra.get("cache_dir"),
        )
        self._batch_size = int(extra.get("batch_size", 64))

    def embedding(self, /, **kwargs: Unpack["LLMEmbeddingArgs"]) -> LLMEmbeddingResponse:
        """Generate local embeddings for a batch of input strings."""
        inputs = kwargs.get("input") or []
        vectors = list(self._model.embed(inputs, batch_size=self._batch_size))
        data = [
            Embedding(object="embedding", embedding=vector.tolist(), index=index)
            for index, vector in enumerate(vectors)
        ]
        return LLMEmbeddingResponse(
            object="list",
            data=data,
            model="fastembed/" + self._model_name,
            usage=LLMEmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )

    async def embedding_async(self, /, **kwargs: Unpack["LLMEmbeddingArgs"]) -> LLMEmbeddingResponse:
        """Run the CPU-bound local embedding operation off the event loop."""
        return await asyncio.to_thread(self.embedding, **kwargs)

    @property
    def metrics_store(self) -> "MetricsStore":
        return self._metrics_store

    @property
    def tokenizer(self) -> "Tokenizer":
        return self._tokenizer


register_embedding("fastembed", FastEmbedEmbedding, scope="singleton")
