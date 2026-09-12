"""Tests for the MCP FastEmbed query encoder."""

from unittest.mock import MagicMock, patch


def test_query_encoder_uses_shared_fastembed_settings(monkeypatch):
    monkeypatch.setenv("FASTEMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("FASTEMBED_CACHE_DIR", "/tmp/fastembed-test")

    fake_vector = MagicMock()
    fake_vector.tolist.return_value = [0.1, 0.2, 0.3]
    fake_model = MagicMock()
    fake_model.embed.return_value = iter([fake_vector])

    with patch("maf_graphrag.mcp_server.retrieval.query_encoder.TextEmbedding", return_value=fake_model) as ctor:
        from maf_graphrag.mcp_server.retrieval.query_encoder import FastEmbedQueryEncoder

        vector = FastEmbedQueryEncoder().encode("hello")

    ctor.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5", cache_dir="/tmp/fastembed-test")
    fake_model.embed.assert_called_once_with(["hello"])
    assert vector == [0.1, 0.2, 0.3]


def test_get_query_encoder_is_cached():
    from maf_graphrag.mcp_server.retrieval import query_encoder

    query_encoder.get_query_encoder.cache_clear()
    instance = MagicMock()
    with patch.object(query_encoder, "FastEmbedQueryEncoder", return_value=instance) as ctor:
        first = query_encoder.get_query_encoder()
        second = query_encoder.get_query_encoder()

    assert first is second is instance
    ctor.assert_called_once_with()
    query_encoder.get_query_encoder.cache_clear()
