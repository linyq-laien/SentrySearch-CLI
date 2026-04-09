"""Tests for sentrysearch.qwen_embedder."""

import os
import types
from unittest.mock import MagicMock, patch

import pytest


def _install_fake_dashscope(monkeypatch, *, multimodal_cls=None):
    fake_module = types.SimpleNamespace(MultiModalEmbedding=multimodal_cls)
    monkeypatch.setitem(__import__("sys").modules, "dashscope", fake_module)


class TestQwenEmbedder:
    def test_raises_without_api_key(self, monkeypatch):
        from sentrysearch.qwen_embedder import QwenAPIKeyError, QwenEmbedder

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(QwenAPIKeyError, match="DASHSCOPE_API_KEY"):
                QwenEmbedder()

    def test_embed_query_calls_multimodal_embedding(self, monkeypatch):
        from sentrysearch.qwen_embedder import DIMENSIONS, EMBED_MODEL, QwenEmbedder

        response = types.SimpleNamespace(
            status_code=200,
            output=types.SimpleNamespace(
                embeddings=[types.SimpleNamespace(embedding=[0.1] * DIMENSIONS)]
            ),
            code="",
            message="",
        )
        multimodal_cls = MagicMock()
        multimodal_cls.call.return_value = response
        _install_fake_dashscope(monkeypatch, multimodal_cls=multimodal_cls)

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            embedder = QwenEmbedder()
            result = embedder.embed_query("红色汽车")

        assert result == [0.1] * DIMENSIONS
        multimodal_cls.call.assert_called_once_with(
            api_key="test-key",
            model=EMBED_MODEL,
            input=[{"text": "红色汽车"}],
            dimension=DIMENSIONS,
        )

    def test_embed_video_chunk_uploads_then_embeds(self, monkeypatch, tiny_video):
        from sentrysearch.qwen_embedder import DIMENSIONS, EMBED_MODEL, QwenEmbedder

        response = types.SimpleNamespace(
            status_code=200,
            output=types.SimpleNamespace(
                embeddings=[types.SimpleNamespace(embedding=[0.2] * DIMENSIONS)]
            ),
            code="",
            message="",
        )
        multimodal_cls = MagicMock()
        multimodal_cls.call.return_value = response
        _install_fake_dashscope(monkeypatch, multimodal_cls=multimodal_cls)

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True), \
             patch("sentrysearch.qwen_embedder.upload_video_for_model", return_value="oss://dashscope/tmp/clip.mp4") as mock_upload:
            embedder = QwenEmbedder()
            result = embedder.embed_video_chunk(tiny_video)

        assert result == [0.2] * DIMENSIONS
        mock_upload.assert_called_once_with(tiny_video, EMBED_MODEL, api_key="test-key")
        multimodal_cls.call.assert_called_once_with(
            api_key="test-key",
            model=EMBED_MODEL,
            input=[{"video": "oss://dashscope/tmp/clip.mp4"}],
            dimension=DIMENSIONS,
            fps=1.0,
        )

    def test_embed_video_chunk_raises_for_large_file(self, monkeypatch, tiny_video):
        from sentrysearch.qwen_embedder import MAX_FILE_BYTES, QwenEmbedder, QwenStorageUploadError

        _install_fake_dashscope(monkeypatch, multimodal_cls=MagicMock())
        stat_result = os.stat_result((0, 0, 0, 0, 0, 0, MAX_FILE_BYTES + 1, 0, 0, 0))

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True), \
             patch("pathlib.Path.stat", return_value=stat_result):
            embedder = QwenEmbedder()
            with pytest.raises(QwenStorageUploadError, match="too large"):
                embedder.embed_video_chunk(tiny_video)


class TestQwenRetry:
    @patch("sentrysearch.qwen_embedder.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        from sentrysearch.qwen_embedder import _retry

        exc = Exception("rate limited")
        exc.status_code = 429
        fn = MagicMock(side_effect=[exc, "ok"])
        assert _retry(fn, max_retries=2, initial_delay=0.01) == "ok"
        assert fn.call_count == 2

    @patch("sentrysearch.qwen_embedder.time.sleep")
    def test_raises_quota_error_after_retries(self, mock_sleep):
        from sentrysearch.qwen_embedder import QwenQuotaError, _retry

        exc = Exception("quota exceeded")
        exc.status_code = 429
        fn = MagicMock(side_effect=exc)
        with pytest.raises(QwenQuotaError):
            _retry(fn, max_retries=1, initial_delay=0.01)
