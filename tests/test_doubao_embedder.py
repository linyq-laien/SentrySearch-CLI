"""Tests for sentrysearch.doubao_embedder."""

import os
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _install_fake_ark(monkeypatch, ark_cls):
    fake_module = types.SimpleNamespace(Ark=ark_cls)
    monkeypatch.setitem(__import__("sys").modules, "volcenginesdkarkruntime", fake_module)


class TestDoubaoEmbedder:
    def test_raises_without_api_key(self, monkeypatch):
        from sentrysearch.doubao_embedder import DoubaoAPIKeyError, DoubaoEmbedder

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(DoubaoAPIKeyError, match="ARK_API_KEY"):
                DoubaoEmbedder()

    def test_embed_query_calls_multimodal_embeddings(self, monkeypatch):
        from sentrysearch.doubao_embedder import DIMENSIONS, EMBED_MODEL, DoubaoEmbedder

        fake_values = [0.1] * DIMENSIONS
        ark_cls = MagicMock()
        client = MagicMock()
        client.multimodal_embeddings.create.return_value = types.SimpleNamespace(
            data=types.SimpleNamespace(embedding=fake_values)
        )
        ark_cls.return_value = client
        _install_fake_ark(monkeypatch, ark_cls)

        with patch.dict(os.environ, {"ARK_API_KEY": "test-key"}, clear=True):
            embedder = DoubaoEmbedder()
            result = embedder.embed_query("a red car")

        assert result == fake_values
        ark_cls.assert_called_once_with(api_key="test-key")
        client.multimodal_embeddings.create.assert_called_once_with(
            model=EMBED_MODEL,
            input=[{"type": "text", "text": "a red car"}],
            dimensions=DIMENSIONS,
        )

    @patch("sentrysearch.doubao_embedder.time.sleep")
    def test_embed_video_chunk_uploads_embeds_and_deletes(self, mock_sleep, monkeypatch, tiny_video):
        from sentrysearch.doubao_embedder import DIMENSIONS, EMBED_MODEL, DoubaoEmbedder

        fake_values = [0.2] * DIMENSIONS
        ark_cls = MagicMock()
        client = MagicMock()
        client.files.create.return_value = types.SimpleNamespace(id="file-123", status="processing")
        client.files.retrieve.side_effect = [
            types.SimpleNamespace(id="file-123", status="processing", error=None),
            types.SimpleNamespace(id="file-123", status="active", error=None),
        ]
        client.multimodal_embeddings.create.return_value = types.SimpleNamespace(
            data=types.SimpleNamespace(embedding=fake_values)
        )
        ark_cls.return_value = client
        _install_fake_ark(monkeypatch, ark_cls)

        with patch.dict(os.environ, {"ARK_API_KEY": "test-key"}, clear=True):
            embedder = DoubaoEmbedder()
            result = embedder.embed_video_chunk(tiny_video)

        assert result == fake_values
        client.files.create.assert_called_once()
        create_kwargs = client.files.create.call_args.kwargs
        assert create_kwargs["file"] == Path(tiny_video)
        assert create_kwargs["purpose"] == "user_data"
        assert create_kwargs["preprocess_configs"] == {"video": {"fps": 1.0}}
        assert create_kwargs["expire_at"] >= int(time.time()) + 86400
        client.multimodal_embeddings.create.assert_called_once_with(
            model=EMBED_MODEL,
            input=[{"type": "video_url", "video_url": {"url": "ark:/files/file-123", "fps": 1.0}}],
            dimensions=DIMENSIONS,
        )
        client.files.delete.assert_called_once_with("file-123")

    @patch("sentrysearch.doubao_embedder.time.sleep")
    def test_embed_video_chunk_raises_when_file_processing_fails(self, mock_sleep, monkeypatch, tiny_video):
        from sentrysearch.doubao_embedder import DoubaoEmbedder, DoubaoFileProcessingError

        ark_cls = MagicMock()
        client = MagicMock()
        client.files.create.return_value = types.SimpleNamespace(id="file-123", status="processing")
        client.files.retrieve.return_value = types.SimpleNamespace(
            id="file-123",
            status="failed",
            error=types.SimpleNamespace(message="processing failed"),
        )
        ark_cls.return_value = client
        _install_fake_ark(monkeypatch, ark_cls)

        with patch.dict(os.environ, {"ARK_API_KEY": "test-key"}, clear=True):
            embedder = DoubaoEmbedder()
            with pytest.raises(DoubaoFileProcessingError, match="processing failed"):
                embedder.embed_video_chunk(tiny_video)

        client.files.delete.assert_called_once_with("file-123")

    def test_embed_video_chunk_deletes_file_when_embedding_fails(self, monkeypatch, tiny_video):
        from sentrysearch.doubao_embedder import DoubaoEmbedder

        ark_cls = MagicMock()
        client = MagicMock()
        client.files.create.return_value = types.SimpleNamespace(id="file-123", status="active")
        client.multimodal_embeddings.create.side_effect = RuntimeError("boom")
        ark_cls.return_value = client
        _install_fake_ark(monkeypatch, ark_cls)

        with patch.dict(os.environ, {"ARK_API_KEY": "test-key"}, clear=True):
            embedder = DoubaoEmbedder()
            with pytest.raises(RuntimeError, match="boom"):
                embedder.embed_video_chunk(tiny_video)

        client.files.delete.assert_called_once_with("file-123")

    def test_embed_video_chunk_falls_back_to_inline_base64_when_file_url_is_rejected(
        self, monkeypatch, tiny_video
    ):
        from sentrysearch.doubao_embedder import DIMENSIONS, DoubaoEmbedder

        fake_values = [0.3] * DIMENSIONS
        unsupported = Exception(
            "Error code: 400 - {'error': {'code': 'InvalidParameter.UnsupportedInput', "
            "'message': 'Only base64, http or https URLs are supported.'}}"
        )
        unsupported.status_code = 400

        ark_cls = MagicMock()
        client = MagicMock()
        client.files.create.return_value = types.SimpleNamespace(id="file-123", status="active")
        client.multimodal_embeddings.create.side_effect = [
            unsupported,
            types.SimpleNamespace(data=types.SimpleNamespace(embedding=fake_values)),
        ]
        ark_cls.return_value = client
        _install_fake_ark(monkeypatch, ark_cls)

        with patch.dict(os.environ, {"ARK_API_KEY": "test-key"}, clear=True):
            embedder = DoubaoEmbedder()
            result = embedder.embed_video_chunk(tiny_video)

        assert result == fake_values
        assert client.multimodal_embeddings.create.call_count == 2
        first_url = client.multimodal_embeddings.create.call_args_list[0].kwargs["input"][0]["video_url"]["url"]
        second_url = client.multimodal_embeddings.create.call_args_list[1].kwargs["input"][0]["video_url"]["url"]
        assert first_url == "ark:/files/file-123"
        assert second_url.startswith("data:video/mp4;base64,")
        client.files.delete.assert_called_once_with("file-123")


class TestDoubaoRetry:
    @patch("sentrysearch.doubao_embedder.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        from sentrysearch.doubao_embedder import _retry

        exc = Exception("rate limited")
        exc.status_code = 429
        fn = MagicMock(side_effect=[exc, "ok"])
        assert _retry(fn, max_retries=2, initial_delay=0.01) == "ok"
        assert fn.call_count == 2

    @patch("sentrysearch.doubao_embedder.time.sleep")
    def test_raises_quota_error_after_retries(self, mock_sleep):
        from sentrysearch.doubao_embedder import DoubaoQuotaError, _retry

        exc = Exception("resource exhausted")
        exc.status_code = 429
        fn = MagicMock(side_effect=exc)
        with pytest.raises(DoubaoQuotaError):
            _retry(fn, max_retries=1, initial_delay=0.01)

    @patch("sentrysearch.doubao_embedder.time.sleep")
    def test_does_not_retry_on_400_even_if_message_contains_504(self, mock_sleep):
        from sentrysearch.doubao_embedder import _retry

        exc = Exception("Error code: 400 - request id: abc504xyz")
        exc.status_code = 400
        fn = MagicMock(side_effect=exc)
        with pytest.raises(Exception, match="Error code: 400"):
            _retry(fn, max_retries=2, initial_delay=0.01)
        assert fn.call_count == 1
        mock_sleep.assert_not_called()
