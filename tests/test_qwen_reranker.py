"""Tests for sentrysearch.qwen_reranker."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestQwenReranker:
    def test_rerank_results_uses_http_with_oss_resolve_header(self):
        from sentrysearch.qwen_reranker import QWEN_RERANK_MODEL, RERANK_URL, rerank_results

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.71},
                ]
            }
        }
        candidates = [
            {"source_file": "/src/a.mp4", "start_time": 0.0, "end_time": 5.0, "score": 0.55},
            {"source_file": "/src/b.mp4", "start_time": 5.0, "end_time": 10.0, "score": 0.65},
        ]

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True), \
             patch("sentrysearch.qwen_reranker.trim_clip", side_effect=lambda source_file, start_time, end_time, output_path, padding=0.0: output_path) as mock_trim, \
             patch("sentrysearch.qwen_reranker.upload_video_for_model", side_effect=[
                 "oss://dashscope/tmp/a.mp4",
                 "oss://dashscope/tmp/b.mp4",
             ]), \
             patch("sentrysearch.qwen_reranker.requests.post", return_value=response) as mock_post:
            results = rerank_results("find red car", candidates)

        assert [r["source_file"] for r in results] == ["/src/b.mp4", "/src/a.mp4"]
        assert results[0]["vector_score"] == pytest.approx(0.65)
        assert results[0]["rerank_score"] == pytest.approx(0.91)
        assert results[0]["similarity_score"] == pytest.approx(0.91)
        assert mock_trim.call_count == 2
        for call in mock_trim.call_args_list:
            assert call.kwargs["padding"] == 0.0
        mock_post.assert_called_once_with(
            RERANK_URL,
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
                "X-DashScope-OssResourceResolve": "enable",
            },
            json={
                "model": QWEN_RERANK_MODEL,
                "input": {
                    "query": {"text": "find red car"},
                    "documents": [
                        {"video": "oss://dashscope/tmp/a.mp4"},
                        {"video": "oss://dashscope/tmp/b.mp4"},
                    ],
                },
                "parameters": {
                    "top_n": 2,
                    "return_documents": True,
                    "fps": 1.0,
                },
            },
            timeout=300,
        )

    def test_rerank_results_cleans_up_temp_clips(self, tmp_path):
        from sentrysearch.qwen_reranker import rerank_results

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "output": {
                "results": [{"index": 0, "relevance_score": 0.8}]
            }
        }
        created = tmp_path / "created.mp4"
        created.write_bytes(b"x")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True), \
             patch("sentrysearch.qwen_reranker.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("sentrysearch.qwen_reranker.trim_clip", return_value=str(created)), \
             patch("sentrysearch.qwen_reranker.upload_video_for_model", return_value="oss://dashscope/tmp/created.mp4"), \
             patch("sentrysearch.qwen_reranker.requests.post", return_value=response):
            rerank_results("find car", [{
                "source_file": "/src/a.mp4",
                "start_time": 0.0,
                "end_time": 5.0,
                "score": 0.4,
            }])

        assert not created.exists()

    def test_rerank_results_limits_video_documents_to_four_and_appends_rest(self):
        from sentrysearch.qwen_reranker import rerank_results

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "output": {
                "results": [
                    {"index": 3, "relevance_score": 0.95},
                    {"index": 1, "relevance_score": 0.90},
                    {"index": 0, "relevance_score": 0.85},
                    {"index": 2, "relevance_score": 0.80},
                ]
            }
        }
        candidates = [
            {"source_file": f"/src/{i}.mp4", "start_time": float(i), "end_time": float(i + 1), "score": 1.0 - i * 0.1}
            for i in range(7)
        ]

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True), \
             patch("sentrysearch.qwen_reranker.trim_clip", side_effect=lambda source_file, start_time, end_time, output_path, padding=0.0: output_path) as mock_trim, \
             patch("sentrysearch.qwen_reranker.upload_video_for_model", side_effect=[
                 f"oss://dashscope/tmp/{i}.mp4" for i in range(4)
             ]), \
             patch("sentrysearch.qwen_reranker.requests.post", return_value=response) as mock_post:
            results = rerank_results("find car", candidates)

        sent_documents = mock_post.call_args.kwargs["json"]["input"]["documents"]
        assert len(sent_documents) == 4
        assert mock_trim.call_count == 4
        assert [item["source_file"] for item in results] == [
            "/src/3.mp4",
            "/src/1.mp4",
            "/src/0.mp4",
            "/src/2.mp4",
            "/src/4.mp4",
            "/src/5.mp4",
            "/src/6.mp4",
        ]
        assert results[4]["rerank_score"] is None
        assert results[4]["vector_score"] == pytest.approx(candidates[4]["score"])

    def test_rerank_results_raises_on_http_error(self):
        from sentrysearch.qwen_reranker import QwenRerankError, rerank_results

        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {
            "code": "InvalidParameter",
            "message": "Video URL must be a valid HTTP/HTTPS link.",
        }
        response.text = "bad request"

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True), \
             patch("sentrysearch.qwen_reranker.trim_clip", side_effect=lambda source_file, start_time, end_time, output_path, padding=0.0: output_path), \
             patch("sentrysearch.qwen_reranker.upload_video_for_model", return_value="oss://dashscope/tmp/a.mp4"), \
             patch("sentrysearch.qwen_reranker.requests.post", return_value=response):
            with pytest.raises(QwenRerankError, match="InvalidParameter"):
                rerank_results("find car", [{
                    "source_file": "/src/a.mp4",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "score": 0.4,
                }])
