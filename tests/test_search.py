"""Tests for sentrysearch.search."""

import math
from unittest.mock import patch

import pytest

from sentrysearch.search import search_footage


def _make_embedding(seed: float, dim: int = 768) -> list[float]:
    vec = [math.sin(seed + i * 0.1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class TestSearchFootage:
    def test_empty_store(self, tmp_store, mock_embed_query):
        results = search_footage("a red car", tmp_store)
        assert results == []

    def test_returns_results(self, tmp_store, mock_embed_query):
        # mock_embed_query returns _fake_embedding(), store a chunk with same vector
        tmp_store.add_chunk("c1", mock_embed_query, {
            "source_file": "video.mp4",
            "start_time": 0.0,
            "end_time": 30.0,
        })
        results = search_footage("anything", tmp_store, n_results=5)
        assert len(results) == 1
        assert results[0]["source_file"] == "video.mp4"
        assert results[0]["similarity_score"] > 0.99

    def test_sorted_by_score(self, tmp_store, mock_embed_query):
        tmp_store.add_chunk("match", mock_embed_query, {
            "source_file": "match.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("diff", _make_embedding(seed=999.0), {
            "source_file": "diff.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        results = search_footage("query", tmp_store, n_results=5)
        assert len(results) == 2
        assert results[0]["source_file"] == "match.mp4"
        assert results[0]["similarity_score"] > results[1]["similarity_score"]

    def test_n_results_limits_output(self, tmp_store, mock_embed_query):
        for i in range(10):
            tmp_store.add_chunk(f"c{i}", _make_embedding(seed=float(i)), {
                "source_file": f"v{i}.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
            })
        results = search_footage("q", tmp_store, n_results=3)
        assert len(results) == 3

    def test_qwen_rerank_fetches_top_10_candidates(self, mock_embed_query):
        class DummyStore:
            def __init__(self):
                self.requested = None

            def search(self, query_embedding, n_results=5):
                self.requested = n_results
                return [
                    {
                        "source_file": f"video{i}.mp4",
                        "start_time": float(i),
                        "end_time": float(i + 1),
                        "score": 0.1 * i,
                    }
                    for i in range(12)
                ]

            def get_backend(self):
                return "qwen"

        store = DummyStore()
        reranked = [{
            "source_file": "video9.mp4",
            "start_time": 9.0,
            "end_time": 10.0,
            "vector_score": 0.9,
            "rerank_score": 0.99,
            "similarity_score": 0.99,
        }]
        with patch("sentrysearch.qwen_reranker.rerank_results", return_value=reranked) as mock_rerank:
            results = search_footage("query", store, n_results=3, rerank=True)

        assert store.requested == 10
        assert results == reranked
        mock_rerank.assert_called_once()

    def test_vector_results_keep_vector_score_without_rerank(self, tmp_store, mock_embed_query):
        tmp_store.add_chunk("c1", mock_embed_query, {
            "source_file": "video.mp4",
            "start_time": 0.0,
            "end_time": 30.0,
        })
        results = search_footage("anything", tmp_store, n_results=1, rerank=False)
        assert results[0]["vector_score"] == pytest.approx(results[0]["similarity_score"])
        assert results[0]["rerank_score"] is None
