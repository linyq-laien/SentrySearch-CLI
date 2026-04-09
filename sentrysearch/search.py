"""Query and retrieval logic."""

from .embedder import embed_query
from .store import SentryStore


def search_footage(
    query: str,
    store: SentryStore,
    n_results: int = 5,
    rerank: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Search indexed footage with a natural language query.

    Args:
        query: Natural language search string.
        store: SentryStore instance to search against.
        n_results: Maximum number of results to return.
        rerank: Whether to apply backend-specific reranking.
        verbose: If True, print debug info to stderr.

    Returns:
        List of result dicts sorted by relevance (best first).
        Each dict contains: source_file, start_time, end_time, similarity_score.
    """
    query_embedding = embed_query(query, verbose=verbose)
    backend = store.get_backend() if hasattr(store, "get_backend") else None
    recall_count = max(n_results, 10) if rerank and backend == "qwen" else n_results
    hits = store.search(query_embedding, n_results=recall_count)

    results = []
    for hit in hits:
        results.append({
            "source_file": hit["source_file"],
            "start_time": hit["start_time"],
            "end_time": hit["end_time"],
            "vector_score": hit["score"],
            "rerank_score": None,
            "similarity_score": hit["score"],
        })

    results.sort(key=lambda r: r["similarity_score"], reverse=True)
    if rerank and backend == "qwen" and results:
        from .qwen_reranker import rerank_results

        results = rerank_results(query, results, verbose=verbose)

    return results[:n_results]
