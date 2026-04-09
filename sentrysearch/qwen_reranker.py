"""DashScope-backed Qwen3-VL reranking over recalled video candidates."""

from __future__ import annotations

import os
import shutil
import tempfile

from dotenv import load_dotenv
import requests

from .qwen_embedder import QwenAPIKeyError
from .qwen_storage import upload_video_for_model
from .trimmer import trim_clip

load_dotenv()

QWEN_RERANK_MODEL = "qwen3-vl-rerank"
# DashScope currently accepts at most 4 video documents per rerank request.
MAX_VIDEO_DOCUMENTS = 4
RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class QwenRerankError(RuntimeError):
    """Raised when Qwen reranking fails."""


def _response_field(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _vector_score(candidate: dict) -> float:
    value = candidate.get("vector_score")
    if value is None:
        value = candidate.get("similarity_score", candidate.get("score"))
    return float(value)


def rerank_results(query_text: str, candidates: list[dict], verbose: bool = False) -> list[dict]:
    """Return *candidates* reordered by Qwen3-VL reranker score."""
    if not candidates:
        return []

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise QwenAPIKeyError(
            "DASHSCOPE_API_KEY is not set.\n\n"
            "Run: sentrysearch init --backend qwen\n\n"
            "Or set it manually:\n"
            "  export DASHSCOPE_API_KEY=your-key"
        )

    rerank_candidates = candidates[:MAX_VIDEO_DOCUMENTS]
    remaining_candidates = candidates[MAX_VIDEO_DOCUMENTS:]
    temp_dir = tempfile.mkdtemp(prefix="sentrysearch_qwen_rerank_")
    documents = []
    clip_paths = []
    try:
        for idx, candidate in enumerate(rerank_candidates):
            clip_path = os.path.join(temp_dir, f"candidate_{idx:03d}.mp4")
            trim_clip(
                source_file=candidate["source_file"],
                start_time=candidate["start_time"],
                end_time=candidate["end_time"],
                output_path=clip_path,
                padding=0.0,
            )
            clip_paths.append(clip_path)
            documents.append({
                "video": upload_video_for_model(clip_path, QWEN_RERANK_MODEL, api_key=api_key)
            })

        response = requests.post(
            RERANK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-OssResourceResolve": "enable",
            },
            json={
                "model": QWEN_RERANK_MODEL,
                "input": {
                    "query": {"text": query_text},
                    "documents": documents,
                },
                "parameters": {
                    "top_n": len(documents),
                    "return_documents": True,
                    "fps": 1.0,
                },
            },
            timeout=300,
        )
    finally:
        for clip_path in clip_paths:
            if os.path.exists(clip_path):
                os.unlink(clip_path)
        shutil.rmtree(temp_dir, ignore_errors=True)

    status_code = _response_field(response, "status_code")
    if status_code and status_code != 200:
        payload = response.json() if hasattr(response, "json") else {}
        code = _response_field(payload, "code") or "unknown_error"
        message = _response_field(payload, "message") or getattr(response, "text", None) or "DashScope rerank request failed."
        raise QwenRerankError(f"{code}: {message}")

    payload = response.json() if hasattr(response, "json") else {}
    output = _response_field(payload, "output")
    result_items = _response_field(output, "results") if output is not None else None
    if not result_items:
        raise QwenRerankError("DashScope rerank response did not contain results.")

    reranked = []
    for item in result_items:
        candidate = dict(rerank_candidates[_response_field(item, "index")])
        vector_score = _vector_score(candidate)
        rerank_score = float(_response_field(item, "relevance_score"))
        candidate["vector_score"] = vector_score
        candidate["rerank_score"] = rerank_score
        candidate["similarity_score"] = rerank_score
        reranked.append(candidate)

    for candidate in remaining_candidates:
        preserved = dict(candidate)
        preserved["vector_score"] = _vector_score(preserved)
        preserved["rerank_score"] = None
        preserved["similarity_score"] = preserved["vector_score"]
        reranked.append(preserved)

    return reranked
