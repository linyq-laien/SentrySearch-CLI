"""DashScope-backed Qwen3-VL embedding backend."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .base_embedder import BaseEmbedder
from .qwen_storage import QwenStorageUploadError, upload_video_for_model

load_dotenv()

EMBED_MODEL = "qwen3-vl-embedding"
DIMENSIONS = 1024
MAX_FILE_BYTES = 512 * 1024 * 1024


class QwenAPIKeyError(RuntimeError):
    """Raised when DASHSCOPE_API_KEY is missing."""


class QwenQuotaError(RuntimeError):
    """Raised when DashScope quota is exceeded."""


def _retry(fn, *, max_retries: int = 5, initial_delay: float = 2.0, max_delay: float = 60.0):
    """Call *fn* with exponential backoff on transient DashScope errors."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            retryable = status in (429, 500, 502, 503, 504)
            if not retryable:
                retryable = any(token in msg for token in ("429", "500", "502", "503", "504", "quota"))
            if not retryable or attempt == max_retries:
                if status == 429 or "quota" in msg:
                    raise QwenQuotaError(
                        "DashScope rate limit exceeded.\n\n"
                        "Wait a minute and retry, or reduce the number of chunks with a larger --chunk-duration."
                    ) from exc
                raise
            wait = min(delay, max_delay)
            print(
                f"  Retryable error (attempt {attempt + 1}/{max_retries}), waiting {wait:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay *= 2


def _response_field(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_embedding(response) -> list[float]:
    status_code = _response_field(response, "status_code")
    if status_code and status_code != 200:
        code = _response_field(response, "code") or "unknown_error"
        message = _response_field(response, "message") or "DashScope request failed."
        raise RuntimeError(f"{code}: {message}")

    output = _response_field(response, "output")
    embeddings = _response_field(output, "embeddings") if output is not None else None
    if not embeddings:
        raise RuntimeError("DashScope response did not contain embeddings.")
    embedding = _response_field(embeddings[0], "embedding")
    if embedding is None:
        raise RuntimeError("DashScope response did not contain embedding values.")
    return list(embedding)


class QwenEmbedder(BaseEmbedder):
    """Qwen3-VL embedding backend via DashScope."""

    def __init__(self):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise QwenAPIKeyError(
                "DASHSCOPE_API_KEY is not set.\n\n"
                "Run: sentrysearch init --backend qwen\n\n"
                "Or set it manually:\n"
                "  export DASHSCOPE_API_KEY=your-key"
            )
        self._api_key = api_key

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        from dashscope import MultiModalEmbedding

        t0 = time.monotonic()
        response = _retry(
            lambda: MultiModalEmbedding.call(
                api_key=self._api_key,
                model=EMBED_MODEL,
                input=[{"text": query_text}],
                dimension=DIMENSIONS,
            )
        )
        embedding = _extract_embedding(response)
        if verbose:
            elapsed = time.monotonic() - t0
            print(
                f"  [verbose] query embedding: dims={len(embedding)}, api_time={elapsed:.2f}s",
                file=sys.stderr,
            )
        return embedding

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        from dashscope import MultiModalEmbedding

        chunk_file = Path(chunk_path)
        if not chunk_file.exists():
            raise FileNotFoundError(chunk_path)
        size_bytes = chunk_file.stat().st_size
        if size_bytes > MAX_FILE_BYTES:
            raise QwenStorageUploadError(
                f"Chunk is too large for DashScope temporary storage ({size_bytes / (1024 * 1024):.1f} MB)."
            )

        if verbose:
            print(
                f"    [verbose] uploading {size_bytes / 1024:.0f}KB chunk to DashScope temporary storage",
                file=sys.stderr,
            )

        upload_url = upload_video_for_model(chunk_path, EMBED_MODEL, api_key=self._api_key)
        t0 = time.monotonic()
        response = _retry(
            lambda: MultiModalEmbedding.call(
                api_key=self._api_key,
                model=EMBED_MODEL,
                input=[{"video": upload_url}],
                dimension=DIMENSIONS,
                fps=1.0,
            )
        )
        embedding = _extract_embedding(response)
        if verbose:
            elapsed = time.monotonic() - t0
            print(
                f"    [verbose] dims={len(embedding)}, api_time={elapsed:.2f}s",
                file=sys.stderr,
            )
        return embedding

    def dimensions(self) -> int:
        return DIMENSIONS
