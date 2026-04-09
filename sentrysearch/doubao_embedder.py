"""Doubao multimodal embedding backend using Volcengine ARK Runtime."""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from .base_embedder import BaseEmbedder

load_dotenv()

EMBED_MODEL = "doubao-embedding-vision-251215"
DIMENSIONS = 2048
# ARK Files API currently enforces a minimum TTL of 24 hours.
FILE_TTL_SECONDS = 25 * 3600
FILE_POLL_TIMEOUT_SECONDS = 60.0
FILE_POLL_INTERVAL_SECONDS = 1.0
MAX_FILE_BYTES = 512 * 1024 * 1024


class DoubaoAPIKeyError(RuntimeError):
    """Raised when ARK_API_KEY is missing."""


class DoubaoFileUploadError(RuntimeError):
    """Raised when ARK file upload fails."""


class DoubaoFileProcessingError(RuntimeError):
    """Raised when an uploaded ARK file never becomes active."""


class DoubaoQuotaError(RuntimeError):
    """Raised when ARK quota is exceeded."""


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_MESSAGE_PATTERNS = (
    "resource exhausted",
    "rate limit",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout",
)


def _status_code_from_message(message: str) -> int | None:
    """Extract an HTTP-like status code from structured error text."""
    match = re.search(r"(?:error|status)\s*code\s*[:=]\s*(429|500|502|503|504)\b", message)
    if match:
        return int(match.group(1))
    return None


def _is_unsupported_file_reference_error(exc: Exception) -> bool:
    """Return True when ARK rejects file references for video_url."""
    msg = str(exc).lower()
    return (
        "invalidparameter.unsupportedinput" in msg
        and "only base64, http or https urls are supported" in msg
    )


def _retry(fn, *, max_retries: int = 5, initial_delay: float = 2.0, max_delay: float = 60.0):
    """Call *fn* with exponential back-off on transient errors."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status is None:
                status = _status_code_from_message(msg)
            retryable = status in _RETRYABLE_STATUS_CODES
            if not retryable:
                retryable = any(pattern in msg for pattern in _RETRYABLE_MESSAGE_PATTERNS)
            if not retryable or attempt == max_retries:
                if status == 429 or "resource exhausted" in msg or "rate limit" in msg:
                    raise DoubaoQuotaError(
                        "Doubao ARK rate limit exceeded.\n\n"
                        "Wait a minute and retry, or reduce the number of chunks "
                        "with a larger --chunk-duration."
                    ) from exc
                raise
            wait = min(delay, max_delay)
            print(
                f"  Retryable error (attempt {attempt + 1}/{max_retries}), "
                f"waiting {wait:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay *= 2


class DoubaoEmbedder(BaseEmbedder):
    """Doubao multimodal embedding backend via ARK Runtime."""

    def __init__(self):
        api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise DoubaoAPIKeyError(
                "ARK_API_KEY is not set.\n\n"
                "Run: sentrysearch init --backend doubao\n\n"
                "Or set it manually:\n"
                "  export ARK_API_KEY=your-key"
            )

        from volcenginesdkarkruntime import Ark

        base_url = os.environ.get("ARK_BASE_URL")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = Ark(**client_kwargs)

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.multimodal_embeddings.create(
                model=EMBED_MODEL,
                input=[{"type": "text", "text": query_text}],
                dimensions=DIMENSIONS,
            )
        )
        embedding = self._extract_embedding(response)

        if verbose:
            elapsed = time.monotonic() - t0
            print(
                f"  [verbose] query embedding: dims={len(embedding)}, "
                f"api_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        return embedding

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        chunk_file = Path(chunk_path)
        if not chunk_file.exists():
            raise FileNotFoundError(chunk_path)

        size_bytes = chunk_file.stat().st_size
        if size_bytes > MAX_FILE_BYTES:
            raise DoubaoFileUploadError(
                f"Chunk is too large for ARK Files API ({size_bytes / (1024 * 1024):.1f} MB).\n\n"
                "Try a smaller --chunk-duration, --target-resolution, or --target-fps."
            )

        if verbose:
            print(
                f"    [verbose] uploading {size_bytes / 1024:.0f}KB chunk to ARK Files API",
                file=sys.stderr,
            )

        expire_at = int(
            (datetime.now(timezone.utc) + timedelta(seconds=FILE_TTL_SECONDS)).timestamp()
        )
        file_id: str | None = None
        t0 = time.monotonic()

        try:
            uploaded = _retry(
                lambda: self._client.files.create(
                    file=chunk_file,
                    purpose="user_data",
                    expire_at=expire_at,
                    preprocess_configs={"video": {"fps": 1.0}},
                )
            )
            file_id = uploaded.id
            if getattr(uploaded, "status", None) != "active":
                self._wait_for_file_active(file_id)

            try:
                response = _retry(
                    lambda: self._client.multimodal_embeddings.create(
                        model=EMBED_MODEL,
                        input=[
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"ark:/files/{file_id}",
                                    "fps": 1.0,
                                },
                            }
                        ],
                        dimensions=DIMENSIONS,
                    )
                )
            except Exception as exc:
                if not _is_unsupported_file_reference_error(exc):
                    raise
                response = _retry(
                    lambda: self._client.multimodal_embeddings.create(
                        model=EMBED_MODEL,
                        input=[
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": self._inline_video_data_url(chunk_file),
                                    "fps": 1.0,
                                },
                            }
                        ],
                        dimensions=DIMENSIONS,
                    )
                )
            embedding = self._extract_embedding(response)
        except DoubaoFileProcessingError:
            raise
        except Exception as exc:
            if file_id is None:
                raise DoubaoFileUploadError(f"Failed to upload chunk to ARK Files API: {exc}") from exc
            raise
        finally:
            if file_id is not None:
                try:
                    self._client.files.delete(file_id)
                except Exception as exc:  # pragma: no cover - best-effort cleanup
                    print(
                        f"Warning: failed to delete ARK file {file_id}: {exc}",
                        file=sys.stderr,
                    )

        if verbose:
            elapsed = time.monotonic() - t0
            print(
                f"    [verbose] dims={len(embedding)}, api_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        return embedding

    def dimensions(self) -> int:
        return DIMENSIONS

    def _wait_for_file_active(self, file_id: str) -> None:
        deadline = time.monotonic() + FILE_POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            file_obj = _retry(lambda: self._client.files.retrieve(file_id))
            status = getattr(file_obj, "status", None)
            if status == "active":
                return
            if status == "failed":
                error = getattr(file_obj, "error", None)
                message = getattr(error, "message", None) or "ARK file preprocessing failed."
                raise DoubaoFileProcessingError(message)
            time.sleep(FILE_POLL_INTERVAL_SECONDS)
        raise DoubaoFileProcessingError(
            f"Timed out waiting for ARK file {file_id} to become active."
        )

    @staticmethod
    def _extract_embedding(response) -> list[float]:
        data = getattr(response, "data", None)
        if data is None:
            raise RuntimeError("Doubao response did not contain embedding data.")
        embedding = getattr(data, "embedding", None)
        if embedding is None:
            raise RuntimeError("Doubao response did not contain embedding values.")
        return list(embedding)

    @staticmethod
    def _inline_video_data_url(chunk_file: Path) -> str:
        """Return a data URL for inline base64 video submission."""
        mime_type, _encoding = mimetypes.guess_type(chunk_file.name)
        if mime_type is None:
            mime_type = "video/mp4"
        video_b64 = base64.b64encode(chunk_file.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{video_b64}"
