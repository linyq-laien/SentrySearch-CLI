"""Helpers for publishing indexed segments into the video-saas backend."""

from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from .gemini_embedder import EMBED_MODEL as GEMINI_EMBED_MODEL

load_dotenv()

DEFAULT_TIMEOUT_SECONDS = 60


class VideoSaaSConfigError(RuntimeError):
    """Raised when the CLI is asked to publish but SaaS config is missing."""


class VideoSaaSRequestError(RuntimeError):
    """Raised when the video-saas API returns an error."""


def _stable_id(*parts: object) -> str:
    raw = "::".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_external_video_id(source_file: str) -> str:
    resolved = str(Path(source_file).resolve())
    return f"src_{_stable_id(resolved)}"


def build_external_segment_id(
    *,
    source_file: str,
    start_time: float,
    end_time: float,
    segmentation: str,
    backend: str,
    model: str | None,
) -> str:
    resolved = str(Path(source_file).resolve())
    model_part = model or backend
    return (
        "seg_"
        f"{_stable_id(resolved, f'{start_time:.3f}', f'{end_time:.3f}', segmentation, backend, model_part)}"
    )


def guess_content_type(path: str) -> str:
    guessed, _encoding = mimetypes.guess_type(path)
    return guessed or "video/mp4"


def resolve_embedding_model_name(backend: str, model: str | None) -> str:
    if backend == "gemini":
        return GEMINI_EMBED_MODEL
    if backend == "doubao":
        from .doubao_embedder import EMBED_MODEL as DOUBAO_EMBED_MODEL

        return DOUBAO_EMBED_MODEL
    if backend == "qwen":
        from .qwen_embedder import EMBED_MODEL as QWEN_EMBED_MODEL

        return QWEN_EMBED_MODEL
    return model or "local"


class VideoSaaSClient:
    """Thin API client around the video-saas ingestion endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        integration_key: str,
        integration_secret: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._headers = {
            "X-Integration-Key": integration_key,
            "X-Integration-Secret": integration_secret,
        }

    @classmethod
    def from_env(cls) -> "VideoSaaSClient":
        base_url = os.environ.get("VIDEO_SAAS_BASE_URL")
        integration_key = os.environ.get("VIDEO_SAAS_INTEGRATION_KEY")
        integration_secret = os.environ.get("VIDEO_SAAS_INTEGRATION_SECRET")
        timeout = int(
            os.environ.get("VIDEO_SAAS_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        )
        missing = [
            name
            for name, value in (
                ("VIDEO_SAAS_BASE_URL", base_url),
                ("VIDEO_SAAS_INTEGRATION_KEY", integration_key),
                ("VIDEO_SAAS_INTEGRATION_SECRET", integration_secret),
            )
            if not value
        ]
        if missing:
            missing_names = ", ".join(missing)
            raise VideoSaaSConfigError(
                "Missing video-saas configuration: "
                f"{missing_names}. Set them in ~/.sentrysearch/.env or the shell environment."
            )
        return cls(
            base_url=base_url,
            integration_key=integration_key,
            integration_secret=integration_secret,
            timeout=timeout,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        response = self._session.request(
            method,
            url,
            headers=self._headers,
            json=json_body,
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            detail = response.text.strip() or f"HTTP {response.status_code}"
            raise VideoSaaSRequestError(f"{method} {path} failed: {detail}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise VideoSaaSRequestError(f"{method} {path} returned a non-JSON object")
        return payload

    def register_source_video(
        self,
        *,
        source_file: str,
        duration_ms: int | None,
        backend: str,
        model: str | None,
        segmentation: str,
    ) -> dict[str, Any]:
        path = Path(source_file)
        body = {
            "external_video_id": build_external_video_id(source_file),
            "title": path.stem,
            "summary": f"Source video indexed by SentrySearch ({segmentation}).",
            "original_filename": path.name,
            "duration_ms": duration_ms,
            "extension_metadata": {
                "source_file": str(path.resolve()),
                "index_backend": backend,
                "embedding_model": resolve_embedding_model_name(backend, model),
                "segmentation": segmentation,
            },
        }
        return self._request(
            "POST",
            "/api/v1/ingestion/source-videos/register",
            json_body=body,
        )

    def add_segments_to_container(
        self,
        *,
        container_id: str,
        segment_ids: list[str],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/containers/{container_id}/segments",
            json_body={"segment_ids": segment_ids},
        )

    def create_segment_upload_session(
        self,
        *,
        source_video_id: str,
        external_segment_id: str,
        original_filename: str,
        content_type: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/ingestion/segments/uploads/sessions",
            json_body={
                "source_video_id": source_video_id,
                "external_segment_id": external_segment_id,
                "original_filename": original_filename,
                "content_type": content_type,
            },
        )

    def upload_segment_file(
        self,
        *,
        file_path: str,
        upload_url: str,
        upload_headers: dict[str, str] | None = None,
    ) -> None:
        headers = upload_headers or {}
        with open(file_path, "rb") as file_obj:
            response = requests.put(
                upload_url,
                headers=headers,
                data=file_obj,
                timeout=max(self._timeout, 300),
            )
        if response.status_code >= 400:
            detail = response.text.strip() or f"HTTP {response.status_code}"
            raise VideoSaaSRequestError(f"PUT upload failed: {detail}")

    def register_segment(
        self,
        *,
        upload_session_id: str,
        callback_token: str,
        source_video_id: str,
        external_segment_id: str,
        title: str,
        summary: str,
        file_path: str,
        start_time: float,
        end_time: float,
        embedding: list[float],
        backend: str,
        model: str | None,
        segmentation: str,
        segment_index: int | None,
        extra_extension_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start_ms = int(round(start_time * 1000))
        end_ms = int(round(end_time * 1000))
        duration_ms = max(0, end_ms - start_ms)
        search_phrases = [
            title,
            f"{Path(file_path).stem}",
            f"{Path(file_path).suffix.lstrip('.') or 'video'} clip",
        ]
        body = {
            "upload_session_id": upload_session_id,
            "callback_token": callback_token,
            "source_video_id": source_video_id,
            "external_segment_id": external_segment_id,
            "title": title,
            "summary": summary,
            "content_type": guess_content_type(file_path),
            "file_size_bytes": os.path.getsize(file_path),
            "checksum": None,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "search_phrases": search_phrases,
            "safety_flags": [],
            "primary_people": [],
            "primary_visual_features": [segmentation],
            "embedding": embedding,
            "embedding_model": resolve_embedding_model_name(backend, model),
            "embedding_version": "sentrysearch-cli-v1",
            "thumbnail_object_key": None,
            "extension_metadata": {
                "source_file": str(Path(file_path).resolve()),
                "index_backend": backend,
                "segmentation": segmentation,
                "segment_index": segment_index,
                **(extra_extension_metadata or {}),
            },
        }
        return self._request(
            "POST",
            "/api/v1/ingestion/segments/register",
            json_body=body,
        )
