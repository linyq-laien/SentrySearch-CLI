"""Helpers for uploading temporary files to DashScope OSS storage."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import requests

UPLOAD_POLICY_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"


class QwenStorageUploadError(RuntimeError):
    """Raised when DashScope temporary storage upload fails."""


def upload_video_for_model(
    file_path: str,
    model_name: str,
    *,
    api_key: str | None = None,
) -> str:
    """Upload *file_path* to DashScope temporary storage for *model_name*."""
    file_obj = Path(file_path)
    if not file_obj.exists():
        raise FileNotFoundError(file_path)

    resolved_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not resolved_key:
        raise QwenStorageUploadError("DASHSCOPE_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }
    policy_response = requests.get(
        UPLOAD_POLICY_URL,
        headers=headers,
        params={"action": "getPolicy", "model": model_name},
        timeout=30,
    )
    if policy_response.status_code != 200:
        raise QwenStorageUploadError(
            f"Failed to get upload policy for {model_name}: {policy_response.text}"
        )

    payload = policy_response.json().get("data") or {}
    upload_host = payload.get("upload_host")
    upload_dir = payload.get("upload_dir")
    if not upload_host or not upload_dir:
        raise QwenStorageUploadError("Upload policy response did not include upload host/dir.")

    object_key = f"{upload_dir}/{uuid.uuid4().hex}_{file_obj.name}"
    with file_obj.open("rb") as handle:
        files = {
            "OSSAccessKeyId": (None, payload.get("oss_access_key_id", "")),
            "Signature": (None, payload.get("signature", "")),
            "policy": (None, payload.get("policy", "")),
            "x-oss-object-acl": (None, payload.get("x_oss_object_acl", "")),
            "x-oss-forbid-overwrite": (None, payload.get("x_oss_forbid_overwrite", "")),
            "key": (None, object_key),
            "success_action_status": (None, "200"),
            "file": (file_obj.name, handle),
        }
        upload_response = requests.post(upload_host, files=files, timeout=300)

    if upload_response.status_code != 200:
        raise QwenStorageUploadError(
            f"Failed to upload file to DashScope temporary storage: {upload_response.text}"
        )
    return f"oss://{object_key}"
