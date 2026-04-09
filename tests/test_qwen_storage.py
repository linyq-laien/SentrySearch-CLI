"""Tests for sentrysearch.qwen_storage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestQwenStorage:
    def test_upload_video_for_model_requests_policy_and_uploads_file(self, tmp_path):
        from sentrysearch.qwen_storage import upload_video_for_model

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video-bytes")

        policy_response = MagicMock()
        policy_response.status_code = 200
        policy_response.json.return_value = {
            "data": {
                "upload_host": "https://oss.example.com",
                "upload_dir": "dashscope/tmp",
                "oss_access_key_id": "key-id",
                "signature": "sig",
                "policy": "policy-token",
                "x_oss_object_acl": "private",
                "x_oss_forbid_overwrite": "true",
            }
        }
        upload_response = MagicMock()
        upload_response.status_code = 200

        with patch("sentrysearch.qwen_storage.requests.get", return_value=policy_response) as mock_get, \
             patch("sentrysearch.qwen_storage.requests.post", return_value=upload_response) as mock_post:
            result = upload_video_for_model(str(video), "qwen3-vl-embedding", api_key="test-key")

        assert result.startswith("oss://dashscope/tmp/")
        assert result.endswith("_clip.mp4")
        mock_get.assert_called_once_with(
            "https://dashscope.aliyuncs.com/api/v1/uploads",
            headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
            params={"action": "getPolicy", "model": "qwen3-vl-embedding"},
            timeout=30,
        )
        files = mock_post.call_args.kwargs["files"]
        assert files["OSSAccessKeyId"] == (None, "key-id")
        assert files["Signature"] == (None, "sig")
        assert files["policy"] == (None, "policy-token")
        assert files["x-oss-object-acl"] == (None, "private")
        assert files["x-oss-forbid-overwrite"] == (None, "true")
        assert files["key"][0] is None
        assert files["key"][1].startswith("dashscope/tmp/")
        assert files["key"][1].endswith("_clip.mp4")
        assert files["success_action_status"] == (None, "200")
        assert files["file"][0] == "clip.mp4"

    def test_upload_video_for_model_generates_unique_keys(self, tmp_path):
        from sentrysearch.qwen_storage import upload_video_for_model

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video-bytes")

        policy_response = MagicMock()
        policy_response.status_code = 200
        policy_response.json.return_value = {
            "data": {
                "upload_host": "https://oss.example.com",
                "upload_dir": "dashscope/tmp",
                "oss_access_key_id": "key-id",
                "signature": "sig",
                "policy": "policy-token",
            }
        }
        upload_response = MagicMock(status_code=200)

        with patch("sentrysearch.qwen_storage.requests.get", return_value=policy_response), \
             patch("sentrysearch.qwen_storage.requests.post", return_value=upload_response) as mock_post:
            first = upload_video_for_model(str(video), "qwen3-vl-embedding", api_key="test-key")
            second = upload_video_for_model(str(video), "qwen3-vl-embedding", api_key="test-key")

        assert first != second
        first_key = mock_post.call_args_list[0].kwargs["files"]["key"][1]
        second_key = mock_post.call_args_list[1].kwargs["files"]["key"][1]
        assert first_key != second_key

    def test_upload_video_for_model_raises_on_policy_failure(self, tmp_path):
        from sentrysearch.qwen_storage import QwenStorageUploadError, upload_video_for_model

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video-bytes")
        policy_response = MagicMock(status_code=403, text="forbidden")

        with patch("sentrysearch.qwen_storage.requests.get", return_value=policy_response):
            with pytest.raises(QwenStorageUploadError, match="Failed to get upload policy"):
                upload_video_for_model(str(video), "qwen3-vl-embedding", api_key="test-key")

    def test_upload_video_for_model_raises_when_file_is_missing(self):
        from sentrysearch.qwen_storage import upload_video_for_model

        with pytest.raises(FileNotFoundError):
            upload_video_for_model("/no/such/file.mp4", "qwen3-vl-embedding", api_key="test-key")
