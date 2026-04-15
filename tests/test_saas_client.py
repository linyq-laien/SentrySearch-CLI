"""Tests for sentrysearch.saas_client."""

from pathlib import Path

from sentrysearch.saas_client import VideoSaaSClient, format_duration_seconds


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self.text = ""
        self._payload = payload

    def json(self):
        return self._payload


class _RecordingSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return _FakeResponse({"id": "segment-id"})


class TestFormatDurationSeconds:
    def test_zero_pads_to_three_decimals(self):
        assert format_duration_seconds(1200) == "1.200"

    def test_preserves_millisecond_precision(self):
        assert format_duration_seconds(1234) == "1.234"


class TestVideoSaaSClient:
    def test_register_segment_sends_duration_seconds_with_three_decimals(
        self, tmp_path
    ):
        session = _RecordingSession()
        client = VideoSaaSClient(
            base_url="https://saas.example",
            integration_key="key",
            integration_secret="secret",
            session=session,
        )
        clip_path = tmp_path / "clip.mp4"
        clip_path.write_bytes(b"clip-bytes")

        payload = client.register_segment(
            upload_session_id="upload-session-id",
            callback_token="callback-token",
            source_video_id="source-video-id",
            external_segment_id="segment-001",
            title="Clip",
            summary="Segment summary",
            file_path=str(clip_path),
            start_time=0.0,
            end_time=1.2344,
            embedding=[0.1, 0.2, 0.3],
            backend="gemini",
            model=None,
            segmentation="shot",
            segment_index=1,
        )

        assert payload == {"id": "segment-id"}
        body = session.calls[0]["json"]
        assert body["duration_ms"] == 1234
        assert body["duration_seconds"] == "1.234"
        assert Path(body["extension_metadata"]["source_file"]).name == "clip.mp4"
