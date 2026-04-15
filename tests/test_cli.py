"""Tests for sentrysearch.cli (Click CLI)."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sentrysearch.cli import _fmt_time, _overlay_output_path, cli


@pytest.fixture
def runner():
    return CliRunner()


class TestFmtTime:
    def test_zero(self):
        assert _fmt_time(0) == "00:00"

    def test_minutes(self):
        assert _fmt_time(125) == "02:05"


class TestOverlayOutputPath:
    def test_mov_input_outputs_mp4(self):
        assert _overlay_output_path("/tmp/iphone.mov") == "/tmp/iphone_overlay.mp4"


class TestCliGroup:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Search dashcam footage" in result.output or "search" in result.output.lower()


class TestInitCommand:
    def test_init_doubao_writes_ark_api_key(self, runner, tmp_path):
        env_path = tmp_path / ".env"
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 2048

        with patch("sentrysearch.cli._ENV_PATH", str(env_path)), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder):
            result = runner.invoke(cli, ["init", "--backend", "doubao"], input="test-key\n")

        assert result.exit_code == 0
        assert env_path.read_text() == "ARK_API_KEY=test-key\n"
        mock_embedder.embed_query.assert_called_once_with("test")

    def test_init_qwen_writes_dashscope_api_key(self, runner, tmp_path):
        env_path = tmp_path / ".env"
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024

        with patch("sentrysearch.cli._ENV_PATH", str(env_path)), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder):
            result = runner.invoke(cli, ["init", "--backend", "qwen"], input="test-key\n")

        assert result.exit_code == 0
        assert env_path.read_text() == "DASHSCOPE_API_KEY=test-key\n"
        mock_embedder.embed_query.assert_called_once_with("test")


class TestStatsCommand:
    def test_stats_empty(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=(None, None, None)):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 0, "unique_source_files": 0, "source_files": [],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "empty" in result.output.lower() or "0" in result.output

    def test_stats_with_data(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=("local", "qwen2b", "chunk")):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10,
                "unique_source_files": 2,
                "source_files": ["/a/video1.mp4", "/b/video2.mp4"],
            }
            inst.get_backend.return_value = "local"
            inst.get_segmentation.return_value = "chunk"
            MockStore.return_value = inst
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "10" in result.output
            assert "qwen2b" in result.output

    def test_stats_with_shot_segmentation(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=("gemini", None, "shot")):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 2,
                "unique_source_files": 1,
                "source_files": ["/a/video1.mp4"],
            }
            inst.get_backend.return_value = "gemini"
            inst.get_segmentation.return_value = "shot"
            MockStore.return_value = inst
            result = runner.invoke(cli, ["stats", "--segmentation", "shot"])
            assert result.exit_code == 0
            assert "shot" in result.output.lower()


class TestSearchCommand:
    def test_search_empty_index(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=(None, None, None)):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 0}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "red car"])
            assert result.exit_code == 0
            assert "No indexed footage" in result.output


class TestIndexCommand:
    def test_index_no_supported_videos(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir)])
            assert result.exit_code == 0
            assert "No supported video files found" in result.output

    def test_index_accepts_backend_option(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir), "--backend", "local"])
            assert result.exit_code == 0

    def test_index_accepts_doubao_backend_option(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir), "--backend", "doubao"])
            assert result.exit_code == 0

    def test_index_accepts_qwen_backend_option(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir), "--backend", "qwen"])
            assert result.exit_code == 0

    def test_index_scans_mov_files(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "iphone.MOV"
        source.write_bytes(b"fake")

        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        chunk_path = chunk_dir / "chunk_000.mp4"
        chunk_path.write_bytes(b"chunk")

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = False
        mock_store.get_stats.return_value = {
            "total_chunks": 1,
            "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.chunker.chunk_video", return_value=[{
                 "chunk_path": str(chunk_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 1.0,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0
        mock_store.add_chunks.assert_called_once()

    def test_index_shot_uses_shot_segments(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "test.mp4"
        source.write_bytes(b"fake")

        shot_dir = tmp_path / "shots"
        shot_dir.mkdir()
        shot_path = shot_dir / "shot_001.mp4"
        shot_path.write_bytes(b"shot")

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = False
        mock_store.get_stats.return_value = {
            "total_chunks": 1,
            "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768

        with patch("sentrysearch.store.SentryStore", return_value=mock_store) as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.chunker.segment_video_shots", return_value=[{
                 "chunk_path": str(shot_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 1.0,
                 "segment_index": 1,
                 "segmentation": "shot",
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--segmentation", "shot", "--no-preprocess"])

        assert result.exit_code == 0
        MockStore.assert_called_once_with(backend="gemini", model=None, segmentation="shot")
        mock_store.add_chunks.assert_called_once()

    def test_index_shot_reports_low_quality_segments(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "test.mp4"
        source.write_bytes(b"fake")

        shot_dir = tmp_path / "shots"
        shot_dir.mkdir()
        shot_path = shot_dir / "shot_001.mp4"
        shot_path.write_bytes(b"shot")

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = False
        mock_store.get_stats.return_value = {
            "total_chunks": 1,
            "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.chunker.segment_video_shots", return_value=[{
                 "chunk_path": str(shot_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 0.4,
                 "segment_index": 1,
                 "segmentation": "shot",
                 "segment_quality": "low",
                 "segment_quality_reason": "too_short",
                 "segment_quality_checked": True,
                 "segment_duration_seconds": 0.4,
                 "segment_scene_count": 0,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--segmentation", "shot", "--no-preprocess"])

        assert result.exit_code == 0
        assert "low-quality shot: duration 0.40s is below 0.50s" in result.output
        assert "flagged 1 low-quality shot segments" in result.output

    def test_index_shot_reports_still_frame_segments(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "test.mp4"
        source.write_bytes(b"fake")

        shot_dir = tmp_path / "shots"
        shot_dir.mkdir()
        shot_path = shot_dir / "shot_001.mp4"
        shot_path.write_bytes(b"shot")

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = False
        mock_store.get_stats.return_value = {
            "total_chunks": 1,
            "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.chunker.segment_video_shots", return_value=[{
                 "chunk_path": str(shot_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 1.2,
                 "segment_index": 1,
                 "segmentation": "shot",
                 "segment_quality": "low",
                 "segment_quality_reason": "still_frame",
                 "segment_quality_checked": True,
                 "segment_duration_seconds": 1.2,
                 "segment_scene_count": 1,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--segmentation", "shot", "--no-preprocess"])

        assert result.exit_code == 0
        assert "low-quality shot: segment appears to be a still/static scene" in result.output
        assert "flagged 1 low-quality shot segments" in result.output

    def test_index_shot_skip_low_quality_skips_embedding_and_store(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "test.mp4"
        source.write_bytes(b"fake")

        shot_dir = tmp_path / "shots"
        shot_dir.mkdir()
        shot_path = shot_dir / "shot_001.mp4"
        shot_path.write_bytes(b"shot")

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = False
        mock_store.get_stats.return_value = {
            "total_chunks": 0,
            "unique_source_files": 0,
        }
        mock_embedder = MagicMock()

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.chunker.segment_video_shots", return_value=[{
                 "chunk_path": str(shot_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 0.4,
                 "segment_index": 1,
                 "segmentation": "shot",
                 "segment_quality": "low",
                 "segment_quality_reason": "too_short",
                 "segment_quality_checked": True,
                 "segment_duration_seconds": 0.4,
                 "segment_scene_count": 0,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(
                cli,
                [
                    "index",
                    str(d),
                    "--segmentation",
                    "shot",
                    "--no-preprocess",
                    "--skip-low-quality",
                ],
            )

        assert result.exit_code == 0
        mock_embedder.embed_video_chunk.assert_not_called()
        mock_store.add_chunks.assert_not_called()
        assert "Skipping chunk 1/1 (low quality: too_short)" in result.output
        assert "skipped 1 low-quality shot segments" in result.output

    def test_index_publish_saas_uploads_segments(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "test.mp4"
        source.write_bytes(b"fake")

        shot_dir = tmp_path / "shots"
        shot_dir.mkdir()
        shot_path = shot_dir / "shot_001.mp4"
        shot_path.write_bytes(b"shot-bytes")

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = False
        mock_store.get_stats.return_value = {
            "total_chunks": 1,
            "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768
        mock_saas = MagicMock()
        mock_saas.register_source_video.return_value = {"id": "source-video-id"}
        mock_saas.create_segment_upload_session.return_value = {
            "id": "upload-session-id",
            "callback_token": "callback-token",
            "upload_url": "https://example.invalid/upload",
            "upload_headers": {"Content-Type": "video/mp4"},
        }
        mock_saas.register_segment.return_value = {"id": "segment-id-1"}

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.saas_client.VideoSaaSClient.from_env", return_value=mock_saas), \
             patch("sentrysearch.chunker.segment_video_shots", return_value=[{
                 "chunk_path": str(shot_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 1.0,
                 "segment_index": 1,
                 "segmentation": "shot",
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(
                cli,
                ["index", str(d), "--segmentation", "shot", "--no-preprocess", "--publish-saas"],
            )

        assert result.exit_code == 0
        mock_saas.register_source_video.assert_called_once()
        mock_saas.create_segment_upload_session.assert_called_once()
        mock_saas.upload_segment_file.assert_called_once()
        mock_saas.register_segment.assert_called_once()
        assert mock_saas.register_segment.call_args.kwargs["extra_extension_metadata"] == {}

    def test_index_publish_saas_forwards_segment_quality_metadata(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "test.mp4"
        source.write_bytes(b"fake")

        shot_dir = tmp_path / "shots"
        shot_dir.mkdir()
        shot_path = shot_dir / "shot_001.mp4"
        shot_path.write_bytes(b"shot-bytes")

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = False
        mock_store.get_stats.return_value = {
            "total_chunks": 1,
            "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768
        mock_saas = MagicMock()
        mock_saas.register_source_video.return_value = {"id": "source-video-id"}
        mock_saas.create_segment_upload_session.return_value = {
            "id": "upload-session-id",
            "callback_token": "callback-token",
            "upload_url": "https://example.invalid/upload",
            "upload_headers": {"Content-Type": "video/mp4"},
        }
        mock_saas.register_segment.return_value = {"id": "segment-id-1"}

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.saas_client.VideoSaaSClient.from_env", return_value=mock_saas), \
             patch("sentrysearch.chunker.segment_video_shots", return_value=[{
                 "chunk_path": str(shot_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 2.0,
                 "segment_index": 1,
                 "segmentation": "shot",
                 "segment_quality": "low",
                 "segment_quality_reason": "internal_scene_cut",
                 "segment_quality_checked": True,
                 "segment_duration_seconds": 2.0,
                 "segment_scene_count": 2,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(
                cli,
                ["index", str(d), "--segmentation", "shot", "--no-preprocess", "--publish-saas"],
            )

        assert result.exit_code == 0
        assert mock_saas.register_segment.call_args.kwargs["extra_extension_metadata"] == {
            "segment_quality": "low",
            "segment_quality_reason": "internal_scene_cut",
            "segment_quality_checked": True,
            "segment_duration_seconds": 2.0,
            "segment_scene_count": 2,
        }

    def test_index_publish_saas_skip_low_quality_does_not_upload_segments(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "test.mp4"
        source.write_bytes(b"fake")

        shot_dir = tmp_path / "shots"
        shot_dir.mkdir()
        shot_path = shot_dir / "shot_001.mp4"
        shot_path.write_bytes(b"shot-bytes")

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = False
        mock_store.get_stats.return_value = {
            "total_chunks": 0,
            "unique_source_files": 0,
        }
        mock_embedder = MagicMock()
        mock_saas = MagicMock()
        mock_saas.register_source_video.return_value = {"id": "source-video-id"}

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.saas_client.VideoSaaSClient.from_env", return_value=mock_saas), \
             patch("sentrysearch.chunker.segment_video_shots", return_value=[{
                 "chunk_path": str(shot_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 2.0,
                 "segment_index": 1,
                 "segmentation": "shot",
                 "segment_quality": "low",
                 "segment_quality_reason": "internal_scene_cut",
                 "segment_quality_checked": True,
                 "segment_duration_seconds": 2.0,
                 "segment_scene_count": 2,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(
                cli,
                [
                    "index",
                    str(d),
                    "--segmentation",
                    "shot",
                    "--no-preprocess",
                    "--publish-saas",
                    "--skip-low-quality",
                ],
            )

        assert result.exit_code == 0
        mock_embedder.embed_video_chunk.assert_not_called()
        mock_store.add_chunks.assert_not_called()
        mock_saas.register_source_video.assert_not_called()
        mock_saas.create_segment_upload_session.assert_not_called()
        mock_saas.upload_segment_file.assert_not_called()
        mock_saas.register_segment.assert_not_called()
        assert "Skipping chunk 1/1 (low quality: internal_scene_cut)" in result.output

    def test_index_publish_saas_reprocesses_locally_indexed_files(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "test.mp4"
        source.write_bytes(b"fake")

        shot_dir = tmp_path / "shots"
        shot_dir.mkdir()
        shot_path = shot_dir / "shot_001.mp4"
        shot_path.write_bytes(b"shot-bytes")

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = True
        mock_store.get_stats.return_value = {
            "total_chunks": 1,
            "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768
        mock_saas = MagicMock()
        mock_saas.register_source_video.return_value = {"id": "source-video-id"}
        mock_saas.create_segment_upload_session.return_value = {
            "id": "upload-session-id",
            "callback_token": "callback-token",
            "upload_url": "https://example.invalid/upload",
            "upload_headers": {"Content-Type": "video/mp4"},
        }
        mock_saas.register_segment.return_value = {"id": "segment-id-1"}

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.saas_client.VideoSaaSClient.from_env", return_value=mock_saas), \
             patch("sentrysearch.chunker.segment_video_shots", return_value=[{
                 "chunk_path": str(shot_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 1.0,
                 "segment_index": 1,
                 "segmentation": "shot",
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(
                cli,
                ["index", str(d), "--segmentation", "shot", "--no-preprocess", "--publish-saas"],
            )

        assert result.exit_code == 0
        mock_saas.register_segment.assert_called_once()
        assert "publishing to video-saas" in result.output

    def test_index_publish_saas_binds_segments_to_collection(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "test.mp4"
        source.write_bytes(b"fake")

        shot_dir = tmp_path / "shots"
        shot_dir.mkdir()
        shot_path = shot_dir / "shot_001.mp4"
        shot_path.write_bytes(b"shot-bytes")

        mock_store = MagicMock()
        mock_store.is_indexed.return_value = False
        mock_store.get_stats.return_value = {
            "total_chunks": 1,
            "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768
        mock_saas = MagicMock()
        mock_saas.register_source_video.return_value = {"id": "source-video-id"}
        mock_saas.create_segment_upload_session.return_value = {
            "id": "upload-session-id",
            "callback_token": "callback-token",
            "upload_url": "https://example.invalid/upload",
            "upload_headers": {"Content-Type": "video/mp4"},
        }
        mock_saas.register_segment.return_value = {"id": "segment-id-1"}

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.saas_client.VideoSaaSClient.from_env", return_value=mock_saas), \
             patch("sentrysearch.chunker.segment_video_shots", return_value=[{
                 "chunk_path": str(shot_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 1.0,
                 "segment_index": 1,
                 "segmentation": "shot",
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(
                cli,
                [
                    "index",
                    str(d),
                    "--segmentation",
                    "shot",
                    "--no-preprocess",
                    "--publish-saas",
                    "--publish-collection",
                    "collection-id",
                ],
            )

        assert result.exit_code == 0
        mock_saas.add_segments_to_container.assert_called_once_with(
            container_id="collection-id",
            segment_ids=["segment-id-1"],
        )
        assert "collection-id" in result.output

    def test_index_publish_collection_requires_publish_saas(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        result = runner.invoke(
            cli,
            ["index", str(d), "--publish-collection", "collection-id"],
        )

        assert result.exit_code != 0
        assert "--publish-collection requires --publish-saas" in result.output

    def test_index_shot_surfaces_missing_dependency(self, runner, tmp_path):
        from sentrysearch.shot_detector import ShotDetectionUnavailableError

        d = tmp_path / "vids"
        d.mkdir()
        (d / "test.mp4").write_bytes(b"fake")

        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch(
                 "sentrysearch.chunker.segment_video_shots",
                 side_effect=ShotDetectionUnavailableError("install sentrysearch[shots]"),
             ):
            inst = MagicMock()
            inst.is_indexed.return_value = False
            MockStore.return_value = inst
            result = runner.invoke(cli, ["index", str(d), "--segmentation", "shot"])

        assert result.exit_code == 1
        assert "install sentrysearch[shots]" in result.output


class TestIndexLocalFlags:
    def test_index_passes_model_to_embedder(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local", "--model", "qwen2b",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_once()
            assert mock_get.call_args[1]["model"] == "qwen2b"

    def test_index_passes_quantize_to_embedder(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local", "--quantize",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_once()
            assert mock_get.call_args[1]["quantize"] is True

    def test_index_passes_no_quantize_to_embedder(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local", "--no-quantize",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_once()
            assert mock_get.call_args[1]["quantize"] is False

    def test_index_auto_detects_model(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.local_embedder.detect_default_model", return_value="qwen2b"):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local",
            ])
            assert result.exit_code == 0
            assert mock_get.call_args[1]["model"] == "qwen2b"

    def test_index_model_implies_local_backend(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--model", "qwen2b",
            ])
            assert result.exit_code == 0
            # Should have inferred backend="local" from --model
            mock_get.assert_called_once_with("local", model="qwen2b", quantize=None)

    def test_index_rejects_model_for_doubao_backend(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(cli, [
            "index", str(empty_dir), "--backend", "doubao", "--model", "qwen2b",
        ])
        assert result.exit_code == 1
        assert "local-only" in result.output

    def test_index_rejects_model_for_qwen_backend(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(cli, [
            "index", str(empty_dir), "--backend", "qwen", "--model", "qwen2b",
        ])
        assert result.exit_code == 1
        assert "local-only" in result.output

    def test_index_passes_backend_and_model_to_store(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "test.mp4").write_bytes(b"fake")
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.local_embedder.detect_default_model", return_value="qwen8b"):
            mock_inst = MagicMock()
            mock_inst.is_indexed.return_value = True
            MockStore.return_value = mock_inst
            runner.invoke(cli, ["index", str(d), "--backend", "local"])
            MockStore.assert_called_once_with(backend="local", model="qwen8b", segmentation="chunk")


class TestSearchLocalFlags:
    def test_search_passes_model_to_embedder(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "test query", "--backend", "local", "--model", "qwen2b",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_with("local", model="qwen2b", quantize=None)

    def test_search_passes_quantize_to_embedder(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.store.detect_index_details", return_value=("local", "qwen8b", "chunk")), \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "test query", "--backend", "local", "--quantize",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_with("local", model="qwen8b", quantize=True)

    def test_search_model_implies_local_backend(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "test query", "--model", "qwen2b",
            ])
            assert result.exit_code == 0
            # --model qwen2b should imply --backend local
            mock_get.assert_called_with("local", model="qwen2b", quantize=None)

    def test_search_auto_detects_backend_and_model(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.store.detect_index_details", return_value=("local", "qwen2b", "chunk")), \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            # No --backend or --model flags
            result = runner.invoke(cli, ["search", "test query"])
            assert result.exit_code == 0
            mock_get.assert_called_with("local", model="qwen2b", quantize=None)
            MockStore.assert_called_once_with(backend="local", model="qwen2b", segmentation="chunk")

    def test_search_auto_detects_shot_segmentation(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.store.detect_index_details", return_value=("local", "qwen2b", "shot")), \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test query"])
            assert result.exit_code == 0
            mock_get.assert_called_with("local", model="qwen2b", quantize=None)
            MockStore.assert_called_once_with(backend="local", model="qwen2b", segmentation="shot")

    def test_search_accepts_doubao_backend(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test query", "--backend", "doubao"])
            assert result.exit_code == 0
            mock_get.assert_called_with("doubao", model=None, quantize=None)

    def test_search_accepts_qwen_backend_without_default_rerank(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.search.search_footage", return_value=[]) as mock_search:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test query", "--backend", "qwen"])
            assert result.exit_code == 0
            mock_get.assert_called_with("qwen", model=None, quantize=None)
            assert mock_search.call_args.kwargs["rerank"] is False

    def test_search_rejects_model_for_doubao_backend(self, runner):
        result = runner.invoke(cli, [
            "search", "test query", "--backend", "doubao", "--model", "qwen2b",
        ])
        assert result.exit_code == 1
        assert "local-only" in result.output

    def test_search_rejects_model_for_qwen_backend(self, runner):
        result = runner.invoke(cli, [
            "search", "test query", "--backend", "qwen", "--model", "qwen2b",
        ])
        assert result.exit_code == 1
        assert "local-only" in result.output

    def test_search_qwen_no_rerank_passes_rerank_false(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.search.search_footage", return_value=[]) as mock_search:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test query", "--backend", "qwen", "--no-rerank"])
            assert result.exit_code == 0
            assert mock_search.call_args.kwargs["rerank"] is False

    def test_search_qwen_rerank_flag_reranks_and_saves_top_results_in_reranked_order(self, runner):
        rerank_response = MagicMock()
        rerank_response.status_code = 200
        rerank_response.json.return_value = {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.75},
                ]
            }
        }
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=False), \
             patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.search.embed_query", return_value=[0.1] * 1024), \
             patch("sentrysearch.qwen_reranker.trim_clip", side_effect=lambda source_file, start_time, end_time, output_path, padding=0.0: output_path), \
             patch("sentrysearch.qwen_reranker.upload_video_for_model", side_effect=[
                 "oss://dashscope/tmp/clip1.mp4",
                 "oss://dashscope/tmp/clip2.mp4",
             ]), \
             patch("sentrysearch.qwen_reranker.requests.post", return_value=rerank_response), \
             patch("sentrysearch.cli._open_file"), \
             patch("sentrysearch.trimmer.trim_top_results", return_value=["/clip1.mp4", "/clip2.mp4"]) as mock_trim:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            inst.get_backend.return_value = "qwen"
            inst.search.return_value = [
                {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0, "score": 0.6},
                {"source_file": "/b.mp4", "start_time": 30.0, "end_time": 60.0, "score": 0.7},
            ]
            MockStore.return_value = inst

            result = runner.invoke(
                cli,
                ["search", "test", "--backend", "qwen", "--rerank", "--save-top", "2", "--verbose"],
            )

        assert result.exit_code == 0
        inst.search.assert_called_once()
        passed_results = mock_trim.call_args.args[0]
        assert [item["source_file"] for item in passed_results[:2]] == ["/a.mp4", "/b.mp4"]
        assert mock_trim.call_args.kwargs["count"] == 2

    def test_search_wrong_model_shows_suggestion(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=("local", "qwen2b", "chunk")):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 0}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "red car", "--model", "qwen8b",
            ])
            assert result.exit_code == 0
            assert "qwen2b" in result.output
            assert "qwen8b" in result.output

    def test_search_save_top_calls_trim_top_results(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index_details", return_value=("gemini", None, "chunk")), \
             patch("sentrysearch.search.search_footage", return_value=[
                 {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0, "similarity_score": 0.9},
                 {"source_file": "/a.mp4", "start_time": 30.0, "end_time": 60.0, "similarity_score": 0.8},
                 {"source_file": "/a.mp4", "start_time": 60.0, "end_time": 90.0, "similarity_score": 0.7},
             ]), \
             patch("sentrysearch.trimmer.trim_top_results", return_value=["/clip1.mp4", "/clip2.mp4", "/clip3.mp4"]) as mock_trim:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test", "--save-top", "3", "--no-trim"])
            assert result.exit_code == 0
            mock_trim.assert_called_once()
            assert mock_trim.call_args[1]["count"] == 3

    def test_search_save_top_rejects_zero(self, runner):
        result = runner.invoke(cli, ["search", "test", "--save-top", "0"])
        assert result.exit_code != 0


class TestShotsCommand:
    def test_shots_prints_detected_ranges(self, runner, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        with patch(
            "sentrysearch.shot_detector.detect_shot_scenes",
            return_value=[(0.0, 1.5), (1.5, 3.0)],
        ):
            result = runner.invoke(cli, ["shots", str(video)])

        assert result.exit_code == 0
        assert "#1" in result.output
        assert "00:00-00:01" in result.output
        assert "00:01-00:03" in result.output

    def test_shots_writes_split_clips(self, runner, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        output_dir = tmp_path / "shots"

        with patch(
            "sentrysearch.shot_detector.detect_shot_scenes",
            return_value=[(0.0, 1.5), (1.5, 3.0)],
        ), patch(
            "sentrysearch.trimmer.trim_clip",
            side_effect=lambda source_file, start_time, end_time, output_path, padding=0.0, prefer_reencode=False, require_reencode=False: output_path,
        ) as mock_trim:
            result = runner.invoke(
                cli,
                ["shots", str(video), "--output-dir", str(output_dir), "--split"],
            )

        assert result.exit_code == 0
        assert mock_trim.call_count == 2
        for call in mock_trim.call_args_list:
            assert call.kwargs["prefer_reencode"] is True
            assert call.kwargs["require_reencode"] is True
        assert str(output_dir) in result.output

    def test_shots_surfaces_missing_dependency(self, runner, tmp_path):
        from sentrysearch.shot_detector import ShotDetectionUnavailableError

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        with patch(
            "sentrysearch.shot_detector.detect_shot_scenes",
            side_effect=ShotDetectionUnavailableError("install sentrysearch[shots]"),
        ):
            result = runner.invoke(cli, ["shots", str(video)])

        assert result.exit_code == 1
        assert "install sentrysearch[shots]" in result.output


class TestLabelCommand:
    def test_label_passes_defaults_to_labeler(self, runner, tmp_path):
        video = tmp_path / "shot.mp4"
        video.write_bytes(b"fake")

        with patch(
            "sentrysearch.labeler.label_videos",
            return_value={
                "processed": 1,
                "skipped": 0,
                "items": [{
                    "video": str(video),
                    "label_path": str(tmp_path / "shot.label.json"),
                    "status": "labeled",
                }],
            },
        ) as mock_label:
            result = runner.invoke(cli, ["label", str(video)])

        assert result.exit_code == 0
        mock_label.assert_called_once_with(
            str(video),
            output_dir=None,
            model="gemini-3.1-flash-lite-preview",
            overwrite=False,
            verbose=False,
        )
        assert "Completed labeling: 1 written, 0 skipped." in result.output

    def test_label_accepts_overrides(self, runner, tmp_path):
        shots_dir = tmp_path / "shots"
        shots_dir.mkdir()
        output_dir = tmp_path / "labels"

        with patch(
            "sentrysearch.labeler.label_videos",
            return_value={"processed": 0, "skipped": 0, "items": []},
        ) as mock_label:
            result = runner.invoke(cli, [
                "label", str(shots_dir),
                "--output-dir", str(output_dir),
                "--model", "gemini-custom",
                "--overwrite",
                "--verbose",
            ])

        assert result.exit_code == 0
        mock_label.assert_called_once_with(
            str(shots_dir),
            output_dir=str(output_dir),
            model="gemini-custom",
            overwrite=True,
            verbose=True,
        )
        assert "No supported video files found" in result.output


class TestHandleError:
    def test_local_model_error(self, runner):
        from sentrysearch.local_embedder import LocalModelError

        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=("local", "qwen8b", "chunk")):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            with patch(
                "sentrysearch.embedder.get_embedder",
                side_effect=LocalModelError("no torch"),
            ):
                result = runner.invoke(cli, ["search", "test query", "--backend", "local"])
                assert result.exit_code == 1
                assert "no torch" in result.output

    def test_backend_mismatch_error(self, runner):
        from sentrysearch.store import BackendMismatchError

        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=("local", "qwen8b", "chunk")):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            with patch(
                "sentrysearch.embedder.get_embedder",
                side_effect=BackendMismatchError("built with gemini"),
            ):
                result = runner.invoke(cli, ["search", "test", "--backend", "local"])
                assert result.exit_code == 1
                assert "gemini" in result.output


class TestYtDlpCommand:
    def test_yt_dlp_invokes_module_for_url(self, runner):
        with patch("sentrysearch.cli.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[sys.executable, "-m", "yt_dlp", "https://example.com/video"],
                returncode=0,
            )

            result = runner.invoke(cli, ["yt-dlp", "https://example.com/video"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            [sys.executable, "-m", "yt_dlp", "https://example.com/video"],
            check=False,
        )

    def test_yt_dlp_forwards_unknown_flags(self, runner):
        args = [
            "yt-dlp",
            "-f",
            "bv*+ba/b",
            "-o",
            "%(title)s.%(ext)s",
            "--write-info-json",
            "--flat-playlist",
            "https://example.com/playlist",
        ]

        with patch("sentrysearch.cli.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[sys.executable, "-m", "yt_dlp", *args[1:]],
                returncode=0,
            )

            result = runner.invoke(cli, args)

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            [sys.executable, "-m", "yt_dlp", *args[1:]],
            check=False,
        )

    def test_yt_dlp_help_is_forwarded(self, runner):
        with patch("sentrysearch.cli.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[sys.executable, "-m", "yt_dlp", "--help"],
                returncode=0,
            )

            result = runner.invoke(cli, ["yt-dlp", "--help"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            [sys.executable, "-m", "yt_dlp", "--help"],
            check=False,
        )

    def test_yt_dlp_without_args_still_invokes_module(self, runner):
        with patch("sentrysearch.cli.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[sys.executable, "-m", "yt_dlp"],
                returncode=0,
            )

            result = runner.invoke(cli, ["yt-dlp"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            [sys.executable, "-m", "yt_dlp"],
            check=False,
        )

    def test_yt_dlp_propagates_non_zero_exit_code(self, runner):
        with patch("sentrysearch.cli.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[sys.executable, "-m", "yt_dlp", "https://example.com/fail"],
                returncode=3,
            )

            result = runner.invoke(cli, ["yt-dlp", "https://example.com/fail"])

        assert result.exit_code == 3

    def test_yt_dlp_launch_failure_shows_clear_error(self, runner):
        with patch(
            "sentrysearch.cli.subprocess.run",
            side_effect=OSError("exec format error"),
        ):
            result = runner.invoke(cli, ["yt-dlp", "https://example.com/video"])

        assert result.exit_code == 1
        assert "yt-dlp" in result.output
        assert "exec format error" in result.output


class TestResetCommand:
    def test_reset_empty_index(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=("gemini", None, "chunk")):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 0, "unique_source_files": 0, "source_files": [],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["reset", "--yes"])
            assert result.exit_code == 0
            assert "already empty" in result.output.lower()

    def test_reset_removes_all(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=("gemini", None, "chunk")):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10, "unique_source_files": 2,
                "source_files": ["/a/v1.mp4", "/b/v2.mp4"],
            }
            inst.remove_file.return_value = 5
            MockStore.return_value = inst
            result = runner.invoke(cli, ["reset", "--yes"])
            assert result.exit_code == 0
            assert "10" in result.output
            assert inst.remove_file.call_count == 2

    def test_reset_targets_shot_segmentation(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=(None, None, None)):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 2, "unique_source_files": 1,
                "source_files": ["/a/v1.mp4"],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["reset", "--segmentation", "shot", "--yes"])
            assert result.exit_code == 0
            MockStore.assert_called_once_with(backend="gemini", model=None, segmentation="shot")


class TestRemoveCommand:
    def test_remove_matching_file(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=("gemini", None, "chunk")):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10, "unique_source_files": 2,
                "source_files": ["/a/video1.mp4", "/b/video2.mp4"],
            }
            inst.remove_file.return_value = 5
            MockStore.return_value = inst
            result = runner.invoke(cli, ["remove", "video1"])
            assert result.exit_code == 0
            assert "Removed 5 chunks" in result.output
            inst.remove_file.assert_called_once_with("/a/video1.mp4")

    def test_remove_no_match(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=("gemini", None, "chunk")):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10, "unique_source_files": 1,
                "source_files": ["/a/video1.mp4"],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["remove", "nonexistent"])
            assert result.exit_code == 0
            assert "No indexed files matching" in result.output

    def test_remove_empty_index(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=("gemini", None, "chunk")):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 0, "unique_source_files": 0, "source_files": [],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["remove", "anything"])
            assert result.exit_code == 0
            assert "empty" in result.output.lower()

    def test_remove_targets_shot_segmentation(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index_details", return_value=(None, None, None)):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 1, "unique_source_files": 1,
                "source_files": ["/a/video1.mp4"],
            }
            inst.remove_file.return_value = 1
            MockStore.return_value = inst
            result = runner.invoke(cli, ["remove", "video1", "--segmentation", "shot"])
            assert result.exit_code == 0
            MockStore.assert_called_once_with(backend="gemini", model=None, segmentation="shot")
