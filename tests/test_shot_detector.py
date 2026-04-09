"""Tests for shot detection helpers."""

import json
import subprocess
from unittest.mock import patch

import numpy as np

from sentrysearch.shot_detector import _get_video_fps, detect_shot_scenes


class TestGetVideoFps:
    def test_reads_avg_frame_rate_from_ffprobe(self):
        payload = {"streams": [{"avg_frame_rate": "30000/1001"}]}

        with patch("sentrysearch.shot_detector.shutil.which", return_value="/usr/bin/ffprobe"), \
             patch(
                 "sentrysearch.shot_detector.subprocess.run",
                 return_value=subprocess.CompletedProcess(
                     args=["ffprobe"],
                     returncode=0,
                     stdout=json.dumps(payload),
                     stderr="",
                 ),
             ):
            fps = _get_video_fps("/tmp/video.mp4")

        assert fps == 30000 / 1001


class _FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class TestDetectShotScenes:
    def test_converts_tensor_predictions_before_scene_conversion(self):
        class FakeModel:
            def predict_video(self, _video_path):
                return None, _FakeTensor([0.0, 0.8, 0.1]), None

            def predictions_to_scenes(self, predictions, threshold=0.5):
                assert isinstance(predictions, np.ndarray)
                assert threshold == 0.5
                return np.array([[0, 2]], dtype=np.int32)

        with patch("sentrysearch.shot_detector._load_transnetv2", return_value=FakeModel), \
             patch("sentrysearch.chunker._get_video_duration", return_value=3.0), \
             patch("sentrysearch.shot_detector._get_video_fps", return_value=1.0):
            scenes = detect_shot_scenes("/tmp/video.mp4")

        assert scenes == [(0.0, 3.0)]
