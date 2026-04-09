"""Optional TransNetV2-based shot detection."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np


class ShotDetectionError(RuntimeError):
    """Raised when shot detection fails at runtime."""


class ShotDetectionUnavailableError(ShotDetectionError):
    """Raised when optional shot-detection dependencies are not installed."""


def _to_numpy(values):
    """Return a NumPy array from NumPy-like or tensor-like values."""
    if isinstance(values, np.ndarray):
        return values
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return values.numpy()
    return np.asarray(values)


def _load_transnetv2():
    """Return a TransNetV2 class from whichever package name is installed."""
    try:
        from transnetv2_pytorch import TransNetV2  # type: ignore

        return TransNetV2
    except ModuleNotFoundError:
        try:
            from transnetv2 import TransNetV2  # type: ignore

            return TransNetV2
        except ModuleNotFoundError as second_exc:
            raise ShotDetectionUnavailableError(
                "Shot detection requires the transnetv2-pytorch package. "
                "Install it with `uv tool install .` or "
                "`pip install sentrysearch`."
            ) from second_exc
        except Exception as exc:  # pragma: no cover - environment specific
            raise ShotDetectionError(f"Failed to load TransNetV2: {exc}") from exc
    except Exception as exc:  # pragma: no cover - environment specific
        raise ShotDetectionError(f"Failed to load TransNetV2: {exc}") from exc


def _get_video_fps(video_path: str) -> float:
    """Return the video stream frame rate."""
    from .chunker import _get_ffmpeg_executable

    ffmpeg_exe = _get_ffmpeg_executable()
    # ffprobe lives alongside ffmpeg; try it first, then fall back to ffmpeg
    ffprobe_exe = shutil.which("ffprobe")
    if not ffprobe_exe:
        # Derive ffprobe path from the ffmpeg we found
        import os

        candidate = ffmpeg_exe.replace("ffmpeg", "ffprobe")
        if os.path.isfile(candidate):
            ffprobe_exe = candidate
    if not ffprobe_exe:
        # Fall back to using ffmpeg directly to get stream info
        result = subprocess.run(
            [ffmpeg_exe, "-i", video_path],
            capture_output=True,
            text=True,
        )
        match = __import__("re").search(
            r"(\d+(?:\.\d+)?)\s*fps|(\d+)/(\d+)\s*tbr",
            result.stderr,
        )
        if match:
            if match.group(1):
                return float(match.group(1))
            if match.group(2) and match.group(3):
                return float(match.group(2)) / float(match.group(3))
        raise ShotDetectionError(
            "Could not determine video frame rate for shot detection. "
            "Install ffmpeg system-wide for best compatibility."
        )

    result = subprocess.run(
        [
            ffprobe_exe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-select_streams",
            "v:0",
            "-show_streams",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    streams = info.get("streams") or []
    if not streams:
        raise ShotDetectionError("No video stream found for shot detection.")
    rate = streams[0].get("avg_frame_rate") or streams[0].get("r_frame_rate")
    if not rate or rate in {"0/0", "0"}:
        raise ShotDetectionError("Could not determine video frame rate for shot detection.")
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den)
    return float(rate)


def _load_video_frames(video_path: str) -> np.ndarray:
    """Extract RGB frames resized to the model's 48x27 input."""
    from .chunker import _get_ffmpeg_executable

    ffmpeg_exe = _get_ffmpeg_executable()
    result = subprocess.run(
        [
            ffmpeg_exe,
            "-i",
            video_path,
            "-vf",
            "scale=48:27",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )

    raw = result.stdout
    frame_size = 27 * 48 * 3
    if len(raw) % frame_size != 0:
        raise ShotDetectionError("TransNetV2 frame extraction produced an unexpected buffer size.")
    return np.frombuffer(raw, np.uint8).reshape([-1, 27, 48, 3])


def _predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert frame-level cut probabilities into inclusive scene ranges."""
    binary = (predictions > threshold).astype(np.uint8)

    scenes = []
    t = -1
    t_prev = 0
    start = 0
    for i, t in enumerate(binary):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    if not scenes:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def detect_shot_scenes(
    video_path: str,
    threshold: float = 0.5,
    *,
    verbose: bool = False,
) -> list[tuple[float, float]]:
    """Return shot-aligned ``(start_time, end_time)`` ranges in seconds."""
    from .chunker import _get_video_duration

    TransNetV2 = _load_transnetv2()
    duration = _get_video_duration(video_path)
    fps = _get_video_fps(video_path)

    try:
        model = TransNetV2()
    except Exception as exc:  # pragma: no cover - environment specific
        raise ShotDetectionError(f"Failed to initialize TransNetV2: {exc}") from exc

    if hasattr(model, "predict_video"):
        result = model.predict_video(video_path)
        if len(result) >= 3:
            _video_frames, single_frame_predictions, _all_frame_predictions = result[:3]
        else:  # pragma: no cover - defensive
            raise ShotDetectionError("Unexpected TransNetV2 predict_video return value.")
    elif hasattr(model, "predict_frames"):
        frames = _load_video_frames(video_path)
        single_frame_predictions, _all_frame_predictions = model.predict_frames(frames)
    else:  # pragma: no cover - defensive
        raise ShotDetectionError("Installed TransNetV2 package does not expose a supported inference API.")

    single_frame_predictions = _to_numpy(single_frame_predictions)

    if verbose:
        import sys

        print(
            f"[verbose] detected {len(single_frame_predictions)} frame predictions "
            f"at threshold={threshold}",
            file=sys.stderr,
        )

    if hasattr(model, "predictions_to_scenes"):
        scenes = model.predictions_to_scenes(single_frame_predictions, threshold=threshold)
    else:
        scenes = _predictions_to_scenes(single_frame_predictions, threshold=threshold)

    ranges: list[tuple[float, float]] = []
    for start_frame, end_frame in scenes:
        start_time = max(0.0, float(start_frame) / fps)
        end_time = min(duration, float(end_frame + 1) / fps)
        if end_time > start_time:
            ranges.append((start_time, end_time))

    if not ranges:
        return [(0.0, duration)]
    if ranges[0][0] > 0.0:
        ranges[0] = (0.0, ranges[0][1])
    if ranges[-1][1] < duration:
        ranges[-1] = (ranges[-1][0], duration)
    return ranges
