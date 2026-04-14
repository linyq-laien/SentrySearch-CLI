"""Video chunking logic."""

import functools
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov")
MIN_SHOT_SEGMENT_DURATION_SECONDS = 0.5
STILL_FRAME_RETAINED_RATIO_THRESHOLD = 0.1


def _parse_last_ffmpeg_frame_count(stderr_text: str) -> int | None:
    """Return the last reported ``frame=`` count from ffmpeg stderr."""
    matches = re.findall(r"frame=\s*(\d+)", stderr_text)
    if not matches:
        return None
    return int(matches[-1])


def is_supported_video_file(path: str) -> bool:
    """Return True when *path* has a supported video extension."""
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def _ffmpeg_runs(path: str) -> bool:
    """Return True if *path* can write a trivial output file.

    A simple ``-version`` check is not enough — snap-sandboxed ffmpeg
    exits 0 for ``-version`` but cannot access the real filesystem.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            probe_path = tmp.name
        try:
            result = subprocess.run(
                [path, "-y", "-f", "lavfi", "-i", "nullsrc=s=2x2:d=0.1",
                 "-frames:v", "1", probe_path],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0 and os.path.getsize(probe_path) > 0
        finally:
            os.unlink(probe_path)
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _get_ffmpeg_executable() -> str:
    """Return a usable ffmpeg executable path.

    Search order:
    1. System ffmpeg on PATH (only if it actually runs — e.g. snap
       sandboxed binaries are skipped)
    2. imageio-ffmpeg bundled binary (if installed)
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg and _ffmpeg_runs(system_ffmpeg):
        return system_ffmpeg

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "ffmpeg not found on PATH and imageio-ffmpeg is not available. "
            "Install ffmpeg system-wide or `uv add imageio-ffmpeg`."
        ) from exc


def _parse_duration_from_ffmpeg_output(stderr_text: str) -> float:
    """Extract duration from ffmpeg stderr line: Duration: HH:MM:SS.xx"""
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr_text)
    if not match:
        # Surface ffmpeg's own error (e.g. "No such file or directory")
        for line in stderr_text.splitlines():
            lower = line.lower()
            if "no such file" in lower:
                raise FileNotFoundError(f"Video file not found: {line.strip()}")
            if "error" in lower:
                raise RuntimeError(f"ffmpeg error: {line.strip()}")
        raise RuntimeError("Could not determine video duration from ffmpeg output.")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds.

    Prefer ffprobe when available. Fallback to parsing ffmpeg stderr.
    """
    ffprobe_exe = shutil.which("ffprobe")
    if ffprobe_exe:
        result = subprocess.run(
            [
                ffprobe_exe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])

    ffmpeg_exe = _get_ffmpeg_executable()
    result = subprocess.run(
        [ffmpeg_exe, "-i", video_path],
        capture_output=True,
        text=True,
        check=False,
    )
    return _parse_duration_from_ffmpeg_output(result.stderr)


def chunk_video(
    video_path: str,
    chunk_duration: int = 30,
    overlap: int = 5,
) -> list[dict]:
    """Split a video into overlapping chunks using ffmpeg.

    Args:
        video_path: Path to the input video file.
        chunk_duration: Duration of each chunk in seconds.
        overlap: Overlap between consecutive chunks in seconds.

    Returns:
        List of dicts with keys: chunk_path, source_file, start_time, end_time.

    Note:
        The caller is responsible for cleaning up the temporary chunk files
        returned in chunk_path.
    """
    video_path = str(Path(video_path).resolve())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    ffmpeg_exe = _get_ffmpeg_executable()
    duration = _get_video_duration(video_path)
    tmp_dir = tempfile.mkdtemp(prefix="sentrysearch_")
    step = chunk_duration - overlap
    chunks = []

    if duration <= chunk_duration:
        chunk_path = os.path.join(tmp_dir, "chunk_000.mp4")
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-ss", "0",
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",
                chunk_path,
            ],
            capture_output=True,
            check=True,
        )
        return [
            {
                "chunk_path": chunk_path,
                "source_file": video_path,
                "start_time": 0.0,
                "end_time": duration,
            }
        ]

    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk_duration, duration)
        t = end - start
        chunk_path = os.path.join(tmp_dir, f"chunk_{idx:03d}.mp4")

        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(t),
                "-c", "copy",
                chunk_path,
            ],
            capture_output=True,
            check=True,
        )

        chunks.append(
            {
                "chunk_path": chunk_path,
                "source_file": video_path,
                "start_time": start,
                "end_time": end,
            }
        )

        start += step
        idx += 1

        if start + overlap >= duration:
            break

    return chunks


def segment_video_shots(
    video_path: str,
    threshold: float = 0.5,
) -> list[dict]:
    """Split a video into shot-aligned clips using TransNetV2 boundaries."""
    from .shot_detector import detect_shot_scenes
    from .trimmer import trim_clip

    video_path = str(Path(video_path).resolve())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    scenes = detect_shot_scenes(video_path, threshold=threshold)
    if not scenes:
        scenes = [(0.0, _get_video_duration(video_path))]

    tmp_dir = tempfile.mkdtemp(prefix="sentrysearch_shots_")
    segments = []
    for idx, (start_time, end_time) in enumerate(scenes, 1):
        clip_path = os.path.join(tmp_dir, f"shot_{idx:03d}.mp4")
        trim_clip(
            source_file=video_path,
            start_time=start_time,
            end_time=end_time,
            output_path=clip_path,
            padding=0.0,
            prefer_reencode=True,
            require_reencode=True,
        )
        duration = float(end_time) - float(start_time)
        quality_metadata = {
            "segment_quality": "ok",
            "segment_quality_reason": "none",
            "segment_quality_checked": True,
            "segment_duration_seconds": duration,
            "segment_scene_count": 1,
        }
        if duration < MIN_SHOT_SEGMENT_DURATION_SECONDS:
            quality_metadata.update({
                "segment_quality": "low",
                "segment_quality_reason": "too_short",
                "segment_scene_count": 0,
            })
        elif is_still_frame_chunk(clip_path):
            quality_metadata.update({
                "segment_quality": "low",
                "segment_quality_reason": "still_frame",
            })
        else:
            validation_scenes = detect_shot_scenes(clip_path, threshold=threshold)
            quality_metadata["segment_scene_count"] = len(validation_scenes)
            if len(validation_scenes) > 1:
                quality_metadata.update({
                    "segment_quality": "low",
                    "segment_quality_reason": "internal_scene_cut",
                })
        segments.append({
            "chunk_path": clip_path,
            "source_file": video_path,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "segment_index": idx,
            "segmentation": "shot",
            **quality_metadata,
        })

    return segments


def is_still_frame_chunk(
    chunk_path: str,
    threshold: float = STILL_FRAME_RETAINED_RATIO_THRESHOLD,
    verbose: bool = False,
) -> bool:
    """Check if a video chunk contains mostly still frames.

    Runs ffmpeg's ``mpdecimate`` filter and checks how many frames remain
    after near-duplicate frames are removed. A clip that is effectively one
    still image encoded as a short video will retain very few unique frames.

    Args:
        chunk_path: Path to the video chunk.
        threshold: Maximum retained-frame ratio to consider the clip still.
        verbose: Print frame sizes to stderr.

    Returns:
        True if the chunk appears to be a still/static scene.
    """
    try:
        ffmpeg_exe = _get_ffmpeg_executable()

        total_result = subprocess.run(
            [ffmpeg_exe, "-i", chunk_path, "-map", "0:v:0",
             "-c", "copy", "-f", "null", "-"],
            capture_output=True, text=True, check=False,
        )
        stderr = total_result.stderr
        total_frames = _parse_last_ffmpeg_frame_count(stderr)
        if total_frames is not None:
            total_frames = int(total_frames)
        else:
            # Estimate from duration and fps in stream info
            fps_match = re.search(r"(\d+(?:\.\d+)?)\s+fps", stderr)
            dur_match = re.search(
                r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr,
             )
            if not fps_match or not dur_match:
                return False
            fps = float(fps_match.group(1))
            h, m, s = dur_match.groups()
            dur = int(h) * 3600 + int(m) * 60 + float(s)
            total_frames = int(dur * fps)
        if total_frames < 3:
            return False

        decimated_result = subprocess.run(
            [
                ffmpeg_exe,
                "-hide_banner",
                "-i", chunk_path,
                "-vf", "mpdecimate",
                "-an",
                "-f", "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        retained_frames = _parse_last_ffmpeg_frame_count(decimated_result.stderr)
        if retained_frames is None or retained_frames <= 0:
            return False

        retained_ratio = retained_frames / total_frames

        if verbose:
            import sys
            print(
                "    [verbose] still-detect retained frames: "
                f"{retained_frames}/{total_frames} ({retained_ratio:.3f})",
                file=sys.stderr,
            )

        return retained_ratio <= threshold

    except Exception:
        return False


def preprocess_chunk(
    chunk_path: str,
    target_resolution: int = 480,
    target_fps: int = 5,
) -> str:
    """Downscale and reduce frame rate of a video chunk for cheaper embedding.

    Args:
        chunk_path: Path to the input video chunk.
        target_resolution: Target height in pixels (width scales to maintain aspect ratio).
        target_fps: Target frames per second.

    Returns:
        Path to the preprocessed file, or the original chunk_path on failure.
    """
    try:
        ffmpeg_exe = _get_ffmpeg_executable()
        base, ext = os.path.splitext(chunk_path)
        out_path = f"{base}_preprocessed{ext}"

        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i", chunk_path,
                "-vf", f"scale=-2:{target_resolution},fps={target_fps}",
                "-c:v", "libx264",
                "-crf", "28",
                "-c:a", "aac",
                "-b:a", "64k",
                out_path,
            ],
            capture_output=True,
            check=True,
        )
        return out_path
    except Exception:
        return chunk_path


def scan_directory(directory_path: str) -> list[str]:
    """Recursively find supported video files in a directory.

    Args:
        directory_path: Root directory to scan.

    Returns:
        Sorted list of absolute file paths.
    """
    video_files = []
    for root, _dirs, files in os.walk(directory_path):
        for f in files:
            if is_supported_video_file(f):
                video_files.append(os.path.join(root, f))
    video_files.sort()
    return video_files
