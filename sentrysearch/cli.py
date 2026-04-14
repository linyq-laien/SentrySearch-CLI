"""Click-based CLI entry point."""

import os
import platform
import shutil
import subprocess
import sys

import click
from dotenv import load_dotenv
from .chunker import SUPPORTED_VIDEO_EXTENSIONS

_ENV_PATH = os.path.join(os.path.expanduser("~"), ".sentrysearch", ".env")

# Load from stable config location first, then cwd as fallback
load_dotenv(_ENV_PATH)
load_dotenv()  # cwd .env can override


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _open_file(path: str) -> None:
    """Open a file with the system's default application."""
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # non-critical — clip is already saved


def _overlay_output_path(path: str) -> str:
    """Return the default overlay output path for a source video."""
    base, _ext = os.path.splitext(path)
    return f"{base}_overlay.mp4"


def _require_publish_collection_only_with_publish_saas(
    publish_saas: bool,
    publish_collection: str | None,
) -> None:
    if publish_collection and not publish_saas:
        raise click.BadOptionUsage(
            "--publish-collection",
            "--publish-collection requires --publish-saas.",
        )


def _segment_quality_warning(chunk: dict) -> str | None:
    """Return a user-facing warning for low-quality shot segments."""
    if chunk.get("segment_quality") != "low":
        return None

    reason = chunk.get("segment_quality_reason", "unknown")
    duration = float(chunk.get("segment_duration_seconds", 0.0))
    if reason == "too_short":
        return f"low-quality shot: duration {duration:.2f}s is below 0.50s"
    if reason == "still_frame":
        return "low-quality shot: segment appears to be a still/static scene"
    if reason == "internal_scene_cut":
        scene_count = int(chunk.get("segment_scene_count", 0))
        return f"low-quality shot: validation detected {scene_count} scenes inside one segment"
    return f"low-quality shot: {reason}"


def _handle_error(e: Exception) -> None:
    """Print a user-friendly error and exit."""
    from .doubao_embedder import (
        DoubaoAPIKeyError,
        DoubaoFileProcessingError,
        DoubaoFileUploadError,
        DoubaoQuotaError,
    )
    from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError
    from .local_embedder import LocalModelError
    from .qwen_embedder import QwenAPIKeyError, QwenQuotaError
    from .qwen_reranker import QwenRerankError
    from .qwen_storage import QwenStorageUploadError
    from .saas_client import VideoSaaSConfigError, VideoSaaSRequestError
    from .shot_detector import ShotDetectionUnavailableError
    from .store import BackendMismatchError

    if isinstance(e, GeminiAPIKeyError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, GeminiQuotaError):
        click.secho("Error: " + str(e), fg="yellow", err=True)
        raise SystemExit(1)
    if isinstance(e, DoubaoAPIKeyError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, DoubaoQuotaError):
        click.secho("Error: " + str(e), fg="yellow", err=True)
        raise SystemExit(1)
    if isinstance(e, (DoubaoFileUploadError, DoubaoFileProcessingError)):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, QwenAPIKeyError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, QwenQuotaError):
        click.secho("Error: " + str(e), fg="yellow", err=True)
        raise SystemExit(1)
    if isinstance(e, (QwenStorageUploadError, QwenRerankError)):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, (VideoSaaSConfigError, VideoSaaSRequestError)):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, LocalModelError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, ShotDetectionUnavailableError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, BackendMismatchError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, PermissionError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, FileNotFoundError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, RuntimeError) and "ffmpeg not found" in str(e).lower():
        click.secho(
            "Error: ffmpeg is not available.\n\n"
            "Install it with one of:\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  macOS:          brew install ffmpeg\n"
            "  pip fallback:   uv add imageio-ffmpeg",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    raise e


def _apply_overlay_to_clip(
    clip_path: str,
    source_file: str,
    start_time: float,
    end_time: float,
    *,
    replace: bool = True,
) -> bool:
    """Apply Tesla telemetry overlay to a clip. Returns True on success.

    When *replace* is True the overlay is written over *clip_path* in-place.
    """
    from .overlay import apply_overlay, get_metadata_samples, reverse_geocode

    samples = get_metadata_samples(source_file, start_time, end_time)
    if samples is None:
        click.secho(
            "No Tesla SEI metadata found — skipping overlay.",
            fg="yellow", err=True,
        )
        return False

    location = None
    mid = samples[len(samples) // 2]
    lat = mid.get("latitude_deg", 0.0)
    lon = mid.get("longitude_deg", 0.0)
    if lat and lon:
        click.echo("Reverse geocoding location...")
        location = reverse_geocode(lat, lon)
        if location is None:
            click.secho(
                "Geocoding failed — continuing without location. "
                "Install deps with: uv tool install \".[tesla]\"",
                fg="yellow", err=True,
            )

    overlay_path = _overlay_output_path(clip_path)
    result_path = apply_overlay(
        clip_path, overlay_path, samples, location,
        source_file=source_file,
        start_time=start_time,
    )
    if result_path == overlay_path and os.path.isfile(overlay_path):
        if replace:
            os.replace(overlay_path, clip_path)
        click.echo("Applied Tesla metadata overlay")
        return True

    click.secho("Overlay failed.", fg="yellow", err=True)
    return False


@click.group()
def cli():
    """Search dashcam footage using natural language queries."""


def _write_env_key(env_path: str, key_name: str, key_value: str) -> None:
    """Write or replace a single key in the stable sentrysearch env file."""
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = f.readlines()
        with open(env_path, "w") as f:
            found = False
            for line in lines:
                if line.startswith(f"{key_name}="):
                    f.write(f"{key_name}={key_value}\n")
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write(f"{key_name}={key_value}\n")
    else:
        with open(env_path, "w") as f:
            f.write(f"{key_name}={key_value}\n")


def _require_local_model_only(backend: str | None, model: str | None) -> None:
    """Reject --model for non-local backends."""
    if backend in {"doubao", "qwen"} and model is not None:
        raise click.ClickException(
            f"--model is local-only and cannot be used with --backend {backend}."
        )


def _resolve_store_model(backend: str, model: str | None) -> str | None:
    """Return the model metadata key to use for store selection."""
    if backend == "doubao":
        from .store import DOUBAO_MODEL

        return DOUBAO_MODEL
    if backend == "qwen":
        from .store import QWEN_MODEL

        return QWEN_MODEL
    return model


def _other_segmentation(segmentation: str) -> str:
    return "shot" if segmentation == "chunk" else "chunk"


# -----------------------------------------------------------------------
# init
# -----------------------------------------------------------------------

@cli.command()
@click.option("--backend", type=click.Choice(["gemini", "doubao", "qwen"]), default="gemini",
              show_default=True, help="Remote embedding backend to configure.")
def init(backend):
    """Set up a remote API key for sentrysearch."""
    env_path = _ENV_PATH
    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    if backend == "gemini":
        key_name = "GEMINI_API_KEY"
        provider_name = "Gemini"
        provider_url = "https://aistudio.google.com/apikey"
        expected_dims = 768
    elif backend == "doubao":
        key_name = "ARK_API_KEY"
        provider_name = "Doubao ARK"
        provider_url = "https://console.volcengine.com/ark"
        expected_dims = 2048
    else:
        key_name = "DASHSCOPE_API_KEY"
        provider_name = "DashScope"
        provider_url = "https://bailian.console.aliyun.com/"
        expected_dims = 1024

    # Check for existing key
    if os.path.exists(env_path):
        with open(env_path) as f:
            contents = f.read()
        if f"{key_name}=" in contents:
            if not click.confirm("API key already configured. Overwrite?", default=False):
                return

    api_key = click.prompt(
        f"Enter your {provider_name} API key\n"
        f"  Get one at {provider_url}\n"
        "  (input is hidden)",
        hide_input=True,
    )

    # Write/update .env
    _write_env_key(env_path, key_name, api_key)

    # Validate by embedding a test string
    os.environ[key_name] = api_key
    click.echo("Validating API key...")
    try:
        from .embedder import get_embedder

        embedder = get_embedder(backend)
        vec = embedder.embed_query("test")
        if len(vec) != expected_dims:
            click.secho(
                f"Unexpected embedding dimension: {len(vec)} (expected {expected_dims}). "
                "The key may be valid but something is off.",
                fg="yellow",
                err=True,
            )
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        click.secho(f"Validation failed: {e}", fg="red", err=True)
        click.secho("Please check your API key and try again.", fg="red", err=True)
        raise SystemExit(1)

    click.secho(
        "Setup complete. You're ready to go — run "
        "`sentrysearch index <directory>` to get started.",
        fg="green",
    )
    if backend == "gemini":
        click.secho(
            "\nTip: Set a spending limit at https://aistudio.google.com/billing "
            "to prevent accidental overspending.",
            fg="yellow",
        )


# -----------------------------------------------------------------------
# yt-dlp
# -----------------------------------------------------------------------

@cli.command(
    name="yt-dlp",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
    add_help_option=False,
)
@click.argument("yt_dlp_args", nargs=-1, type=click.UNPROCESSED)
def yt_dlp(yt_dlp_args):
    """Proxy arguments to the bundled yt-dlp module."""
    command = [sys.executable, "-m", "yt_dlp", *yt_dlp_args]

    try:
        result = subprocess.run(command, check=False)
    except OSError as e:
        click.secho(f"Error: failed to launch bundled yt-dlp: {e}", fg="red", err=True)
        raise SystemExit(1)

    if result.returncode != 0:
        raise SystemExit(result.returncode)


# -----------------------------------------------------------------------
# label
# -----------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("-o", "--output-dir", default=None,
              help="Directory to write label JSON files into (default: next to each video).")
@click.option("--model", default=None, show_default=False,
              help="Gemini model to use for shot labeling "
                   f"(default: gemini-3.1-flash-lite-preview).")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True,
              help="Overwrite existing .label.json files.")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def label(path, output_dir, model, overwrite, verbose):
    """Generate strict JSON labels for a shot clip or a directory of clips."""
    from .labeler import DEFAULT_LABEL_MODEL, label_videos

    try:
        result = label_videos(
            path,
            output_dir=output_dir,
            model=model or DEFAULT_LABEL_MODEL,
            overwrite=overwrite,
            verbose=verbose,
        )
    except Exception as e:
        _handle_error(e)
        return

    if not result["items"]:
        supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS if "SUPPORTED_VIDEO_EXTENSIONS" in globals() else [".mp4", ".mov"])
        click.echo(f"No supported video files found ({supported}).")
        return

    for item in result["items"]:
        action = "Skipped" if item["status"] == "skipped" else "Labeled"
        click.echo(f"{action}: {item['video']} -> {item['label_path']}")

    click.echo(
        f"\nCompleted labeling: {result['processed']} written, "
        f"{result['skipped']} skipped."
    )


# -----------------------------------------------------------------------
# index
# -----------------------------------------------------------------------

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--segmentation", type=click.Choice(["chunk", "shot"]), default="chunk",
              show_default=True, help="Segment videos by fixed windows or detected shots.")
@click.option("--chunk-duration", default=30, show_default=True,
              help="Chunk duration in seconds.")
@click.option("--overlap", default=5, show_default=True,
              help="Overlap between chunks in seconds.")
@click.option("--shot-threshold", default=0.5, show_default=True,
              help="Cut confidence threshold for shot detection (shot segmentation only).")
@click.option("--preprocess/--no-preprocess", default=True, show_default=True,
              help="Downscale and reduce frame rate before embedding.")
@click.option("--target-resolution", default=480, show_default=True,
              help="Target video height in pixels for preprocessing.")
@click.option("--target-fps", default=5, show_default=True,
              help="Target frames per second for preprocessing.")
@click.option("--skip-still/--no-skip-still", default=True, show_default=True,
              help="Skip chunks with no meaningful visual change.")
@click.option("--backend", type=click.Choice(["gemini", "local", "doubao", "qwen"]), default=None,
              help="Embedding backend (default: gemini, or local when --model is set).")
@click.option("--model", default=None, show_default=False,
              help="Model for local backend: qwen8b, qwen2b, or HuggingFace ID "
                   "(default: auto-detect from hardware). Implies --backend local.")
@click.option("--quantize/--no-quantize", default=None,
              help="Enable/disable 4-bit quantization for local backend (default: auto-detect).")
@click.option("--publish-saas/--no-publish-saas", default=False, show_default=True,
              help="After local indexing, upload each segment to video-saas via its ingestion API.")
@click.option("--publish-collection", default=None, show_default=False,
              help="Collection id in video-saas. Uploaded segments will be bound to it.")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def index(directory, segmentation, chunk_duration, overlap, shot_threshold, preprocess, target_resolution,
          target_fps, skip_still, backend, model, quantize, publish_saas, publish_collection, verbose):
    """Index supported video files in DIRECTORY for searching."""
    from .chunker import (
        SUPPORTED_VIDEO_EXTENSIONS,
        chunk_video,
        is_still_frame_chunk,
        preprocess_chunk,
        segment_video_shots,
        scan_directory,
    )
    from .embedder import get_embedder, reset_embedder
    from .local_embedder import detect_default_model, normalize_model_key
    from .saas_client import (
        VideoSaaSClient,
        VideoSaaSRequestError,
        build_external_segment_id,
        guess_content_type,
    )
    from .store import SentryStore

    try:
        _require_publish_collection_only_with_publish_saas(publish_saas, publish_collection)
        # --model implies --backend local
        if model is not None and backend is None:
            backend = "local"
        _require_local_model_only(backend, model)
        if backend is None:
            backend = "gemini"

        # Auto-detect model from hardware when using local backend
        if backend == "local" and model is None:
            model = detect_default_model()
            click.echo(f"Auto-detected model: {model}", err=True)

        # Normalize model key for consistent collection naming
        if backend == "local":
            model = normalize_model_key(model)
        store_model = _resolve_store_model(backend, model)

        embedder = get_embedder(backend, model=model, quantize=quantize)
        saas_client = VideoSaaSClient.from_env() if publish_saas else None
        target_collection_id = publish_collection if saas_client and publish_collection else None
        if target_collection_id is not None:
            click.echo(
                "Publishing to video-saas collection id: "
                f"{target_collection_id}"
            )

        if os.path.isfile(directory):
            videos = [os.path.abspath(directory)]
        else:
            videos = scan_directory(directory)

        if not videos:
            supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
            click.echo(f"No supported video files found ({supported}).")
            return

        store = SentryStore(backend=backend, model=store_model, segmentation=segmentation)
        total_files = len(videos)
        new_files = 0
        new_chunks = 0
        skipped_chunks = 0
        low_quality_chunks = 0

        if verbose:
            click.echo(f"[verbose] DB path: {store._client._identifier}", err=True)
            click.echo(
                f"[verbose] backend={backend}, segmentation={segmentation}, "
                f"chunk_duration={chunk_duration}s, overlap={overlap}s",
                err=True,
            )

        for file_idx, video_path in enumerate(videos, 1):
            abs_path = os.path.abspath(video_path)
            basename = os.path.basename(video_path)

            if store.is_indexed(abs_path) and not publish_saas:
                click.echo(f"Skipping ({file_idx}/{total_files}): {basename} (already indexed)")
                continue
            if store.is_indexed(abs_path) and publish_saas:
                click.echo(
                    f"Reprocessing ({file_idx}/{total_files}): {basename} "
                    "(already indexed locally, publishing to video-saas)",
                )

            if segmentation == "shot":
                chunks = segment_video_shots(abs_path, threshold=shot_threshold)
            else:
                chunks = chunk_video(abs_path, chunk_duration=chunk_duration, overlap=overlap)
            num_chunks = len(chunks)
            embedded = []

            if verbose:
                click.echo(f"  [verbose] {basename}: duration split into {num_chunks} chunks", err=True)

            source_video = None
            published_segment_ids = []
            if saas_client and chunks:
                duration_ms = int(round(max(chunk["end_time"] for chunk in chunks) * 1000))
                source_video = saas_client.register_source_video(
                    source_file=abs_path,
                    duration_ms=duration_ms,
                    backend=backend,
                    model=store_model,
                    segmentation=segmentation,
                )

            # Track files to clean up after processing
            files_to_cleanup = []

            for chunk_idx, chunk in enumerate(chunks, 1):
                quality_warning = _segment_quality_warning(chunk)
                if quality_warning:
                    low_quality_chunks += 1
                    click.echo(f"  Warning: {quality_warning}", err=True)

                if skip_still and is_still_frame_chunk(
                    chunk["chunk_path"], verbose=verbose,
                ):
                    click.echo(
                        f"Skipping chunk {chunk_idx}/{num_chunks} (still frame)"
                    )
                    skipped_chunks += 1
                    # Clean up the skipped chunk file
                    files_to_cleanup.append(chunk["chunk_path"])
                    continue

                click.echo(
                    f"Indexing file {file_idx}/{total_files}: {basename} "
                    f"[chunk {chunk_idx}/{num_chunks}]"
                )

                embed_path = chunk["chunk_path"]
                if preprocess:
                    original_size = os.path.getsize(embed_path)
                    embed_path = preprocess_chunk(
                        embed_path,
                        target_resolution=target_resolution,
                        target_fps=target_fps,
                    )
                    if verbose:
                        new_size = os.path.getsize(embed_path)
                        click.echo(
                            f"    [verbose] preprocess: {original_size / 1024:.0f}KB -> "
                            f"{new_size / 1024:.0f}KB "
                            f"({100 * (1 - new_size / original_size):.0f}% reduction)",
                            err=True,
                        )
                    # Track preprocessed file for cleanup
                    if embed_path != chunk["chunk_path"]:
                        files_to_cleanup.append(embed_path)

                try:
                    embedding = embedder.embed_video_chunk(embed_path, verbose=verbose)
                except RuntimeError as embed_err:
                    click.echo(
                        f"  Warning: skipping chunk {chunk_idx}/{num_chunks}: {embed_err}",
                        err=True,
                    )
                    files_to_cleanup.append(chunk["chunk_path"])
                    skipped_chunks += 1
                    continue

                if embedding is None:
                    click.echo(
                        f"  Warning: skipping chunk {chunk_idx}/{num_chunks} (null embedding)",
                        err=True,
                    )
                    files_to_cleanup.append(chunk["chunk_path"])
                    skipped_chunks += 1
                    continue

                embedded.append({**chunk, "embedding": embedding})

                if saas_client and source_video is not None:
                    external_segment_id = build_external_segment_id(
                        source_file=abs_path,
                        start_time=float(chunk["start_time"]),
                        end_time=float(chunk["end_time"]),
                        segmentation=segmentation,
                        backend=backend,
                        model=store_model,
                    )
                    original_filename = os.path.basename(chunk["chunk_path"])
                    upload_session = saas_client.create_segment_upload_session(
                        source_video_id=source_video["id"],
                        external_segment_id=external_segment_id,
                        original_filename=original_filename,
                        content_type=guess_content_type(chunk["chunk_path"]),
                    )
                    saas_client.upload_segment_file(
                        file_path=chunk["chunk_path"],
                        upload_url=upload_session["upload_url"],
                        upload_headers=upload_session.get("upload_headers"),
                    )
                    segment_index = chunk.get("segment_index")
                    title_suffix = (
                        f"shot {segment_index:03d}"
                        if isinstance(segment_index, int)
                        else f"{float(chunk['start_time']):.1f}s-{float(chunk['end_time']):.1f}s"
                    )
                    segment_title = f"{os.path.splitext(basename)[0]} {title_suffix}"
                    segment_summary = (
                        f"Segment from {basename} covering "
                        f"{float(chunk['start_time']):.1f}s to {float(chunk['end_time']):.1f}s."
                    )
                    segment = saas_client.register_segment(
                        upload_session_id=upload_session["id"],
                        callback_token=upload_session["callback_token"],
                        source_video_id=source_video["id"],
                        external_segment_id=external_segment_id,
                        title=segment_title,
                        summary=segment_summary,
                        file_path=chunk["chunk_path"],
                        start_time=float(chunk["start_time"]),
                        end_time=float(chunk["end_time"]),
                        embedding=embedding,
                        backend=backend,
                        model=store_model,
                        segmentation=segmentation,
                        segment_index=segment_index if isinstance(segment_index, int) else None,
                        extra_extension_metadata={
                            key: value
                            for key, value in {
                                "segment_quality": chunk.get("segment_quality"),
                                "segment_quality_reason": chunk.get("segment_quality_reason"),
                                "segment_quality_checked": chunk.get("segment_quality_checked"),
                                "segment_duration_seconds": chunk.get("segment_duration_seconds"),
                                "segment_scene_count": chunk.get("segment_scene_count"),
                            }.items()
                            if value is not None
                        },
                    )
                    segment_id = segment.get("id")
                    if not isinstance(segment_id, str) or not segment_id:
                        raise VideoSaaSRequestError(
                            "video-saas register segment response did not include a segment id"
                        )
                    published_segment_ids.append(segment_id)

                # Clean up chunk file after embedding
                files_to_cleanup.append(chunk["chunk_path"])

            # Clean up temporary chunk files
            for f in files_to_cleanup:
                try:
                    os.unlink(f)
                except OSError:
                    pass

            # Clean up the temporary directory containing chunks
            if chunks:
                tmp_dir = os.path.dirname(chunks[0]["chunk_path"])
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if embedded:
                store.add_chunks(embedded)
                new_files += 1
                new_chunks += len(embedded)
            if (
                saas_client
                and target_collection_id is not None
                and published_segment_ids
            ):
                saas_client.add_segments_to_container(
                    container_id=target_collection_id,
                    segment_ids=published_segment_ids,
                )
                click.echo(
                    f"Bound {len(published_segment_ids)} uploaded segments from {basename} "
                    f"to collection id {target_collection_id}."
                )

        stats = store.get_stats()
        summary_notes = []
        if skipped_chunks:
            summary_notes.append(f"skipped {skipped_chunks} still")
        if low_quality_chunks:
            summary_notes.append(f"flagged {low_quality_chunks} low-quality shot segments")
        summary_suffix = f" ({', '.join(summary_notes)})" if summary_notes else ""
        click.echo(
            f"\nIndexed {new_chunks} new chunks from {new_files} files{summary_suffix}. "
            f"Total: {stats['total_chunks']} chunks from "
            f"{stats['unique_source_files']} files."
        )

    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


# -----------------------------------------------------------------------
# shots
# -----------------------------------------------------------------------

@cli.command(name="shots")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--threshold", default=0.5, show_default=True, type=click.FloatRange(min=0.0, max=1.0),
              help="Cut confidence threshold for TransNetV2.")
@click.option("--output-dir", default=None, show_default=False,
              help="Directory to write split shot clips into.")
@click.option("--split/--no-split", default=False, show_default=True,
              help="Write each detected shot as its own clip.")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def shots(video_path, threshold, output_dir, split, verbose):
    """Detect shot boundaries in VIDEO_PATH with TransNetV2."""
    from .shot_detector import detect_shot_scenes
    from .trimmer import trim_clip

    try:
        video_path = os.path.abspath(video_path)
        scenes = detect_shot_scenes(video_path, threshold=threshold, verbose=verbose)

        if not scenes:
            click.echo("No shots detected.")
            return

        for idx, (start_time, end_time) in enumerate(scenes, 1):
            click.echo(
                f"#{idx} {_fmt_time(start_time)}-{_fmt_time(end_time)} "
                f"({end_time - start_time:.2f}s)"
            )

        if split:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.expanduser(output_dir or f"./{base_name}_shots")
            os.makedirs(output_dir, exist_ok=True)

            for idx, (start_time, end_time) in enumerate(scenes, 1):
                clip_path = os.path.join(output_dir, f"shot_{idx:03d}.mp4")
                trim_clip(
                    source_file=video_path,
                    start_time=start_time,
                    end_time=end_time,
                    output_path=clip_path,
                    padding=0.0,
                    prefer_reencode=True,
                    require_reencode=True,
                )

            click.echo(f"\nSaved {len(scenes)} shot clips to {output_dir}")

    except Exception as e:
        _handle_error(e)


# -----------------------------------------------------------------------
# search
# -----------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("-n", "--results", "n_results", default=5, show_default=True,
              help="Number of results to return.")
@click.option("-o", "--output-dir", default="~/sentrysearch_clips", show_default=True,
              help="Directory to save trimmed clips.")
@click.option("--trim/--no-trim", default=True, show_default=True,
              help="Auto-trim the top result.")
@click.option("--save-top", default=None, type=click.IntRange(min=1),
              help="Save the top N matching clips instead of just the #1 result (e.g. --save-top 3).")
@click.option("--threshold", default=0.41, show_default=True, type=float,
              help="Minimum similarity score to consider a confident match.")
@click.option("--overlay/--no-overlay", default=False, show_default=True,
              help="Burn Tesla telemetry overlay (speed, GPS, turn signals) onto trimmed clip.")
@click.option("--segmentation", type=click.Choice(["chunk", "shot"]), default=None,
              help="Index segmentation mode to search (auto-detected if omitted).")
@click.option("--backend", type=click.Choice(["gemini", "local", "doubao", "qwen"]), default=None,
              help="Embedding backend (auto-detected from index if omitted).")
@click.option("--model", default=None, show_default=False,
              help="Model for local backend: qwen8b, qwen2b, or HuggingFace ID "
                   "(default: auto-detect from index). Implies --backend local.")
@click.option("--quantize/--no-quantize", default=None,
              help="Enable/disable 4-bit quantization for local backend (default: auto-detect).")
@click.option("--rerank/--no-rerank", default=None,
              help="Apply Qwen VL reranking after vector recall (default: off).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def search(query, n_results, output_dir, trim, save_top, threshold, overlay, segmentation, backend, model, quantize, rerank, verbose):
    """Search indexed footage with a natural language QUERY."""
    from .embedder import get_embedder, reset_embedder
    from .local_embedder import normalize_model_key
    from .search import search_footage
    from .store import SentryStore, detect_index, detect_index_details

    output_dir = os.path.expanduser(output_dir)

    try:
        # --model implies --backend local
        if model is not None and backend is None:
            backend = "local"
        _require_local_model_only(backend, model)

        # Normalize model key for consistent collection naming
        if model is not None:
            model = normalize_model_key(model)

        detected_backend = detected_model = detected_segmentation = None
        if segmentation is None:
            detected_backend, detected_model, detected_segmentation = detect_index_details()
            segmentation = detected_segmentation or "chunk"
        else:
            detected_backend, detected_model = detect_index(segmentation=segmentation)

        # Auto-detect backend and model from whichever collection has data
        if backend is None:
            backend = detected_backend or "gemini"
            if model is None:
                model = detected_model
        elif backend == "local" and model is None and detected_backend == "local":
            model = detected_model
        elif backend in {"doubao", "qwen"}:
            model = _resolve_store_model(backend, model)

        store = SentryStore(backend=backend, model=model, segmentation=segmentation)

        if store.get_stats()["total_chunks"] == 0:
            alt_segmentation = _other_segmentation(segmentation)
            alt_store = SentryStore(backend=backend, model=model, segmentation=alt_segmentation)
            if alt_store.get_stats()["total_chunks"] > 0:
                click.echo(
                    f"No footage indexed with the {segmentation} segmentation. "
                    f"Your {backend} index uses {alt_segmentation}.\n\n"
                    f"Try: sentrysearch search \"{query}\" --segmentation {alt_segmentation}"
                )
                return
            # Check if data exists under a different model
            det_backend, det_model, _ = detect_index_details()
            if det_backend == backend and det_model and det_model != model:
                click.echo(
                    f"No footage indexed with the {model} model. "
                    f"Your index uses {det_model}.\n\n"
                    f"Try: sentrysearch search \"{query}\" --model {det_model}"
                )
            elif det_backend and det_backend != backend:
                click.echo(
                    f"No footage indexed with the {backend} backend. "
                    f"Your index uses {det_backend}."
                )
            else:
                click.echo(
                    "No indexed footage found. "
                    "Run `sentrysearch index <directory>` first."
                )
            return

        embedder_model = model if backend == "local" else None
        get_embedder(backend, model=embedder_model, quantize=quantize)
        if rerank is None:
            rerank = False

        # Ensure we fetch enough results for --save-top
        if save_top is not None and save_top > n_results:
            n_results = save_top

        if verbose:
            click.echo(f"  [verbose] backend={backend}, similarity threshold: {threshold}", err=True)

        results = search_footage(
            query,
            store,
            n_results=n_results,
            rerank=rerank,
            verbose=verbose,
        )

        if not results:
            click.echo(
                "No results found.\n\n"
                "Suggestions:\n"
                "  - Try a broader or different query\n"
                "  - Re-index with smaller --chunk-duration for finer granularity\n"
                "  - Check `sentrysearch stats` to see what's indexed"
            )
            return

        best_score = results[0]["similarity_score"]
        used_rerank = results[0].get("rerank_score") is not None
        low_confidence = (best_score < threshold) if not used_rerank else False

        if low_confidence and not trim:
            click.secho(
                f"(low confidence — best score: {best_score:.2f})",
                fg="yellow",
                err=True,
            )

        for i, r in enumerate(results, 1):
            basename = os.path.basename(r["source_file"])
            start_str = _fmt_time(r["start_time"])
            end_str = _fmt_time(r["end_time"])
            score = r["similarity_score"]
            if verbose:
                if r.get("rerank_score") is not None:
                    click.echo(
                        f"  #{i} [rerank={score:.6f} vector={r['vector_score']:.6f}] "
                        f"{basename} @ {start_str}-{end_str}"
                    )
                else:
                    click.echo(
                        f"  #{i} [{score:.6f}] {basename} "
                        f"@ {start_str}-{end_str}"
                    )
            else:
                click.echo(
                    f"  #{i} [{score:.2f}] {basename} "
                    f"@ {start_str}-{end_str}"
                )

        should_trim = trim or save_top is not None
        if should_trim:
            if low_confidence:
                if not click.confirm(
                    f"No confident match found (best score: {best_score:.2f}). "
                    "Show results anyway?",
                    default=False,
                ):
                    return

            from .trimmer import trim_top_results
            count = save_top if save_top is not None else 1
            clip_paths = trim_top_results(results, output_dir, count=count)

            for i, clip_path in enumerate(clip_paths):
                if overlay:
                    r = results[i]
                    _apply_overlay_to_clip(
                        clip_path, r["source_file"],
                        r["start_time"], r["end_time"],
                    )
                click.echo(f"\nSaved clip: {clip_path}")

            if clip_paths:
                _open_file(clip_paths[0])

    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


# -----------------------------------------------------------------------
# overlay
# -----------------------------------------------------------------------

@cli.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default=None,
              help="Output path (default: <video>_overlay.mp4).")
def overlay(video, output):
    """Apply Tesla telemetry overlay to a VIDEO file for testing."""
    from .chunker import _get_video_duration

    video = os.path.abspath(video)
    if output is None:
        output = _overlay_output_path(video)

    try:
        duration = _get_video_duration(video)
    except Exception as e:
        _handle_error(e)

    success = _apply_overlay_to_clip(
        video, video, 0.0, duration, replace=False,
    )
    if success:
        overlay_path = _overlay_output_path(video)
        if output != overlay_path and os.path.isfile(overlay_path):
            os.replace(overlay_path, output)
        click.secho(f"Saved: {output}", fg="green")
        _open_file(output)
    else:
        raise SystemExit(1)


# -----------------------------------------------------------------------
# stats
# -----------------------------------------------------------------------

@cli.command()
@click.option("--segmentation", type=click.Choice(["chunk", "shot"]), default=None,
              help="Index segmentation mode to inspect (auto-detected if omitted).")
def stats(segmentation):
    """Print index statistics."""
    from .store import SentryStore, detect_index_details

    backend, model, detected_segmentation = detect_index_details(segmentation=segmentation)
    segmentation = segmentation or detected_segmentation or "chunk"
    if backend is None:
        backend = "gemini"
    store = SentryStore(backend=backend, model=model, segmentation=segmentation)
    s = store.get_stats()

    if s["total_chunks"] == 0:
        alt_segmentation = _other_segmentation(segmentation)
        alt_store = SentryStore(backend=backend, model=model, segmentation=alt_segmentation)
        if alt_store.get_stats()["total_chunks"] > 0:
            click.echo(
                f"No indexed footage for the {segmentation} segmentation. "
                f"Try `sentrysearch stats --segmentation {alt_segmentation}`."
            )
            return
        click.echo("Index is empty. Run `sentrysearch index <directory>` first.")
        return

    click.echo(f"Total chunks:  {s['total_chunks']}")
    click.echo(f"Source files:  {s['unique_source_files']}")
    backend_label = store.get_backend()
    if model:
        backend_label += f" ({model})"
    click.echo(f"Backend:       {backend_label}")
    click.echo(f"Segmentation:  {store.get_segmentation()}")
    click.echo("\nIndexed files:")
    for f in s["source_files"]:
        exists = os.path.exists(f)
        label = "" if exists else "  [missing]"
        click.echo(f"  {f}{label}")


# -----------------------------------------------------------------------
# reset
# -----------------------------------------------------------------------

@cli.command()
@click.option("--segmentation", type=click.Choice(["chunk", "shot"]), default=None,
              help="Index segmentation mode to reset (auto-detected if omitted).")
@click.option("--backend", type=click.Choice(["gemini", "local", "doubao", "qwen"]), default=None,
              help="Backend to reset (auto-detected if omitted).")
@click.option("--model", default=None,
              help="Model to reset (auto-detected if omitted). Implies --backend local.")
@click.confirmation_option(prompt="This will delete all indexed data. Continue?")
def reset(segmentation, backend, model):
    """Delete all indexed data."""
    from .store import SentryStore, detect_index_details

    if model is not None and backend is None:
        backend = "local"
    _require_local_model_only(backend, model)
    detected_backend = detected_model = detected_segmentation = None
    if segmentation is None:
        detected_backend, detected_model, detected_segmentation = detect_index_details()
        segmentation = detected_segmentation or "chunk"
    else:
        detected_backend, detected_model, _ = detect_index_details(segmentation=segmentation)
    if backend is None:
        backend = detected_backend or "gemini"
        if model is None:
            model = detected_model
    model = _resolve_store_model(backend, model)

    store = SentryStore(backend=backend, model=model, segmentation=segmentation)
    s = store.get_stats()

    if s["total_chunks"] == 0:
        alt_segmentation = _other_segmentation(segmentation)
        alt_store = SentryStore(backend=backend, model=model, segmentation=alt_segmentation)
        if alt_store.get_stats()["total_chunks"] > 0:
            click.echo(
                f"No indexed footage for the {segmentation} segmentation. "
                f"Try `sentrysearch reset --segmentation {alt_segmentation}`."
            )
            return
        click.echo("Index is already empty.")
        return

    for f in s["source_files"]:
        store.remove_file(f)

    click.echo(f"Removed {s['total_chunks']} chunks from {s['unique_source_files']} files.")


# -----------------------------------------------------------------------
# remove
# -----------------------------------------------------------------------

@cli.command()
@click.argument("files", nargs=-1, required=True)
@click.option("--segmentation", type=click.Choice(["chunk", "shot"]), default=None,
              help="Index segmentation mode to remove from (auto-detected if omitted).")
@click.option("--backend", type=click.Choice(["gemini", "local", "doubao", "qwen"]), default=None,
              help="Backend to remove from (auto-detected if omitted).")
@click.option("--model", default=None,
              help="Model to remove from (auto-detected if omitted). Implies --backend local.")
def remove(files, segmentation, backend, model):
    """Remove specific files from the index.

    Accepts full paths or substrings that match indexed file paths.
    """
    from .store import SentryStore, detect_index_details

    if model is not None and backend is None:
        backend = "local"
    _require_local_model_only(backend, model)
    detected_backend = detected_model = detected_segmentation = None
    if segmentation is None:
        detected_backend, detected_model, detected_segmentation = detect_index_details()
        segmentation = detected_segmentation or "chunk"
    else:
        detected_backend, detected_model, _ = detect_index_details(segmentation=segmentation)
    if backend is None:
        backend = detected_backend or "gemini"
        if model is None:
            model = detected_model
    model = _resolve_store_model(backend, model)

    store = SentryStore(backend=backend, model=model, segmentation=segmentation)
    s = store.get_stats()

    if s["total_chunks"] == 0:
        alt_segmentation = _other_segmentation(segmentation)
        alt_store = SentryStore(backend=backend, model=model, segmentation=alt_segmentation)
        if alt_store.get_stats()["total_chunks"] > 0:
            click.echo(
                f"No indexed footage for the {segmentation} segmentation. "
                f"Try `sentrysearch remove <file> --segmentation {alt_segmentation}`."
            )
            return
        click.echo("Index is empty.")
        return

    total_removed = 0
    for pattern in files:
        # Match against indexed source files (substring match)
        matches = [f for f in s["source_files"] if pattern in f]
        if not matches:
            click.echo(f"No indexed files matching '{pattern}'")
            continue
        for source_file in matches:
            removed = store.remove_file(source_file)
            click.echo(f"Removed {removed} chunks from {source_file}")
            total_removed += removed

    if total_removed:
        click.echo(f"\nTotal: removed {total_removed} chunks.")
