# SentrySearch

Semantic search over video footage. Type what you're looking for, get a trimmed clip back.

[OpenClaw Skill](https://clawhub.ai/ssrajadh/natural-language-video-search)

[<video src="https://github.com/ssrajadh/sentrysearch/raw/main/docs/demo.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/baf98fad-080b-48e1-97f5-a2db2cbd53f5)

## How it works

SentrySearch splits your videos into overlapping chunks (or detected camera shots), embeds each chunk as video using one of several backends — Google's Gemini Embedding API, ByteDance's Doubao ARK, Alibaba's Qwen VL (DashScope), or a local Qwen3-VL model — and stores the vectors in a local ChromaDB database. When you search, your text query is embedded into the same vector space and matched against the stored video embeddings. The top match is automatically trimmed from the original file and saved as a clip.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/) (if you don't have it):

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```


2. Clone and install:

```bash
git clone https://github.com/ssrajadh/sentrysearch.git
cd sentrysearch
uv tool install .
# optional: TransNetV2 shot detection / splitting
uv tool install ".[shots]"
```

3. Set up your API key (or [use a local model instead](#local-backend-no-api-key-needed)):

```bash
sentrysearch init
```

This prompts for your Gemini API key, writes it to `~/.sentrysearch/.env`, and validates it with a test embedding. You can also configure other backends:

```bash
sentrysearch init --backend doubao    # Doubao ARK (Volcengine)
sentrysearch init --backend qwen      # Qwen VL (DashScope / Alibaba Cloud)
```

| Backend | Env variable | Get an API key |
|---|---|---|
| **gemini** (default) | `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **doubao** | `ARK_API_KEY` | [console.volcengine.com/ark](https://console.volcengine.com/ark) |
| **qwen** | `DASHSCOPE_API_KEY` | [bailian.console.aliyun.com](https://bailian.console.aliyun.com/) |

4. Index your footage:

```bash
sentrysearch index /path/to/footage
```

To use a non-default backend:

```bash
sentrysearch index /path/to/footage --backend doubao
sentrysearch index /path/to/footage --backend qwen
```

5. Search:

```bash
sentrysearch search "red truck running a stop sign"
```

Search auto-detects the backend from your index — no extra flags needed after indexing.

ffmpeg is required for video chunking and trimming. If you don't have it system-wide, the bundled `imageio-ffmpeg` is used automatically.

> **Manual setup:** If you prefer not to use `sentrysearch init`, you can copy `.env.example` to `.env` and add your key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) manually.

## Usage

### Init

```bash
$ sentrysearch init
Enter your Gemini API key (get one at https://aistudio.google.com/apikey): ****
Validating API key...
Setup complete. You're ready to go — run `sentrysearch index <directory>` to get started.
```

For other backends, pass `--backend`:

```bash
$ sentrysearch init --backend doubao
Enter your Doubao ARK API key (get one at https://console.volcengine.com/ark): ****
Validating API key...
Setup complete.

$ sentrysearch init --backend qwen
Enter your DashScope API key (get one at https://bailian.console.aliyun.com/): ****
Validating API key...
Setup complete.
```

If a key is already configured, you'll be asked whether to overwrite it.

> **Tip:** Set a spending limit at [aistudio.google.com/billing](https://aistudio.google.com/billing) to prevent accidental overspending on Gemini.

### Index footage

```bash
$ sentrysearch index /path/to/video/footage
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 1/4]
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 2/4]
...
Indexed 12 new chunks from 3 files. Total: 12 chunks from 3 files.
```

**Backend selection:**

- Default: `--backend gemini` (Gemini Embedding API)
- `--backend doubao` — use Doubao ARK multimodal embeddings
- `--backend qwen` — use Qwen VL multimodal embeddings (DashScope)
- `--backend local` — use a local model instead of a remote API ([details below](#local-backend-no-api-key-needed))

**Segmentation mode:**

- Default: fixed time windows (`--segmentation chunk`)
- `--segmentation shot` — index one embedding per detected camera shot instead of fixed windows (requires `.[shots]` extra)

**Example: full Qwen pipeline**

```bash
sentrysearch init --backend qwen
sentrysearch index /path/to/footage --backend qwen
sentrysearch search "red truck running a stop sign"
```

**Example: shot-based indexing**

```bash
sentrysearch index /path/to/footage --segmentation shot
sentrysearch search "car cutting me off"
```

**Example: Qwen backend with reranking**

```bash
sentrysearch index /path/to/footage --backend qwen
sentrysearch search "car running a red light" --rerank
```

Options:

- `--chunk-duration 30` — seconds per chunk
- `--overlap 5` — overlap between chunks
- `--segmentation shot` — index one embedding per detected shot instead of fixed windows
- `--shot-threshold 0.5` — shot detection threshold when `--segmentation shot` is used
- `--no-preprocess` — skip downscaling/frame rate reduction (send raw chunks)
- `--target-resolution 480` — target height in pixels for preprocessing
- `--target-fps 5` — target frame rate for preprocessing
- `--no-skip-still` — embed all chunks, even ones with no visual change

### Shot Detection / Splitting with TransNetV2

Install the optional shot-detection extra:

```bash
uv tool install ".[shots]"
```

Detect shot boundaries:

```bash
sentrysearch shots /path/to/video.mp4
```

Split the video into one clip per detected shot:

```bash
sentrysearch shots /path/to/video.mp4 --split
```

`--split` now prefers re-encoding so each exported shot lands cleanly on the detected boundaries.

Write the clips into a specific directory:

```bash
sentrysearch shots /path/to/video.mp4 --split --output-dir ./my_shots
```

This runs [TransNetV2](https://github.com/soCzech/TransNetV2) as a standalone CLI feature. You can also reuse the same shot detector during indexing with:

```bash
sentrysearch index /path/to/video/footage --segmentation shot
```

### Search

```bash
$ sentrysearch search "red truck running a stop sign"
  #1 [0.87] front_2024-01-15_14-30.mp4 @ 02:15-02:45
  #2 [0.74] left_2024-01-15_14-30.mp4 @ 02:10-02:40
  #3 [0.61] front_2024-01-20_09-15.mp4 @ 00:30-01:00

Saved clip: ./match_front_2024-01-15_14-30_02m15s-02m45s.mp4
```

If the best result's similarity score is below the confidence threshold (default 0.41), you'll be prompted before trimming:

```
No confident match found (best score: 0.28). Show results anyway? [y/N]:
```

With `--no-trim`, low-confidence results are shown with a note instead of a prompt.

Options: `--results N`, `--output-dir DIR`, `--no-trim` to skip auto-trimming, `--threshold 0.5` to adjust the confidence cutoff, `--save-top N` to save the top N clips instead of just the best match, `--rerank` to apply Qwen VL reranking for higher precision (Qwen backend only), `--segmentation shot` to search a shot-based index instead of the default chunk-based index. Backend, model, and segmentation are auto-detected from the index — pass `--backend`, `--model`, or `--segmentation` only to override. `--model` is local-backend only.

### Shot Labeling

Generate strict JSON labels for a single shot clip or an entire directory of split shots:

```bash
# label one clip
sentrysearch label /path/to/shot_001.mp4

# label every .mp4/.mov clip in a directory
sentrysearch label /path/to/shots
```

By default each result is written next to the source clip as `<clip>.label.json`, using the `gemini-3.1-flash-lite-preview` model and a fixed schema tuned for retrieval and remix workflows.

Useful options:

```bash
# write labels into a separate directory
sentrysearch label /path/to/shots --output-dir ./labels

# re-run and replace existing JSON
sentrysearch label /path/to/shots --overwrite

# override the Gemini model
sentrysearch label /path/to/shots --model your-model-name
```

### `yt-dlp` passthrough

`sentrysearch` now bundles a transparent `yt-dlp` passthrough command:

```bash
sentrysearch yt-dlp [yt-dlp args...]
```

This forwards arguments directly to the upstream `yt-dlp` module, preserving its help text, output, and exit codes.

Examples:

```bash
# Download a single video
sentrysearch yt-dlp "https://www.youtube.com/watch?v=VIDEO_ID"

# Print metadata as JSON
sentrysearch yt-dlp --dump-single-json "https://www.youtube.com/watch?v=VIDEO_ID"

# List available formats
sentrysearch yt-dlp -F "https://www.youtube.com/watch?v=VIDEO_ID"

# Extract audio only
sentrysearch yt-dlp -x --audio-format mp3 "https://www.youtube.com/watch?v=VIDEO_ID"

# Download subtitles
sentrysearch yt-dlp --write-subs --sub-langs en --skip-download "https://www.youtube.com/watch?v=VIDEO_ID"

# Work with playlists
sentrysearch yt-dlp --flat-playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

For the full upstream option set, run:

```bash
sentrysearch yt-dlp --help
```

### Local Backend (no API key needed)

Index and search using a local Qwen3-VL-Embedding model instead of a remote API. Free, private, and runs entirely on your machine. For the best search quality, use the Gemini backend — the local 8B model is a solid alternative when you need offline/private search, and the 2B model is a fallback when hardware can't support 8B.

The model is **auto-detected from your hardware** — qwen8b for NVIDIA GPUs and Macs with 24 GB+ RAM, qwen2b for smaller Macs and CPU-only systems. You can override with `--model qwen2b` or `--model qwen8b`. Pick an install based on your hardware:

| Hardware | Install command | Auto-detected model | Notes |
|---|---|---|---|
| **Apple Silicon, 24 GB+ RAM** | `uv tool install ".[local]"` | qwen8b | Full float16 via MPS |
| **Apple Silicon, 16 GB RAM** | `uv tool install ".[local]"` | qwen2b | 8B won't fit; 2B uses ~6 GB |
| **Apple Silicon, 8 GB RAM** | `uv tool install ".[local]"` | qwen2b | Tight — may swap under load; Gemini API recommended instead |
| **NVIDIA, 18 GB+ VRAM** | `uv tool install ".[local]"` | qwen8b | Full bf16 precision |
| **NVIDIA, 8–16 GB VRAM** | `uv tool install ".[local-quantized]"` | qwen8b | 4-bit quantization (~6–8 GB) |

> **Won't work well:** Intel Macs and machines without a dedicated GPU. These fall back to CPU with float32 — too slow and memory-hungry for practical use. Use the **Gemini API backend** (the default) instead.

> **Not sure?** On Mac, use `".[local]"`. On NVIDIA, use `".[local-quantized]"` — 4-bit quantization works on the widest range of NVIDIA hardware with minimal quality loss. (bitsandbytes requires CUDA and does not work on Mac/MPS.)

**Mac prerequisite:** Install system FFmpeg (the local model's video processor requires it — the Gemini backend uses a bundled ffmpeg instead):

```bash
brew install ffmpeg
```

Index with `--backend local` and search — no extra flags needed:

```bash
sentrysearch index /path/to/footage --backend local
sentrysearch search "car running a red light"
```

The search command auto-detects the backend and model from whatever you indexed with. You can also use `--model` as a shorthand — it implies `--backend local`:

```bash
sentrysearch index /path/to/footage --model qwen2b   # same as --backend local --model qwen2b
sentrysearch search "car running a red light"          # auto-detects local/qwen2b from index
```

Options:
- `--model qwen2b` — smaller model, lower quality but only ~6 GB memory (also accepts full HuggingFace IDs)
- `--quantize` / `--no-quantize` — force 4-bit quantization on or off (default: auto-detect based on whether bitsandbytes is installed)

Notes:
- First run downloads the model (~16 GB for 8B, ~4 GB for 2B).
- Embeddings from different backends and models are **not compatible**. Each backend/model combination gets its own isolated index, so they can't accidentally mix. If you search with a model that has no indexed data, you'll be told which model was actually used.
- Speed varies by GPU core count — base M-series chips are slower than Pro/Max but produce identical results.

### Why the local model is fast

The local backend stays fast and memory-efficient through a few techniques that compound:

- **Preprocessing shrinks chunks before they hit the model.** Each 30s chunk is downscaled to 480p at 5fps via ffmpeg before embedding. A ~19 MB dashcam chunk becomes ~1 MB — a 95% reduction in pixels the model has to process. Model inference time scales with pixel count, not video duration, so this is the single biggest speedup.
- **Low frame sampling.** The video processor sends at most 32 frames per chunk to the model (`fps=1.0`, `max_frames=32`). A 30-second chunk produces ~30 frames — not hundreds.
- **MRL dimension truncation.** Qwen3-VL-Embedding supports [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147). Only the first 768 dimensions of each embedding are kept and L2-normalized, reducing storage and distance computation in ChromaDB.
- **Auto-quantization.** On NVIDIA GPUs with limited VRAM, the 8B model is automatically loaded in 4-bit (bitsandbytes) — dropping from ~18 GB to ~6-8 GB with minimal quality loss. A 4090 (24 GB) runs the full bf16 model with headroom to spare.
- **Still-frame skipping.** Chunks with no meaningful visual change (e.g. a parked car) are detected by comparing JPEG file sizes across sampled frames and skipped entirely — saving a full forward pass per chunk.

With all of this, expect ~2-5s per chunk on an A100 and ~3-8s on a T4. On a 4090, the 8B model in bf16 should be in the low single digits per chunk.

### Tesla Metadata Overlay

Burn speed, location, and time onto trimmed clips:

```bash
sentrysearch search "car cutting me off" --overlay
```

This extracts telemetry embedded in Tesla dashcam files (speed, GPS) and renders a HUD overlay. The overlay shows:

- **Top center:** speed and MPH label on a light gray card
- **Below card:** date and time (12-hour with AM/PM)
- **Top left:** city and road name (via reverse geocoding)

![tesla overlay](docs/tesla-overlay.png)

Requirements:

- Tesla firmware 2025.44.25 or later, HW3+
- SEI metadata is only present in driving footage (not parked/Sentry Mode)
- Reverse geocoding uses [OpenStreetMap's Nominatim API](https://nominatim.openstreetmap.org/) via geopy (optional)

Install with Tesla overlay support:

```bash
uv tool install ".[tesla]"
```

Without geopy, the overlay still works but omits the city/road name.

Source: [teslamotors/dashcam](https://github.com/teslamotors/dashcam)

### Managing the index

```bash
# Show index info (files marked [missing] no longer exist on disk)
sentrysearch stats

# Remove specific files by path substring
sentrysearch remove path/to/footage

# Wipe the entire index
sentrysearch reset
```

All three commands also accept `--segmentation chunk|shot` to target a specific index mode.

### Verbose mode

Add `--verbose` to either command for debug info (embedding dimensions, API response times, similarity scores).

## How is this possible?

All supported backends — Gemini Embedding 2, Doubao ARK, Qwen VL, and the local Qwen3-VL-Embedding — can natively embed video: raw video pixels are projected into the same vector space as text queries. There's no transcription, no frame captioning, no text middleman. A text query like "red truck at a stop sign" is directly comparable to a 30-second video clip at the vector level. This is what makes sub-second semantic search over hours of footage practical.

## Cost

Indexing 1 hour of footage costs ~$2.84 with Gemini's embedding API (default settings: 30s chunks, 5s overlap):

> 1 hour = 3,600 seconds of video = 3,600 frames processed by the model.
> 3,600 frames × $0.00079 = ~$2.84/hr

The Gemini API natively extracts and tokenizes exactly 1 frame per second from uploaded video, regardless of the file's actual frame rate. The preprocessing step (which downscales chunks to 480p at 5fps via ffmpeg) is a local/bandwidth optimization — it keeps payload sizes small so API requests are fast and don't timeout — but does not change the number of frames the API processes.

Two built-in optimizations help reduce costs in different ways:

- **Preprocessing** (on by default) — chunks are downscaled to 480p at 5fps before uploading. Since the API processes at 1fps regardless, this only reduces upload size and transfer time, not the number of frames billed. It primarily improves speed and prevents request timeouts.
- **Still-frame skipping** (on by default) — chunks with no meaningful visual change (e.g. a parked car) are skipped entirely. This saves real API calls and directly reduces cost. The savings depend on your footage — Sentry Mode recordings with hours of idle time benefit the most, while action-packed driving footage may have nothing to skip.

Search queries are negligible (text embedding only).

Tuning options:

- `--chunk-duration` / `--overlap` — longer chunks with less overlap = fewer API calls = lower cost
- `--no-skip-still` — embed every chunk even if nothing is happening
- `--target-resolution` / `--target-fps` — adjust preprocessing quality
- `--no-preprocess` — send raw chunks to the API

## Known Warnings (harmless)

The local backend may print warnings during indexing and search. These are cosmetic and don't affect results:

- **`MPS: nonzero op is not natively supported`** — A known PyTorch limitation on Apple Silicon. The operation falls back to CPU for one step; everything else stays on the GPU. No impact on output quality.
- **`video_reader_backend torchcodec error, use torchvision as default`** — torchcodec can't find a compatible FFmpeg on macOS. The video processor falls back to torchvision automatically. This is expected and produces identical results.
- **`You are sending unauthenticated requests to the HF Hub`** — The model downloads from Hugging Face without a token. Download speeds may be slightly lower, but the model loads fine. Set a `HF_TOKEN` environment variable to silence this if it bothers you.

## Limitations & Future Work

- **Still-frame detection is heuristic** — it uses JPEG file size comparison across sampled frames. It may occasionally skip chunks with subtle motion or embed chunks that are truly static. Disable with `--no-skip-still` if you need every chunk indexed.
- **Search quality depends on chunk boundaries** — if an event spans two chunks, the overlapping window helps but isn't perfect. Smarter chunking (e.g. scene detection) could improve this.
- **Gemini Embedding 2 is in preview** — API behavior and pricing may change.

## Compatibility

This works with `.mp4` and `.mov` footage, not just Tesla Sentry Mode. The directory scanner recursively finds both file types regardless of folder structure.

## Requirements

- Python 3.11+
- `ffmpeg` on PATH, or use bundled ffmpeg via `imageio-ffmpeg` (installed by default)
- **Gemini backend:** Gemini API key ([get one free](https://aistudio.google.com/apikey))
- **Doubao backend:** ARK API key ([get one](https://console.volcengine.com/ark))
- **Qwen backend:** DashScope API key ([get one](https://bailian.console.aliyun.com/))
- **Local backend:**
  - GPU with CUDA or Apple Metal (see [hardware table](#local-backend-no-api-key-needed) for VRAM/RAM requirements)
  - **macOS:** `brew install ffmpeg` (required by the video decoder)
  - **Linux/Windows:** no extra system dependencies
