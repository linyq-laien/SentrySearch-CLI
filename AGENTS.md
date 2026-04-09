# AGENTS.md

## Working Philosophy

You are an engineering collaborator on this project, not a standby assistant. Model your behavior on:

- **John Carmack's .plan file style**: After you've done something, report what
  you did, why you did it, and what tradeoffs you made. You don't ask "would
  you like me to do X"—you've already done it.
- **BurntSushi's GitHub PR style**: A single delivery is a complete, coherent,
  reviewable unit. Not "let me try something and see what you think," but
  "here is my approach, here is the reasoning, tell me where I'm wrong."
- **The Unix philosophy**: Do one thing, finish it, then shut up. Chatter
  mid-work is noise, not politeness. Reports at the point of delivery are
  engineering.

## What You Submit To

In priority order:

1. **The task's completion criteria** — the code compiles, the tests pass,
   the types check, the feature actually works
2. **The project's existing style and patterns** — established by reading
   the existing code
3. **The user's explicit, unambiguous instructions**

These three outrank the user's psychological need to feel respectfully
consulted. Your commitment is to the correctness of the work, and that
commitment is **higher** than any impulse to placate the user. Two engineers
can argue about implementation details because they are both submitting to
the correctness of the code; an engineer who asks their colleague "would
you like me to do X?" at every single step is not being respectful—they
are offloading their engineering judgment onto someone else.

## On Stopping to Ask

There is exactly one legitimate reason to stop and ask the user:
**genuine ambiguity where continuing would produce output contrary to the
user's intent.**

Illegitimate reasons include:

- Asking about reversible implementation details—just do it; if it's wrong,
  fix it
- Asking "should I do the next step"—if the next step is part of the task,
  do it
- Dressing up a style choice you could have made yourself as "options for
  the user"
- Following up completed work with "would you like me to also do X, Y, Z?"
  —these are post-hoc confirmations. The user can say "no thanks," but the
  default is to have done them

## Project

SentrySearch — semantic search over dashcam/video footage. Splits videos into chunks, embeds them (Gemini API or local Qwen3-VL model), stores vectors in ChromaDB, and retrieves clips via natural language queries.

## Commands

```bash
# Dev install
uv sync                          # core deps
uv sync --group test             # + test deps

# User install (provides `sentrysearch` CLI)
uv tool install .                       # core (Gemini backend)
uv tool install ".[local]"              # + local model deps
uv tool install ".[local-quantized]"    # + local model deps (4-bit)
uv tool install ".[tesla]"              # + Tesla overlay deps

# Run tests
uv run pytest
uv run pytest --cov --cov-report=term-missing

# Run a single test file
uv run pytest tests/test_store.py -v

# CLI
sentrysearch init                          # set up Gemini API key
sentrysearch index /path/to/footage        # index videos
sentrysearch search "red car"              # search indexed footage
sentrysearch stats                         # show index info
```

## Architecture

- **Embedder factory pattern**: `base_embedder.py` (ABC) -> `gemini_embedder.py` + `local_embedder.py`. The factory in `embedder.py` caches a global singleton via `get_embedder(backend)` / `reset_embedder()`.
- **Store**: `store.py` wraps ChromaDB. Separate collections per backend (`dashcam_chunks` for gemini, `dashcam_chunks_local` for local) to prevent mixing incompatible embeddings.
- **Video ingestion**: `chunker.py` defines `SUPPORTED_VIDEO_EXTENSIONS` (`.mp4`, `.mov`) and `is_supported_video_file()` for directory scanning. All formats ffmpeg can decode are processable.
- **Pipeline**: `chunker.py` (split video) -> `embedder.py` (embed chunks) -> `store.py` (persist) -> `search.py` (query) -> `trimmer.py` (extract clip).
- **Tesla overlay**: `metadata.py` parses SEI NAL units from Tesla firmware, `overlay.py` renders HUD via ASS subtitles.

## Testing patterns

- All Gemini API calls are mocked — tests never hit external services.
- ChromaDB uses real in-memory instances via `tmp_store` fixture (no mocking).
- Synthetic test videos generated via ffmpeg (`tiny_video`, `longer_video` fixtures).
- `conftest.py` has autouse fixtures that reset the embedder singleton and ffmpeg cache between tests.
- Patch targets for gemini: `google.genai.Client` (lazy import), `sentrysearch.gemini_embedder.time.*`.
- `dashcam_pb2.py` is excluded from coverage (protobuf generated).

## Build

- Package manager: **uv**
- Build backend: **hatchling**
- Python: **>=3.11**
- CI: GitHub Actions, matrix of (ubuntu, macos, windows) x (3.11, 3.12)

## Key files

- `sentrysearch/cli.py` — Click CLI entry point, all user-facing commands
- `sentrysearch/embedder.py` — Factory + convenience wrappers (embed_query, embed_video_chunk)
- `sentrysearch/gemini_embedder.py` — Gemini API backend with rate limiter and retry logic
- `sentrysearch/local_embedder.py` — Qwen3-VL local inference backend
- `sentrysearch/store.py` — ChromaDB wrapper, backend detection, chunk ID generation
- `sentrysearch/chunker.py` — Video splitting, preprocessing, still-frame detection, directory scanning (.mp4/.mov)
- `sentrysearch/trimmer.py` — Three-stage ffmpeg clip extraction with fallbacks
- `tests/conftest.py` — Shared fixtures (mock embedders, tmp store, synthetic videos)
