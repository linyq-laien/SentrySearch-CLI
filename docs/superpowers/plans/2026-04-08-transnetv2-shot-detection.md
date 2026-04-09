# TransNetV2 Shot Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional TransNetV2-based shot detection to `sentrysearch index` so indexing can align chunks to shot boundaries without breaking the existing fixed-window flow.

**Architecture:** Keep the new behavior isolated in a dedicated shot-detection module and a small `chunker` integration layer. The CLI only selects the chunking mode and passes detector parameters through; chunk cleanup, preprocessing, embedding, and storage continue to use the existing indexing pipeline.

**Tech Stack:** Click, ffmpeg/ffprobe, optional `transnetv2-pytorch`, pytest

---

### Task 1: Lock the CLI contract

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `sentrysearch/cli.py`

- [ ] **Step 1: Write the failing test**

Add tests asserting `sentrysearch index` accepts a shot-aware chunking flag and forwards shot parameters into `chunk_video`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -k shot -v`
Expected: FAIL because the CLI options and forwarding logic do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add the Click options and thread the selected chunking mode/shot parameters through the indexing loop.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -k shot -v`
Expected: PASS

### Task 2: Add shot-aware chunk generation

**Files:**
- Modify: `tests/test_chunker.py`
- Create: `sentrysearch/shot_detector.py`
- Modify: `sentrysearch/chunker.py`

- [ ] **Step 1: Write the failing test**

Add unit tests covering scene-to-chunk conversion, short-shot merging, and fallback splitting of long scenes.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunker.py -k shot -v`
Expected: FAIL because the helper and detector entrypoint do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement the shot detector wrapper plus `chunk_video(..., strategy="shots", ...)`, keeping the fixed strategy untouched.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chunker.py -k shot -v`
Expected: PASS

### Task 3: Document and verify the optional dependency path

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

No code test; verify behavior with an error-path CLI test already added in Task 1.

- [ ] **Step 2: Write minimal implementation**

Add an optional dependency group and document install/use/error behavior for shot detection.

- [ ] **Step 3: Run verification**

Run: `uv run pytest tests/test_cli.py tests/test_chunker.py -v`
Expected: PASS
