---
name: video-download
description: 使用 SentrySearch 内置的 yt-dlp 代理下载在线视频、播放列表、字幕或元数据到本地。适用于 Agent 需要先获取待索引视频素材、查看可用格式、导出 JSON 元数据，或在本仓库内统一通过 `uv run sentrysearch yt-dlp ...` 转发任意 yt-dlp 参数的场景。
---

# Video Download

使用仓库内置的 `sentrysearch yt-dlp` 包装器，不要直接改用裸 `yt-dlp`，除非用户明确要求。

## 快速规则

1. 在仓库根目录执行命令。
2. 优先使用 `uv run sentrysearch yt-dlp ...`。
3. 保留上游 `yt-dlp` 参数原样透传；这个包装器会保留帮助、输出和退出码。
4. 下载完成后，检查目标文件或命令输出，确认素材已落盘。

## 常用流程

### 1. 下载单个视频

```bash
uv run sentrysearch yt-dlp "https://www.youtube.com/watch?v=VIDEO_ID"
```

### 2. 指定输出目录和文件名模板

```bash
mkdir -p /Users/apple/Desktop/saas-test
uv run sentrysearch yt-dlp \
  -o "/Users/apple/Desktop/saas-test/%(title)s.%(ext)s" \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

### 3. 只看元数据，不下载文件

```bash
uv run sentrysearch yt-dlp --dump-single-json "https://www.youtube.com/watch?v=VIDEO_ID"
```

### 4. 查看可用清晰度/编码

```bash
uv run sentrysearch yt-dlp -F "https://www.youtube.com/watch?v=VIDEO_ID"
```

### 5. 下载字幕或播放列表

```bash
uv run sentrysearch yt-dlp --write-subs --sub-langs en --skip-download "https://www.youtube.com/watch?v=VIDEO_ID"
uv run sentrysearch yt-dlp --flat-playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

### 6. 下载 TikTok 博主主页的全部视频

先用 `--flat-playlist` 快速确认主页可枚举：

```bash
uv run sentrysearch yt-dlp --flat-playlist "https://www.tiktok.com/@livia._.paige"
```

再把主页下全部视频下载到指定目录：

```bash
mkdir -p /Users/apple/Desktop/saas-test/tiktok-livia
uv run sentrysearch yt-dlp \
  -o "/Users/apple/Desktop/saas-test/tiktok-livia/%(uploader)s/%(upload_date)s_%(id)s.%(ext)s" \
  "https://www.tiktok.com/@livia._.paige"
```

如果只想先小范围验证，可先限制前 3 条：

```bash
uv run sentrysearch yt-dlp \
  --playlist-end 3 \
  -o "/Users/apple/Desktop/saas-test/tiktok-livia/%(uploader)s/%(upload_date)s_%(id)s.%(ext)s" \
  "https://www.tiktok.com/@livia._.paige"
```

若遇到 TikTok 提示 impersonation / 风控相关警告，先记录原始输出，再按需补充 cookies 或浏览器会话参数，不要擅自更换下载入口。

## 交付检查

- 若用户要“下载视频供后续索引”，优先把文件保存到明确目录，例如 `/Users/apple/Desktop/saas-test/`。
- 若命令退出码非 0，视为下载失败，继续根据原始 `yt-dlp` 输出排查。
- 若参数不确定，先运行：

```bash
uv run sentrysearch yt-dlp --help
```

## 与本项目的关系

- 该命令本质上执行 `python -m yt_dlp`，但统一走 `sentrysearch` CLI，便于其他 Agent 复用项目内同一入口。
- 下载到本地的视频可以直接交给 `uv run sentrysearch index ...` 做后续索引。
