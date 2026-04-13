---
name: index-shot-publish-saas
description: 对本地单个视频或视频目录执行 shot 分段索引，并在本地写入向量索引的同时把每个分段发布到 video-saas。适用于 Agent 需要执行或复用 `uv run sentrysearch index PATH --segmentation shot --publish-saas`，尤其是处理类似 `/Users/apple/Desktop/saas-test/video1.mp4` 这类本地素材的场景。
---

# Index Shot Publish SaaS

围绕这一条命令工作：

```bash
uv run sentrysearch index /Users/apple/Desktop/saas-test/video1.mp4 --segmentation shot --publish-saas
```

## 先决条件

1. 在仓库根目录执行。
2. 待处理文件存在，例如 `/Users/apple/Desktop/saas-test/video1.mp4`。
3. 已配置 video-saas 环境变量；CLI 会优先读取 `~/.sentrysearch/.env`，再读取当前目录 `.env`：

```bash
VIDEO_SAAS_BASE_URL=http://localhost:8000
VIDEO_SAAS_INTEGRATION_KEY=your-key
VIDEO_SAAS_INTEGRATION_SECRET=your-secret
```

4. 明白默认后端：如果命令里**没有**显式写 `--backend` 或 `--model`，当前实现会默认使用 `gemini`。
   - 如果你想强制本地模型，改用：

```bash
uv run sentrysearch index /Users/apple/Desktop/saas-test/video1.mp4 \
  --segmentation shot \
  --backend local \
  --model qwen2b \
  --publish-saas
```

## 标准执行步骤

### 1. 检查输入文件

```bash
ls -lh /Users/apple/Desktop/saas-test/video1.mp4
```

### 2. 执行主命令

```bash
uv run sentrysearch index /Users/apple/Desktop/saas-test/video1.mp4 --segmentation shot --publish-saas
```

### 3. 读取成功信号

预期会看到以下类型的输出：

- `Indexing file 1/1: video1.mp4 [chunk X/Y]`
- 若本地已索引过且这次仍带 `--publish-saas`，会出现 `publishing to video-saas`
- 结束时会出现 `Indexed ... Total: ...`

### 4. 校验本地索引

```bash
uv run sentrysearch stats --segmentation shot
```

## 这个命令实际做什么

1. 按镜头边界而不是固定窗口切段。
2. 对每个分段做嵌入并写入本地 ChromaDB 的 `shot` 索引。
3. 为每个分段创建 video-saas 上传会话、上传分段文件，并注册带 embedding 的 segment。
4. 临时切片文件会在流程结束后自动清理。

## 常见问题

### 缺少 video-saas 配置

如果报错包含 `Missing video-saas configuration`，补齐上述 3 个环境变量后重试。

### 误以为这是本地模型索引

这条精确命令默认不是 local backend，而是 `gemini`。只有显式提供 `--backend local` 或 `--model ...` 才会走本地模型。

### 只想做本地索引，不上传 SaaS

去掉 `--publish-saas`：

```bash
uv run sentrysearch index /Users/apple/Desktop/saas-test/video1.mp4 --segmentation shot
```
