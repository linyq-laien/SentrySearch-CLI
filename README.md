# SentrySearch

视频语义搜索。输入你想找的内容，直接获得剪辑片段。

[OpenClaw Skill](https://clawhub.ai/ssrajadh/natural-language-video-search)

[<video src="https://github.com/ssrajadh/sentrysearch/raw/main/docs/demo.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/baf98fad-080b-48e1-97f5-a2db2cbd53f5)

## 工作原理

SentrySearch 将视频切分为重叠片段（或检测到的镜头），通过多种后端之一——Google Gemini Embedding API、字节跳动 Doubao ARK、阿里 Qwen VL（DashScope）或本地 Qwen3-VL 模型——对每个片段进行视频嵌入，并将向量存储在本地 ChromaDB 数据库中。搜索时，文本查询会被嵌入到同一向量空间中，与已存储的视频嵌入进行匹配。最佳匹配结果会自动从原始文件中裁剪并保存为片段。

## 快速开始

1. 安装 [uv](https://docs.astral.sh/uv/)（如果尚未安装）：

**macOS/Linux：**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows：**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```


2. 安装 ffmpeg（视频分块和裁剪必需）：

**macOS：**
```bash
brew install ffmpeg
```

**Ubuntu/Debian：**
```bash
sudo apt install ffmpeg
```

**Windows：**
```powershell
winget install ffmpeg
```

> 如果系统未安装 ffmpeg，将自动使用内置的 `imageio-ffmpeg` 作为 fallback（部分功能如镜头检测的帧率探测需要系统 ffmpeg）。

3. 克隆并安装：

```bash
git clone https://github.com/ssrajadh/sentrysearch.git
cd sentrysearch
uv tool install .
```

4. 配置 API Key（或[使用本地模型](#本地后端无需-api-key)）：

```bash
sentrysearch init
```

该命令会提示输入 Gemini API Key，写入 `~/.sentrysearch/.env`，并通过测试嵌入进行验证。你也可以配置其他后端：

```bash
sentrysearch init --backend doubao    # Doubao ARK（火山引擎）
sentrysearch init --backend qwen      # Qwen VL（DashScope / 阿里云）
```

| 后端 | 环境变量 | 获取 API Key |
|---|---|---|
| **gemini**（默认） | `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **doubao** | `ARK_API_KEY` | [console.volcengine.com/ark](https://console.volcengine.com/ark) |
| **qwen** | `DASHSCOPE_API_KEY` | [bailian.console.aliyun.com](https://bailian.console.aliyun.com/) |

5. 索引视频素材：

```bash
sentrysearch index /path/to/footage
```

使用非默认后端：

```bash
sentrysearch index /path/to/footage --backend doubao
sentrysearch index /path/to/footage --backend qwen
```

6. 搜索：

```bash
sentrysearch search "闯红灯的红色卡车"
```

搜索会自动从索引中检测后端——索引后无需额外参数。

> **手动配置：** 如果不想使用 `sentrysearch init`，可以复制 `.env.example` 为 `.env`，手动填入从 [aistudio.google.com/apikey](https://aistudio.google.com/apikey) 获取的 API Key。

7. 发布到 video-saas：

```bash
uv run sentrysearch index /Users/apple/test1.mp4 --segmentation shot --publish-saas  --skip-low-quality  --verbose 
```

> **提示：** 需要先配置 `VIDEO_SAAS_BASE_URL` 和 `VIDEO_SAAS_INTEGRATION_KEY` 环境变量。

## 用法

### 初始化

```bash
$ sentrysearch init
Enter your Gemini API key (get one at https://aistudio.google.com/apikey): ****
Validating API key...
Setup complete. You're ready to go — run `sentrysearch index <directory>` to get started.
```

其他后端，传入 `--backend`：

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

如果 Key 已经配置过，会提示是否覆盖。

> **提示：** 在 [aistudio.google.com/billing](https://aistudio.google.com/billing) 设置消费限额，防止 Gemini 意外超支。

### 索引视频素材

```bash
$ sentrysearch index /path/to/video/footage
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 1/4]
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 2/4]
...
Indexed 12 new chunks from 3 files. Total: 12 chunks from 3 files.
```

**后端选择：**

- 默认：`--backend gemini`（Gemini Embedding API）
- `--backend doubao` — 使用 Doubao ARK 多模态嵌入
- `--backend qwen` — 使用 Qwen VL 多模态嵌入（DashScope）
- `--backend local` — 使用本地模型代替远程 API（[详见下文](#本地后端无需-api-key)）

**分段模式：**

- 默认：固定时间窗口（`--segmentation chunk`）
- `--segmentation shot` — 按检测到的镜头分段索引，而非固定窗口

**示例：完整的 Qwen 流程**

```bash
sentrysearch init --backend qwen
sentrysearch index /path/to/footage --backend qwen
sentrysearch search "闯红灯的红色卡车"
```

**示例：镜头分段索引**

```bash
sentrysearch index /path/to/footage --segmentation shot
sentrysearch search "别车的车辆"
```

**示例：本地索引后直接发布到 video-saas**

```bash
export VIDEO_SAAS_BASE_URL=http://localhost:8000
export VIDEO_SAAS_INTEGRATION_KEY=int_xxx
export VIDEO_SAAS_INTEGRATION_SECRET=secret_xxx

sentrysearch index /path/to/footage --segmentation shot --publish-saas
```

**示例：上传后直接绑定到指定合集**

```bash
sentrysearch index /path/to/footage \
  --segmentation shot \
  --shot-threshold 0.9 \
  --publish-saas \
  --publish-collection "collection_xxx"
```

**示例：跳过所有 low-quality shot 片段（不做 embedding / 不上传）**

```bash
sentrysearch index /path/to/footage \
  --segmentation shot \
  --publish-saas \
  --skip-low-quality
```

**示例：Qwen 后端 + 重排序**

```bash
sentrysearch index /path/to/footage --backend qwen
sentrysearch search "闯红灯的车辆" --rerank
```

选项：

- `--chunk-duration 30` — 每个片段的时长（秒）
- `--overlap 5` — 片段之间的重叠时长（秒）
- `--segmentation shot` — 按检测到的镜头分段索引，而非固定窗口
- `--shot-threshold 0.5` — 使用 `--segmentation shot` 时的镜头检测阈值
- `--segmentation shot` 时会额外做片段质量校验，并给分段打上质量元数据：
  - `< 0.5s` → `too_short`
  - ffmpeg `mpdecimate` 判断为重复静态帧视频 → `still_frame`
  - 片段内部再次检测到多个 scene → `internal_scene_cut`
- `--no-preprocess` — 跳过降分辨率/降帧率（发送原始片段）
- `--target-resolution 480` — 预处理目标高度（像素）
- `--target-fps 5` — 预处理目标帧率
- `--no-skip-still` — 嵌入所有片段，包括无视觉变化的静帧
- `--publish-saas` — 在本地分段和 embedding 之后，把每个片段上传到 video-saas（注册 source video、申请上传会话、上传到 R2、注册 segment）
- `--publish-collection "collection_xxx"` — 与 `--publish-saas` 搭配使用；按合集 id 绑定上传后的 segments
- `--skip-low-quality` — 跳过被标记为 low-quality 的 shot 片段（`too_short` / `still_frame` / `internal_scene_cut`），不做 embedding、不写入本地库、也不上传到 video-saas

启用 `--publish-saas` 时，需要以下环境变量：

- `VIDEO_SAAS_BASE_URL`
- `VIDEO_SAAS_INTEGRATION_KEY`
- `VIDEO_SAAS_INTEGRATION_SECRET`
- 可选：`VIDEO_SAAS_TIMEOUT_SECONDS`

### TransNetV2 镜头检测 / 分割

镜头检测功能已包含在默认安装中，无需额外步骤。

检测镜头边界：

```bash
sentrysearch shots /path/to/video.mp4
```

将视频按镜头分割为独立片段：

```bash
sentrysearch shots /path/to/video.mp4 --split
```

`--split` 优先使用重编码，确保每个导出的镜头精确落在检测到的边界上。

将片段写入指定目录：

```bash
sentrysearch shots /path/to/video.mp4 --split --output-dir ./my_shots
```

该功能基于 [TransNetV2](https://github.com/soCzech/TransNetV2) 作为独立 CLI 特性运行。你也可以在索引时复用同一镜头检测器：

```bash
sentrysearch index /path/to/video/footage --segmentation shot
```

当使用 `--segmentation shot` 进行索引时，CLI 会对每个切出来的镜头片段做一次质量复核：

- 时长小于 `0.5s` 的片段会标记为 `too_short`
- 用 ffmpeg `mpdecimate` 去重后仅保留极少帧的“单图重复播放”片段会标记为 `still_frame`
- 片段内部如果再次检测到多个镜头，则会标记为 `internal_scene_cut`

这些质量标记会：

- 写入本地 ChromaDB metadata
- 在 `--publish-saas` 时透传到 segment 的 `extension_metadata`
- 在 CLI 输出中显示 low-quality warning / summary

默认情况下，`--skip-still` 仍然生效，所以被识别为静态帧的片段会被跳过嵌入；如果你要强制保留这些片段，请显式传入 `--no-skip-still`。

### 搜索

```bash
$ sentrysearch search "闯红灯的红色卡车"
  #1 [0.87] front_2024-01-15_14-30.mp4 @ 02:15-02:45
  #2 [0.74] left_2024-01-15_14-30.mp4 @ 02:10-02:40
  #3 [0.61] front_2024-01-20_09-15.mp4 @ 00:30-01:00

Saved clip: ./match_front_2024-01-15_14-30_02m15s-02m45s.mp4
```

如果最佳结果的相似度分数低于置信阈值（默认 0.41），裁剪前会提示确认：

```
No confident match found (best score: 0.28). Show results anyway? [y/N]:
```

使用 `--no-trim` 时，低置信度结果只会显示提示，不会弹确认。

选项：`--results N`、`--output-dir DIR`、`--no-trim` 跳过自动裁剪、`--threshold 0.5` 调整置信阈值、`--save-top N` 保存前 N 个片段而非仅最佳匹配、`--rerank` 应用 Qwen VL 重排序以提升精度（仅 Qwen 后端）、`--segmentation shot` 搜索镜头分段索引而非默认的固定窗口索引。后端、模型和分段模式会从索引中自动检测——仅在需要覆盖时传入 `--backend`、`--model` 或 `--segmentation`。`--model` 仅限本地后端使用。

### 镜头标注

为单个镜头片段或整个目录的分割镜头生成结构化 JSON 标签：

```bash
# 标注单个片段
sentrysearch label /path/to/shot_001.mp4

# 标注目录下所有 .mp4/.mov 片段
sentrysearch label /path/to/shots
```

默认将结果写在源片段旁边，文件名为 `<clip>.label.json`，使用 `gemini-3.1-flash-lite-preview` 模型和专为检索与混剪工作流优化的固定 schema。

常用选项：

```bash
# 将标签写入单独的目录
sentrysearch label /path/to/shots --output-dir ./labels

# 重新运行并覆盖已有 JSON
sentrysearch label /path/to/shots --overwrite

# 覆盖 Gemini 模型
sentrysearch label /path/to/shots --model your-model-name
```

### `yt-dlp` 代理命令

`sentrysearch` 内置了透明的 `yt-dlp` 代理命令：

```bash
sentrysearch yt-dlp [yt-dlp 参数...]
```

该命令将参数直接转发给上游 `yt-dlp` 模块，保留其帮助文本、输出和退出码。

示例：

```bash
# 下载单个视频
sentrysearch yt-dlp "https://www.youtube.com/watch?v=VIDEO_ID"

# 以 JSON 格式输出元数据
sentrysearch yt-dlp --dump-single-json "https://www.youtube.com/watch?v=VIDEO_ID"

# 列出可用格式
sentrysearch yt-dlp -F "https://www.youtube.com/watch?v=VIDEO_ID"

# 仅提取音频
sentrysearch yt-dlp -x --audio-format mp3 "https://www.youtube.com/watch?v=VIDEO_ID"

# 下载字幕
sentrysearch yt-dlp --write-subs --sub-langs en --skip-download "https://www.youtube.com/watch?v=VIDEO_ID"

# 处理播放列表
sentrysearch yt-dlp --flat-playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

查看完整上游选项：

```bash
sentrysearch yt-dlp --help
```

### 本地后端（无需 API Key）

使用本地 Qwen3-VL-Embedding 模型代替远程 API 进行索引和搜索。免费、私密、完全在本地运行。如需最佳搜索质量，建议使用 Gemini 后端——本地 8B 模型是离线/私密搜索的可靠替代，2B 模型则适用于硬件不足以支持 8B 的场景。

模型会**根据硬件自动检测**——NVIDIA GPU 和 24 GB+ RAM 的 Mac 使用 qwen8b，较小内存的 Mac 和纯 CPU 系统使用 qwen2b。可通过 `--model qwen2b` 或 `--model qwen8b` 手动覆盖。根据硬件选择安装方式：

| 硬件 | 安装命令 | 自动检测模型 | 说明 |
|---|---|---|---|
| **Apple Silicon, 24 GB+ RAM** | `uv tool install ".[local]"` | qwen8b | 通过 MPS 完整 float16 |
| **Apple Silicon, 16 GB RAM** | `uv tool install ".[local]"` | qwen2b | 8B 放不下；2B 占用约 6 GB |
| **Apple Silicon, 8 GB RAM** | `uv tool install ".[local]"` | qwen2b | 较紧张——负载下可能交换内存；建议使用 Gemini API |
| **NVIDIA, 18 GB+ VRAM** | `uv tool install ".[local]"` | qwen8b | 完整 bf16 精度 |
| **NVIDIA, 8–16 GB VRAM** | `uv tool install ".[local-quantized]"` | qwen8b | 4-bit 量化（约 6-8 GB） |

> **不适用：** Intel Mac 和无独立 GPU 的机器。这些会回退到 CPU float32——太慢且内存消耗大，建议使用 **Gemini API 后端**（默认）。

> **不确定？** Mac 上使用 `".[local]"`。NVIDIA 上使用 `".[local-quantized]"`——4-bit 量化兼容最广泛的 NVIDIA 硬件，质量损失极小。（bitsandbytes 需要 CUDA，不支持 Mac/MPS。）

**Mac 前置条件：** 确保已安装系统 FFmpeg（已在[快速开始](#快速开始)第 2 步中安装）。本地模型的视频处理器需要系统 ffmpeg；Gemini 后端则使用内置 ffmpeg。

使用 `--backend local` 索引并搜索——无需额外参数：

```bash
sentrysearch index /path/to/footage --backend local
sentrysearch search "闯红灯的车辆"
```

搜索命令会自动从索引中检测后端和模型。你也可以使用 `--model` 作为简写——它隐含 `--backend local`：

```bash
sentrysearch index /path/to/footage --model qwen2b   # 等同于 --backend local --model qwen2b
sentrysearch search "闯红灯的车辆"                      # 从索引自动检测 local/qwen2b
```

选项：
- `--model qwen2b` — 更小的模型，质量略低但仅需约 6 GB 内存（也接受完整 HuggingFace ID）
- `--quantize` / `--no-quantize` — 强制开启/关闭 4-bit 量化（默认：根据 bitsandbytes 是否安装自动检测）

注意事项：
- 首次运行会下载模型（8B 约 16 GB，2B 约 4 GB）。
- 不同后端和模型的嵌入**互不兼容**。每个后端/模型组合拥有独立的隔离索引，不会意外混合。如果用未建索引的模型搜索，会提示实际使用的模型。
- 速度随 GPU 核心数变化——基础 M 系列芯片慢于 Pro/Max，但结果相同。

### 本地模型为何快速

本地后端通过多项叠加技术保持高效和低内存占用：

- **预处理在模型处理前压缩片段。** 每个约 30 秒的片段在嵌入前通过 ffmpeg 降分辨率至 480p 5fps。约 19 MB 的行车记录仪片段变为约 1 MB——像素量减少 95%。模型推理时间与像素量成正比，而非视频时长，因此这是最大的加速项。
- **低帧采样。** 视频处理器每个片段最多向模型发送 32 帧（`fps=1.0`、`max_frames=32`）。30 秒片段产生约 30 帧——而非数百帧。
- **MRL 维度截断。** Qwen3-VL-Embedding 支持 [Matryoshka 表示学习](https://arxiv.org/abs/2205.13147)。仅保留每个嵌入的前 768 维并做 L2 归一化，减少 ChromaDB 中的存储和距离计算。
- **自动量化。** 在 VRAM 有限的 NVIDIA GPU 上，8B 模型自动以 4-bit 加载（bitsandbytes）——从约 18 GB 降至约 6-8 GB，质量损失极小。4090（24 GB）可运行完整 bf16 模型且绰绰有余。
- **静帧跳过。** 通过比较采样帧的 JPEG 文件大小，检测无视觉变化的片段（如停放的车辆）并完全跳过——每片段节省一次完整前向传播。

综合以上优化，A100 上每片段约 2-5 秒，T4 上约 3-8 秒。4090 上 8B bf16 模型每片段应在个位数秒内。

### Tesla 元数据叠加

将速度、位置和时间信息烧录到裁剪片段上：

```bash
sentrysearch search "别我的车" --overlay
```

该功能提取 Tesla 行车记录仪文件中嵌入的遥测数据（速度、GPS）并渲染 HUD 叠加。叠加显示：

- **顶部居中：** 速度和 MPH 标签（浅灰色卡片）
- **卡片下方：** 日期和时间（12 小时制，含 AM/PM）
- **左上角：** 城市和道路名称（通过逆地理编码）

![tesla overlay](docs/tesla-overlay.png)

要求：

- Tesla 固件 2025.44.25 或更高版本，HW3+
- SEI 元数据仅在行驶画面中存在（不在驻车/Sentry 模式录像中）
- 逆地理编码使用 [OpenStreetMap 的 Nominatim API](https://nominatim.openstreetmap.org/)，通过 geopy 实现（可选）

安装 Tesla 叠加支持：

```bash
uv tool install ".[tesla]"
```

未安装 geopy 时叠加仍可工作，但省略城市/道路名称。

来源：[teslamotors/dashcam](https://github.com/teslamotors/dashcam)

### 管理索引

```bash
# 显示索引信息（标记 [missing] 的文件在磁盘上已不存在）
sentrysearch stats

# 按路径子串移除指定文件
sentrysearch remove path/to/footage

# 清空整个索引
sentrysearch reset
```

三个命令均支持 `--segmentation chunk|shot` 指定目标索引模式。

### 详细模式

在命令后添加 `--verbose` 可查看调试信息（嵌入维度、API 响应时间、相似度分数）。

## 这如何实现？

所有支持的后端——Gemini Embedding 2、Doubao ARK、Qwen VL 和本地 Qwen3-VL-Embedding——都能原生嵌入视频：原始视频像素被投影到与文本查询相同的向量空间中。无需转录、无需帧描述、没有文本中间人。"红灯处停着的红色卡车"这样的文本查询可以直接在向量层面与 30 秒的视频片段进行比较。正是这一点使得对数小时素材的亚秒级语义搜索成为可能。

## 费用

使用 Gemini Embedding API 索引 1 小时视频素材约花费 $2.84（默认设置：30 秒片段，5 秒重叠）：

> 1 小时 = 3,600 秒视频 = 模型处理 3,600 帧。
> 3,600 帧 × $0.00079 = 约 $2.84/小时

Gemini API 从上传的视频中原生提取并编码每秒正好 1 帧，与视频实际帧率无关。预处理步骤（通过 ffmpeg 将片段降至 480p 5fps）是本地/带宽优化——保持请求体较小，使 API 请求快速且不超时——但不改变 API 处理的帧数。

两项内置优化以不同方式降低费用：

- **预处理**（默认开启）——上传前将片段降至 480p 5fps。由于 API 无论如何以 1fps 处理，这仅减少上传大小和传输时间，不影响计费帧数。主要提升速度并防止请求超时。
- **静帧跳过**（默认开启）——完全跳过无视觉变化的片段（如停放的车辆）。这节省实际的 API 调用，直接降低费用。节省量取决于素材——包含数小时空闲时间的 Sentry 模式录像受益最大，而全程行驶的画面可能没有可跳过的内容。

搜索查询的费用可忽略（仅文本嵌入）。

调优选项：

- `--chunk-duration` / `--overlap` — 更长的片段加更少的重叠 = 更少的 API 调用 = 更低费用
- `--no-skip-still` — 嵌入每个片段，即使画面无变化
- `--target-resolution` / `--target-fps` — 调整预处理质量
- `--no-preprocess` — 发送原始片段到 API

## 已知警告（无害）

本地后端在索引和搜索时可能输出警告。这些是外观问题，不影响结果：

- **`MPS: nonzero op is not natively supported`** — Apple Silicon 上的已知 PyTorch 限制。该操作回退到 CPU 执行一步；其余仍在 GPU 上。不影响输出质量。
- **`video_reader_backend torchcodec error, use torchvision as default`** — torchcodec 在 macOS 上找不到兼容的 FFmpeg。视频处理器自动回退到 torchvision。这是预期行为，结果相同。
- **`You are sending unauthenticated requests to the HF Hub`** — 模型从 Hugging Face 下载时未使用令牌。下载速度可能略低，但模型可正常加载。设置 `HF_TOKEN` 环境变量可消除此提示。

## 限制与未来计划

- **静帧检测仍是启发式的** — 当前使用 ffmpeg `mpdecimate` 统计去重后的保留帧比例，能够识别“同一张图重复播放成一段视频”的情况，但在极轻微运动、字幕闪烁或压缩噪声较大的片段上仍可能误判。如需索引每个片段，请使用 `--no-skip-still`。
- **搜索质量取决于片段边界** — 如果一个事件跨越两个片段，重叠窗口有帮助但不完美。更智能的分块（如场景检测）可以改善此问题。
- **Gemini Embedding 2 处于预览阶段** — API 行为和定价可能变更。

## 兼容性

支持 `.mp4` 和 `.mov` 格式的视频素材，不仅限于 Tesla Sentry 模式。目录扫描器会递归查找这两种文件类型，不受文件夹结构影响。

## 系统要求

- Python 3.11+
- `ffmpeg` 在 PATH 中，或使用内置 ffmpeg（通过 `imageio-ffmpeg`，默认安装）
- **Gemini 后端：** Gemini API Key（[免费获取](https://aistudio.google.com/apikey)）
- **Doubao 后端：** ARK API Key（[获取地址](https://console.volcengine.com/ark)）
- **Qwen 后端：** DashScope API Key（[获取地址](https://bailian.console.aliyun.com/)）
- **本地后端：**
  - 支持 CUDA 或 Apple Metal 的 GPU（VRAM/RAM 要求见[硬件表](#本地后端无需-api-key)）
  - **macOS：** `brew install ffmpeg`（视频解码器需要）
  - **Linux/Windows：** 无额外系统依赖
