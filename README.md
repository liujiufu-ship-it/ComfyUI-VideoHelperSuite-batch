# ComfyUI-VideoHelperSuite (Batch Edition)

> **Fork of [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)**  
> 在原版基础上新增了**批量视频目录处理**和**保留原始文件名保存**两个节点，专为批量视频工作流设计。

---

## 新增节点

### Load Videos From Directory (Path) 🎥🅥🅗🅢

**节点 ID：** `VHS_LoadVideosFromDirectoryPath`  
**分类：** `Video Helper Suite 🎥🅥🅗🅢`

遍历指定目录下的所有视频文件，每次 workflow 执行处理一个视频，处理完后自动 requeue 下一个，直到全部完成。

**显存优势：** 每次只有一个视频的帧在内存中，峰值显存恒定，不随视频数量叠加，适合大批量、长视频处理。

#### 输入参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `directory` | STRING | 视频目录路径（绝对路径）|
| `skip_first_videos` | INT | 跳过前 N 个视频 |
| `video_load_cap` | INT | 最多处理几个视频，`0` 表示全部 |
| `select_every_nth_video` | INT | 每 N 个视频取 1 个 |
| `force_rate` | FLOAT/INT | 强制帧率，`0` 禁用 |
| `custom_width` / `custom_height` | INT | 自定义分辨率，`0` 保持原始 |
| `frame_load_cap` | INT | 每个视频最多加载的帧数 |
| `skip_first_frames` | INT | 每个视频跳过前 N 帧 |
| `select_every_nth` | INT | 每 N 帧取 1 帧 |
| `meta_batch` | VHS_BatchManager | 可选，批次管理器 |
| `vae` | VAE | 可选，直接输出 Latent |
| `format` | COMBO | 加载格式预设（AnimateDiff / LTXV / Wan 等）|

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `IMAGE` | IMAGE / LATENT | 当前视频帧序列 |
| `frame_count` | INT | 当前视频帧数 |
| `audio` | AUDIO | 当前视频音频 |
| `video_info` | VHS_VIDEOINFO | 当前视频元信息 |
| `current_index` | INT | 当前是第几个视频（0-based）|
| `total_count` | INT | 目录下共几个视频 |
| `current_filename` | STRING | 当前视频文件名（含扩展名，用于命名输出）|

#### 工作原理

```
第 1 次执行：扫描目录 → 加载 video[0] → 处理 → 输出 → 自动 requeue
第 2 次执行：加载 video[1] → 处理 → 输出 → 自动 requeue
...
第 N 次执行：加载 video[N-1] → 处理 → 输出 → 完成，停止
```

---

### Save Video (Keep Filename) 🎥🅥🅗🅢

**节点 ID：** `VHS_SaveVideoWithFilename`  
**分类：** `Video Helper Suite 🎥🅥🅗🅢`

以指定文件名保存视频，不附加计数器后缀。配合 `LoadVideosFromDirectory` 的 `current_filename` 输出，可实现"输入叫什么、输出就叫什么"。

#### 输入参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `images` | IMAGE / LATENT | 视频帧 |
| `frame_rate` | FLOAT/INT | 输出帧率 |
| `loop_count` | INT | 循环次数，`0` 不循环 |
| `filename_stem` | STRING | **输出文件名（不含扩展名）**，直接连 `current_filename` |
| `subfolder` | STRING | 输出子目录（相对 ComfyUI output 目录），留空放根目录 |
| `format` | COMBO | 输出格式，同 VideoCombine（gif / webp / mp4 / webm 等）|
| `pingpong` | BOOLEAN | 乒乓循环 |
| `save_output` | BOOLEAN | `true`=output 目录 / `false`=temp 目录 |
| `overwrite` | BOOLEAN | 是否覆盖同名文件，关闭时自动追加 `_001/_002…` |
| `audio` | AUDIO | 可选，混流音频 |
| `vae` | VAE | 可选，VAE decode |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `Filenames` | VHS_FILENAMES | 所有生成文件路径列表 |

---

### 典型批量处理 Workflow

```
VHS_LoadVideosFromDirectoryPath
├── IMAGE ──────────────────────────────► 处理节点（AI 推理等）
│                                               │
│                                               ▼ IMAGE
└── current_filename ──────────────► VHS_SaveVideoWithFilename
                                        ← filename_stem
                                        subfolder: "processed"
                                        format: video/h264-mp4
```

- 输入 `clip_000.mp4` → 输出 `output/processed/clip_000.mp4`  
- 输入 `clip_001.mp4` → 输出 `output/processed/clip_001.mp4`  
- 全部完成后自动停止，无需人工干预

---

## 原版节点

### Load Video (Upload / Path)

将视频文件转换为帧序列。

- `force_rate`：强制帧率，设为 `0` 禁用。可用于快速匹配 AnimateDiff 的 8fps。
- `force_size`：快速缩放到建议尺寸，部分选项可只设宽度或高度并由长宽比推算另一值。
- `frame_load_cap`：返回帧数上限，即最大 batch size。
- `skip_first_frames`：从视频开头（已按强制帧率调整后）跳过的帧数。配合 `frame_load_cap` 可分段处理长视频。
- `select_every_nth`：每 N 帧取 1 帧，不考虑基准帧率，不会造成帧复制，适合处理 animated gif。

Path 版本支持从外部路径加载视频。

### Load Image Sequence (Upload / Path)

从子目录加载所有图片文件，选项与 Load Video 类似。

- `image_load_cap`：返回图片数量上限。
- `skip_first_images`：跳过开头的图片数量。
- `select_every_nth`：每 N 张取 1 张。

### Video Combine

将一系列图像合成输出视频，可选混入音频。

- `frame_rate`：每秒显示的输入帧数。
- `loop_count`：额外重复次数。
- `filename_prefix`：输出文件名前缀，支持子目录和时间戳格式。
- `format`：文件格式，详见 [Video Formats](#video-formats)。
- `pingpong`：反向播放一次以创建无缝循环。
- `save_output`：输出到 output 目录还是 temp 目录。

返回 `VHS_FILENAMES`，包含 save_output 状态和所有生成文件的完整路径列表。

### Load Audio

加载独立音频文件。

- `seek_seconds`：音频起始时间（秒）。

---

## Latent / Image 工具节点

以下节点均有 Latent、Image、Mask 三个等价版本：

| 节点 | 功能 |
|------|------|
| Split Batch | 按 `split_index` 将输入分为 A（前段）和 B（后段）|
| Merge Batch | 将 A 和 B 合并为一个序列，支持不同尺寸时的缩放策略 |
| Select Every Nth | 每 N 个取第 1 个，其余丢弃 |
| Get Count | 获取当前批次数量 |
| Duplicate Batch | 重复复制批次 |
| Select (by index list) | 按索引列表选取 |

---

## 视频预览

以下节点支持动画预览：Load Video (Upload/Path)、Load Images (Upload/Path)、Video Combine、Load Videos From Directory (Path)。

右键预览区域可：
- 在浏览器中打开原文件
- 保存预览
- 暂停 / 隐藏 / 同步预览

### 高级预览

在设置中手动开启 **VHS Advanced Previews**，启用后 Load Video 节点的预览将实时反映 `skip_first_frames`、`frame_load_cap` 等参数设置，方便精确截取视频片段。

---

## Video Formats

熟悉 ffmpeg 的用户可在 `video_formats` 目录下添加 JSON 文件，为 Video Combine 增加新的输出格式。

```json
{
    "main_pass": [
        "-c:v", "libsvtav1",
        "-pix_fmt", "yuv420p10le",
        "-crf", ["crf", "INT", {"default": 23, "min": 0, "max": 100, "step": 1}]
    ],
    "audio_pass": ["-c:a", "libopus"],
    "extension": "webm",
    "environment": {"SVT_LOG": "1"}
}
```

- `main_pass`：传给 ffmpeg 的参数列表，列表项可以是字符串或暴露为 widget 的三元组 `[name, type, options]`。
- `audio_pass`：有音频输入时传给 ffmpeg 的参数。
- `extension`：文件扩展名（同时决定容器格式）。
- `environment`：可选，执行时附加的环境变量。
- `input_color_depth`：像素传输位深，`8bit`（默认）或 `16bit`（实验性，质量更高）。
- `save_metadata`：设为 `True` 时将 workflow 嵌入输出视频元数据，支持拖拽加载。

---

## 致谢

本项目基于 [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) 开发，感谢原作者的出色工作。
