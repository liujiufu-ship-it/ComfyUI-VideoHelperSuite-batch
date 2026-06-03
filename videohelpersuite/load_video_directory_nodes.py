"""
VHS_LoadVideosFromDirectory
批量遍历目录下的多个视频，逐个 requeue 处理，避免显存叠加。
"""

import os
import server

import folder_paths
from .logger import logger
from .utils import (
    BIGMAX, DIMMAX, calculate_file_hash, get_sorted_dir_files_from_directory,
    lazy_get_audio, hash_path, validate_path, strip_path,
    is_safe_path, floatOrInt, imageOrLatent, ENCODE_ARGS,
)
from .load_video_nodes import (
    load_video, get_load_formats, resized_cv_frame_gen,
    video_extensions,
)

# ---------------------------------------------------------------------------
# 全局状态：用 node unique_id 做 key，存放目录扫描结果和当前进度
# ---------------------------------------------------------------------------
_dir_video_state: dict[str, dict] = {}


def _get_video_files(directory: str,
                     skip_first_videos: int = 0,
                     video_load_cap: int = 0,
                     select_every_nth: int = 1) -> list[str]:
    """扫描目录，返回符合条件的视频文件列表（绝对路径）。"""
    ext_set = {"." + e for e in video_extensions}
    files = get_sorted_dir_files_from_directory(
        directory,
        skip_first_images=skip_first_videos,
        select_every_nth=select_every_nth,
        extensions=ext_set,
    )
    if video_load_cap > 0:
        files = files[:video_load_cap]
    return files


# ---------------------------------------------------------------------------
# 节点类
# ---------------------------------------------------------------------------

class LoadVideosFromDirectoryPath:
    """
    遍历目录下所有视频，每次 workflow 执行处理一个，
    处理完后自动 requeue 下一个，直到全部完成。
    显存峰值 = 单个视频峰值，不会随视频数量增加。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": (
                    "STRING",
                    {
                        "placeholder": "X:/path/to/videos",
                        "vhs_path_extensions": [],
                    },
                ),
                "skip_first_videos": (
                    "INT",
                    {"default": 0, "min": 0, "max": BIGMAX, "step": 1},
                ),
                "video_load_cap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": BIGMAX,
                        "step": 1,
                        "tooltip": "最多处理几个视频，0 表示全部",
                    },
                ),
                "select_every_nth_video": (
                    "INT",
                    {"default": 1, "min": 1, "max": BIGMAX, "step": 1},
                ),
                # ---- 以下与 LoadVideoPath 相同的帧加载参数 ----
                "force_rate": (
                    floatOrInt,
                    {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0},
                ),
                "custom_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": DIMMAX, "disable": 0},
                ),
                "custom_height": (
                    "INT",
                    {"default": 0, "min": 0, "max": DIMMAX, "disable": 0},
                ),
                "frame_load_cap": (
                    "INT",
                    {"default": 0, "min": 0, "max": BIGMAX, "step": 1, "disable": 0},
                ),
                "skip_first_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": BIGMAX, "step": 1},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": BIGMAX, "step": 1},
                ),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": get_load_formats(),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = (imageOrLatent, "INT", "AUDIO", "VHS_VIDEOINFO",
                    "INT", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info",
                    "current_index", "total_count", "current_filename")

    FUNCTION = "load_video_from_dir"

    # ------------------------------------------------------------------

    def load_video_from_dir(
        self,
        directory: str,
        skip_first_videos: int = 0,
        video_load_cap: int = 0,
        select_every_nth_video: int = 1,
        prompt=None,
        unique_id=None,
        **kwargs,
    ):
        directory = strip_path(directory)
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")

        # ---- 从 prompt 读取 requeue 计数 ----
        requeue = 0
        if unique_id is not None and prompt is not None:
            requeue = prompt[unique_id]["inputs"].get("requeue", 0)

        # ---- requeue==0：重新扫描目录，初始化状态 ----
        if requeue == 0 or unique_id not in _dir_video_state:
            files = _get_video_files(
                directory, skip_first_videos, video_load_cap, select_every_nth_video
            )
            if len(files) == 0:
                raise FileNotFoundError(f"目录 '{directory}' 下没有找到视频文件")
            _dir_video_state[unique_id] = {
                "files": files,
                "total": len(files),
            }
            logger.info(
                f"[LoadVideosFromDir] 扫描到 {len(files)} 个视频，开始处理第 1 个"
            )

        state = _dir_video_state[unique_id]
        files: list[str] = state["files"]
        total: int = state["total"]

        # current_index 就是 requeue 值（0-based）
        current_index: int = requeue
        if current_index >= total:
            # 理论上不应走到这，保险起见
            raise RuntimeError("所有视频已处理完毕，不应再次执行")

        current_file = files[current_index]
        current_filename = os.path.basename(current_file)
        logger.info(
            f"[LoadVideosFromDir] 处理 [{current_index + 1}/{total}]: {current_filename}"
        )

        # ---- 调用底层 load_video ----
        kwargs["video"] = current_file
        result = load_video(generator=resized_cv_frame_gen, **kwargs)
        # result = (images, frame_count, audio, video_info)

        # ---- 判断是否还有下一个，决定是否 requeue ----
        has_next = (current_index + 1) < total

        if has_next:
            self._requeue(unique_id, prompt, current_index + 1)
        else:
            # 全部完成，清理状态
            logger.info("[LoadVideosFromDir] 所有视频处理完毕")
            _dir_video_state.pop(unique_id, None)

        return (*result, current_index, total, current_filename)

    # ------------------------------------------------------------------

    @staticmethod
    def _requeue(unique_id: str, prompt: dict, next_index: int):
        """将下一次执行加入队列，并把 requeue 计数写入 prompt。"""
        prompt_queue = server.PromptServer.instance.prompt_queue
        assert len(prompt_queue.currently_running) == 1

        value = next(iter(prompt_queue.currently_running.values()))
        if len(value) == 6:
            (number, prompt_id, cur_prompt, extra_data, outputs_to_execute, sensitive) = value
        else:
            (number, prompt_id, cur_prompt, extra_data, outputs_to_execute) = value
            sensitive = {}

        new_prompt = cur_prompt.copy()
        # 写入 requeue 计数，让下次执行知道从哪里继续
        new_prompt[unique_id] = new_prompt[unique_id].copy()
        new_prompt[unique_id]["inputs"] = new_prompt[unique_id]["inputs"].copy()
        new_prompt[unique_id]["inputs"]["requeue"] = next_index

        import uuid as _uuid
        new_number = -server.PromptServer.instance.number
        server.PromptServer.instance.number += 1
        new_prompt_id = str(_uuid.uuid4())
        prompt_queue.put(
            (new_number, new_prompt_id, new_prompt, extra_data, outputs_to_execute, sensitive)
        )
        logger.info(f"[LoadVideosFromDir] 已加入队列：下一个 index={next_index}")

    # ------------------------------------------------------------------

    @classmethod
    def IS_CHANGED(cls, directory: str, **kwargs):
        """目录内容变化时触发重新执行。"""
        directory = strip_path(directory)
        if not os.path.isdir(directory):
            return ""
        # 对目录下所有视频文件做哈希，任何文件变动都会触发
        ext_set = {"." + e for e in video_extensions}
        files = get_sorted_dir_files_from_directory(directory, extensions=ext_set)
        h = "|".join(calculate_file_hash(f) for f in files)
        return h

    @classmethod
    def VALIDATE_INPUTS(cls, directory: str, **kwargs):
        directory = strip_path(directory)
        if directory is None:
            return "directory 不能为空"
        if not os.path.isdir(directory):
            return f"目录不存在: '{directory}'"
        return True
