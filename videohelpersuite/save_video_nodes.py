"""
VHS_SaveVideoWithFilename
保存视频时以指定文件名（通常来自输入视频的原始文件名）作为输出文件名，
不附加计数器后缀，保证"输入叫啥输出就叫啥"。
"""

import os
import sys
import json
import datetime
import subprocess
import re
import itertools

import numpy as np
import torch
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo

import folder_paths
from .logger import logger
from .video_combine_utils import (
    get_video_formats,
    apply_format_widgets,
    tensor_to_bytes,
    tensor_to_shorts,
    ffmpeg_process,
    gifski_process,
    to_pingpong,
)
from .utils import (
    ffmpeg_path, gifski_path, get_audio,
    imageOrLatent, BIGMAX, merge_filter_args,
    ENCODE_ARGS, floatOrInt, ContainsAll,
)
from comfy.utils import ProgressBar


class SaveVideoWithFilename:
    """
    以指定文件名保存视频，输出文件名 = output_directory / filename_stem + 格式扩展名。
    不附加计数器，适合批量处理时保持原始文件名。
    若输出文件已存在，默认覆盖（可通过 overwrite 开关控制）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True}]]
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (
                    floatOrInt,
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                # 文件名主体（不含扩展名），通常接 LoadVideosFromDirectory 的 current_filename
                "filename_stem": (
                    "STRING",
                    {
                        "default": "output",
                        "tooltip": "输出文件名（不含扩展名），直接使用输入视频文件名即可",
                    },
                ),
                # 输出子目录（相对于 ComfyUI output 目录），留空则直接放 output 根目录
                "subfolder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "输出子目录，留空则放 output 根目录",
                    },
                ),
                "format": (
                    ["image/gif", "image/webp"] + ffmpeg_formats,
                    {'formats': format_widgets},
                ),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "若目标文件已存在是否覆盖，关闭时会自动追加 _001 / _002 …",
                    },
                ),
            },
            "optional": {
                "audio": ("AUDIO",),
                "vae": ("VAE",),
            },
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }),
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "save_video"

    # ------------------------------------------------------------------

    def save_video(
        self,
        images=None,
        latents=None,
        frame_rate: int = 8,
        loop_count: int = 0,
        filename_stem: str = "output",
        subfolder: str = "",
        format: str = "image/gif",
        pingpong: bool = False,
        save_output: bool = True,
        overwrite: bool = True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        vae=None,
        **kwargs,
    ):
        if latents is not None:
            images = latents
        if images is None:
            return ((save_output, []),)
        if vae is not None:
            if isinstance(images, dict):
                images = images['samples']
            else:
                vae = None

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)

        # ---- 清理文件名：去掉原始扩展名，只保留 stem ----
        stem = os.path.splitext(os.path.basename(filename_stem))[0]
        # 去掉不安全字符（保留中文、字母、数字、下划线、横线、点）
        stem = re.sub(r'[\\/:*?"<>|]', '_', stem)
        if not stem:
            stem = "output"

        # ---- 确定输出目录 ----
        base_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        if subfolder:
            full_output_folder = os.path.join(base_dir, subfolder)
        else:
            full_output_folder = base_dir
        os.makedirs(full_output_folder, exist_ok=True)

        # ---- 元数据 ----
        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            extra_options = extra_pnginfo.get('workflow', {}).get('extra', {})
        else:
            extra_options = {}
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        num_frames = len(images)
        pbar = ProgressBar(num_frames)

        # ---- VAE decode ----
        if vae is not None:
            downscale_ratio = getattr(vae, "downscale_ratio", 8)
            width = images.size(-1) * downscale_ratio
            height = images.size(-2) * downscale_ratio
            frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1

            def _batched(it, n):
                while batch := tuple(itertools.islice(it, n)):
                    yield batch

            def _batched_encode(imgs, v, fpb):
                for batch in _batched(iter(imgs), fpb):
                    yield from v.decode(torch.from_numpy(np.array(batch)))

            images = _batched_encode(images, vae, frames_per_batch)
            first_image = next(images)
            images = itertools.chain([first_image], images)
            while len(first_image.shape) > 3:
                first_image = first_image[0]
        else:
            first_image = images[0]
            images = iter(images)

        output_files = []

        # ---- 保存元数据预览图（与 VideoCombine 保持一致） ----
        preview_png = f"{stem}.png"
        preview_png_path = os.path.join(full_output_folder, preview_png)
        if extra_options.get('VHS_MetadataImage', True) is not False:
            Image.fromarray(tensor_to_bytes(first_image)).save(
                preview_png_path, pnginfo=metadata, compress_level=4,
            )
        output_files.append(preview_png_path)

        # ---- 格式分支 ----
        format_type, format_ext = format.split("/")

        if format_type == "image":
            # Pillow 直接保存 gif / webp
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
                image_kwargs['lossless'] = kwargs.get("lossless", True)

            file = f"{stem}.{format_ext}"
            file_path = self._resolve_path(full_output_folder, file, overwrite)

            if pingpong:
                images = to_pingpong(images)

            def _frames_gen(imgs):
                for img in imgs:
                    pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(img))

            frames = _frames_gen(images)
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs,
            )
            output_files.append(file_path)

        else:
            # FFmpeg / gifski
            if ffmpeg_path is None:
                raise ProcessLookupError(
                    "ffmpeg is required for video outputs and could not be found."
                )

            has_alpha = first_image.shape[-1] == 4
            kwargs["has_alpha"] = has_alpha
            video_format = apply_format_widgets(format_ext, kwargs)
            dim_alignment = video_format.get("dim_alignment", 2)

            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                to_pad = (
                    -first_image.shape[1] % dim_alignment,
                    -first_image.shape[0] % dim_alignment,
                )
                padding = (
                    to_pad[0] // 2, to_pad[0] - to_pad[0] // 2,
                    to_pad[1] // 2, to_pad[1] - to_pad[1] // 2,
                )
                padfunc = torch.nn.ReplicationPad2d(padding)

                def _pad(image):
                    image = image.permute((2, 0, 1))
                    return padfunc(image.to(dtype=torch.float32)).permute((1, 2, 0))

                images = map(_pad, images)
                dimensions = (
                    -first_image.shape[1] % dim_alignment + first_image.shape[1],
                    -first_image.shape[0] % dim_alignment + first_image.shape[0],
                )
                logger.warn("Output images were padded to match dim_alignment")
            else:
                dimensions = (first_image.shape[1], first_image.shape[0])

            if pingpong:
                images = to_pingpong(images)
                if num_frames > 2:
                    num_frames += num_frames - 2
                    pbar.total = num_frames

            loop_args = (
                ["-vf", f"loop=loop={loop_count}:size={num_frames}"]
                if loop_count > 0
                else []
            )

            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                i_pix_fmt = 'rgba64' if has_alpha else 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                i_pix_fmt = 'rgba' if has_alpha else 'rgb24'

            file = f"{stem}.{video_format['extension']}"
            file_path = self._resolve_path(full_output_folder, file, overwrite)

            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                unit = "M" if video_format.get('megabit') == 'True' else "K"
                bitrate_arg = ["-b:v", f"{bitrate}{unit}"]

            args = [
                ffmpeg_path, "-v", "error",
                "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                "-color_range", "pc", "-colorspace", "rgb",
                "-color_primaries", "bt709",
                "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
                "-s", f"{dimensions[0]}x{dimensions[1]}",
                "-r", str(frame_rate), "-i", "-",
            ] + loop_args

            images = map(lambda x: x.tobytes(), images)
            env = os.environ.copy()
            if "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                in_args_len = args.index("-i") + 2
                pre_pass_args = args[:in_args_len] + video_format['pre_pass']
                merge_filter_args(pre_pass_args)
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n"
                                    + e.stderr.decode(*ENCODE_ARGS))

            if "inputs_main_pass" in video_format:
                in_args_len = args.index("-i") + 2
                args = args[:in_args_len] + video_format['inputs_main_pass'] + args[in_args_len:]

            if 'gifski_pass' in video_format:
                fmt = 'image/gif'
                output_process = gifski_process(
                    args, dimensions, frame_rate, video_format, file_path, env
                )
                audio = None
            else:
                args += video_format['main_pass'] + bitrate_arg
                merge_filter_args(args)
                output_process = ffmpeg_process(
                    args, video_format, video_metadata, file_path, env
                )

            output_process.send(None)  # 启动生成器

            total_frames_output = 0
            if isinstance(images, list):
                # pre_pass 已消耗，images 是 bytes list
                for img in images:
                    pbar.update(1)
                    output_process.send(img)
            else:
                for img in images:
                    pbar.update(1)
                    output_process.send(img)

            try:
                total_frames_output = output_process.send(None)
                output_process.send(None)
            except StopIteration:
                pass

            output_files.append(file_path)

            # ---- 混流音频 ----
            a_waveform = None
            if audio is not None:
                try:
                    a_waveform = audio['waveform']
                except Exception:
                    pass

            if a_waveform is not None:
                audio_out_file = f"{stem}-audio.{video_format['extension']}"
                audio_out_path = self._resolve_path(full_output_folder, audio_out_file, overwrite)

                if "audio_pass" not in video_format:
                    logger.warn("Selected video format has no explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]

                channels = audio['waveform'].size(1)
                min_audio_dur = total_frames_output / frame_rate + 1
                apad = (
                    []
                    if video_format.get('trim_to_audio', 'False') != 'False'
                    else ["-af", f"apad=whole_dur={min_audio_dur}"]
                )
                mux_args = [
                    ffmpeg_path, "-v", "error", "-y",
                    "-i", file_path,
                    "-ar", str(audio['sample_rate']),
                    "-ac", str(channels),
                    "-f", "f32le", "-i", "-",
                    "-c:v", "copy",
                ] + video_format["audio_pass"] + apad + ["-shortest", audio_out_path]

                audio_data = (
                    audio['waveform'].squeeze(0).transpose(0, 1).numpy().tobytes()
                )
                merge_filter_args(mux_args, '-af')
                try:
                    res = subprocess.run(
                        mux_args, input=audio_data, env=env,
                        capture_output=True, check=True,
                    )
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg subprocess:\n"
                                    + e.stderr.decode(*ENCODE_ARGS))
                if res.stderr:
                    print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                output_files.append(audio_out_path)
                file = audio_out_file
                file_path = audio_out_path

        # ---- 清理中间文件 ----
        if extra_options.get('VHS_KeepIntermediate', True) is False:
            for intermediate in output_files[1:-1]:
                if os.path.exists(intermediate):
                    os.remove(intermediate)

        preview = {
            "filename": os.path.basename(file_path),
            "subfolder": subfolder,
            "type": "output" if save_output else "temp",
            "format": format,
            "frame_rate": frame_rate,
            "workflow": os.path.basename(preview_png_path),
            "fullpath": output_files[-1],
        }
        if num_frames == 1 and 'png' in format:
            preview['format'] = 'image/png'

        logger.info(f"[SaveVideoWithFilename] 已保存: {output_files[-1]}")
        return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files),)}

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(folder: str, filename: str, overwrite: bool) -> str:
        """
        若 overwrite=True 直接返回路径（FFmpeg 会覆盖）。
        否则检测文件是否存在，存在则追加 _001 / _002 … 后缀。
        """
        path = os.path.join(folder, filename)
        if overwrite or not os.path.exists(path):
            return path
        stem, ext = os.path.splitext(filename)
        counter = 1
        while True:
            new_path = os.path.join(folder, f"{stem}_{counter:03d}{ext}")
            if not os.path.exists(new_path):
                return new_path
            counter += 1
