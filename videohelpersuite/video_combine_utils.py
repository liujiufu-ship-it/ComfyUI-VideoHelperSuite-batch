"""
video_combine_utils.py
视频合成相关的公共函数，抽离自 nodes.py 以避免循环导入。
nodes.py 和 save_video_nodes.py 都从这里导入。
"""

import os
import sys
import json
import subprocess
import numpy as np
from string import Template

import folder_paths
from .logger import logger
from .utils import gifski_path, ENCODE_ARGS, cached


# ---------------------------------------------------------------------------
# 视频格式 JSON 目录
# ---------------------------------------------------------------------------
base_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats")


# ---------------------------------------------------------------------------
# iterate_format（从 nodes.py 原样搬运）
# ---------------------------------------------------------------------------
def flatten_list(l):
    ret = []
    for e in l:
        if isinstance(e, list):
            ret.extend(e)
        else:
            ret.append(e)
    return ret


def iterate_format(video_format, for_widgets=True):
    """Provides an iterator over widgets, or arguments"""
    def indirector(cont, index):
        if isinstance(cont[index], list) and (not for_widgets
          or len(cont[index]) > 1 and not isinstance(cont[index][1], dict)):
            inp = yield cont[index]
            if inp is not None:
                cont[index] = inp
                yield
    for k in video_format:
        if k == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif k.endswith("_pass"):
            for i in range(len(video_format[k])):
                yield from indirector(video_format[k], i)
            if not for_widgets:
                video_format[k] = flatten_list(video_format[k])
        else:
            yield from indirector(video_format, k)


# ---------------------------------------------------------------------------
# 格式加载与应用
# ---------------------------------------------------------------------------
@cached(5)
def get_video_formats():
    format_files = {}
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_files[format_name] = folder_paths.get_full_path("VHS_video_formats", format_name)
    for item in os.scandir(base_formats_dir):
        if not item.is_file() or not item.name.endswith('.json'):
            continue
        format_files[item.name[:-5]] = item.path
    formats = []
    format_widgets = {}
    for format_name, path in format_files.items():
        with open(path, 'r') as stream:
            video_format = json.load(stream)
        if "gifski_pass" in video_format and gifski_path is None:
            continue
        widgets = list(iterate_format(video_format))
        formats.append("video/" + format_name)
        if len(widgets) > 0:
            format_widgets["video/" + format_name] = widgets
    return formats, format_widgets


def apply_format_widgets(format_name, kwargs):
    if os.path.exists(os.path.join(base_formats_dir, format_name + ".json")):
        video_format_path = os.path.join(base_formats_dir, format_name + ".json")
    else:
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name)
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in iterate_format(video_format):
        if w[0] not in kwargs:
            if len(w) > 2 and 'default' in w[2]:
                default = w[2]['default']
            else:
                if type(w[1]) is list:
                    default = w[1][0]
                else:
                    default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[1]]
            kwargs[w[0]] = default
            logger.warn(f"Missing input for {w[0]} has been set to {default}")
    wit = iterate_format(video_format, False)
    for w in wit:
        while isinstance(w, list):
            if len(w) == 1:
                w = [Template(x).substitute(**kwargs) for x in w[0]]
                break
            elif isinstance(w[1], dict):
                w = w[1][str(kwargs[w[0]])]
            elif len(w) > 3:
                w = Template(w[3]).substitute(val=kwargs[w[0]])
            else:
                w = str(kwargs[w[0]])
        wit.send(w)
    return video_format


# ---------------------------------------------------------------------------
# Tensor → bytes
# ---------------------------------------------------------------------------
def tensor_to_int(tensor, bits):
    tensor = tensor.cpu().numpy() * (2 ** bits - 1) + 0.5
    return np.clip(tensor, 0, (2 ** bits - 1))


def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)


def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)


# ---------------------------------------------------------------------------
# FFmpeg / gifski 进程生成器
# ---------------------------------------------------------------------------
def ffmpeg_process(args, video_format, video_metadata, file_path, env):
    res = None
    frame_data = yield
    total_frames_output = 0
    if video_format.get('save_metadata', 'False') != 'False':
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")

        def escape_ffmpeg_metadata(key, value):
            value = str(value)
            value = value.replace("\\", "\\\\")
            value = value.replace(";", "\\;")
            value = value.replace("#", "\\#")
            value = value.replace("=", "\\=")
            value = value.replace("\n", "\\\n")
            return f"{key}={value}"

        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            if "prompt" in video_metadata:
                f.write(escape_ffmpeg_metadata("prompt", json.dumps(video_metadata["prompt"])) + "\n")
            if "workflow" in video_metadata:
                f.write(escape_ffmpeg_metadata("workflow", json.dumps(video_metadata["workflow"])) + "\n")
            for k, v in video_metadata.items():
                if k not in ["prompt", "workflow"]:
                    f.write(escape_ffmpeg_metadata(k, json.dumps(v)) + "\n")

        m_args = args[:1] + ["-i", metadata_path] + args[1:] + [
            "-metadata", "creation_time=now", "-movflags", "use_metadata_tags"
        ]
        with subprocess.Popen(m_args + ["-y", file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output += 1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError:
                err = proc.stderr.read()
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n"
                                    + err.decode(*ENCODE_ARGS))
                print(err.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                logger.warn("An error occurred when saving with metadata")

    if res != b'':
        with subprocess.Popen(args + ["-y", file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output += 1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n"
                                + res.decode(*ENCODE_ARGS))
    yield total_frames_output
    if len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)


def gifski_process(args, dimensions, frame_rate, video_format, file_path, env):
    frame_data = yield
    with subprocess.Popen(
        args + video_format['main_pass'] + ['-f', 'yuv4mpegpipe', '-'],
        stderr=subprocess.PIPE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env
    ) as procff:
        with subprocess.Popen(
            [gifski_path] + video_format['gifski_pass']
            + ['-W', f'{dimensions[0]}', '-H', f'{dimensions[1]}']
            + ['-r', f'{frame_rate}']
            + ['-q', '-o', file_path, '-'],
            stderr=subprocess.PIPE, stdin=procff.stdout,
            stdout=subprocess.PIPE, env=env
        ) as procgs:
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)
                    frame_data = yield
                procff.stdin.flush()
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                outgs = procgs.stdout.read()
            except BrokenPipeError:
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                raise Exception(
                    "An error occurred while creating gifski output\n"
                    "Make sure you are using gifski --version >=1.32.0\nffmpeg: "
                    + resff.decode(*ENCODE_ARGS) + '\ngifski: ' + resgs.decode(*ENCODE_ARGS)
                )
    if len(resff) > 0:
        print(resff.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if len(resgs) > 0:
        print(resgs.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if len(outgs) > 0:
        print(outgs.decode(*ENCODE_ARGS))


# ---------------------------------------------------------------------------
# 乒乓
# ---------------------------------------------------------------------------
def to_pingpong(inp):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp) - 2, 0, -1):
        yield inp[i]
