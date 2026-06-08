import argparse
import atexit
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from platform import system
from zipfile import ZipFile

import progressbar
from notifier import EmailNotifier, TelegramNotifier
from tqdm import tqdm
from utils import md5sum

ZSTD_BIN = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "zstd.exe" if system().lower() == "windows" else "zstd",
    )
)
MSU_VQMT_PATH = "msu_vqmt"
CMD_METRICS_TO_JSON = {
    "psnr": ("psnr", ""),
    "ssim_precise": ("ssim", "precise"),
    "vmaf": ("vmaf", ""),
    "niqe": ("niqe", ""),
}
JOB_TIMEOUT = 60 * 60
WAIT_MULTIPLIER = 200
TELEGRAM_NOTIFIER = None


def load_proxy_config(proxy_file):
    """load proxy config fro json"""
    if proxy_file is None:
        return None

    proxy_path = Path(proxy_file)
    if not proxy_path.exists():
        return None

    with open(proxy_path, "r") as f:
        config = json.load(f)

    # proxy_url
    protocol = config.get("protocol", "socks5")  # socks5, socks4, http, https
    server = config["server"]
    port = config["port"]
    proxy_url = f"{protocol}://{server}:{port}"

    request_kwargs = {"proxy_url": proxy_url}

    # есть авторизация
    if "username" in config and "password" in config:
        request_kwargs["urllib3_proxy_kwargs"] = {
            "username": config["username"],
            "password": config["password"],
        }

    return request_kwargs


def LoadJsonFromFile(filename):
    f = open(filename, "r")
    obj = json.load(f, strict=False)
    f.close()
    return obj


def DumpJsonToFile(obj, filename):
    f = open(filename, "w")
    json.dump(obj, f, sort_keys=True)
    f.close()


def StarFunc(input_args):
    func, args = input_args
    if isinstance(args, list):
        return func(*args)
    else:
        return func(**args)


def md5hash(txt):
    d = hashlib.md5()
    d.update(txt.encode("utf-8"))
    return d.hexdigest()


def check_wsl_has_linux():
    """Simple check if WSL has any Linux distributions installed"""
    try:
        result = subprocess.run(
            ["wsl", "--list", "--quiet"],
            capture_output=True,
            text=True,
            shell=True,
            encoding="utf-16-le",
        )

        if result.returncode == 0:
            distributions = [
                line.strip().lower()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            return "ubuntu" in distributions
        return False

    except FileNotFoundError:
        return False


def convert_path_to_wsl(windows_path):
    """Конвертирует Windows путь в WSL путь"""
    path = Path(windows_path)
    # Получаем абсолютный путь
    abs_path = path.absolute()

    # Преобразуем в строку и меняем разделители
    path_str = str(abs_path).replace("\\", "/")

    # Обрабатываем букву диска
    if ":" in path_str:
        drive, rest = path_str.split(":", 1)
        return f"/mnt/{drive.lower()}{rest}"

    return path_str


def check_streamlake(streamlake_dir):
    win_input_yuv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "crowd_run_1920x1080_50_1frame.yuv")
    )
    win_output = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "test_encoded")
    )
    run_command = "[ ! -f /opt/intel/openvino_2024.3.0/setupvars.sh ] && (mkdir -p /opt/intel/openvino_2024.3.0 && tar -xzvf l_openvino_toolkit_linux_2024.3.0.16041.1e3b88e4e3f_x86_64.tgz --strip-components=1 -C /opt/intel/openvino_2024.3.0); source /opt/intel/openvino_2024.3.0/setupvars.sh"
    if system().lower() == "windows":
        if not check_wsl_has_linux():
            print("Ubuntu is not installed in WSL")
            return 1
        input_yuv = convert_path_to_wsl(win_input_yuv)
        output = convert_path_to_wsl(win_output)

        run_command += f' && ./slvcEncoder-3.0.8 -i "{input_yuv}" -o "{output}" -s 1920x1080 -r 50 -f 1 --crf 21 --preset exp'
        run_command = [
            "start",
            "/wait",
            "cmd",
            "/c",
            "wsl",
            "-d",
            "Ubuntu",
            "-u",
            "root",
            "--",
            "bash",
            "-c",
            run_command,
        ]
    else:
        run_command += f' && ./slvcEncoder-3.0.8 -i "{win_input_yuv}" -o "{win_output}" -s 1920x1080 -r 50 -f 1 --crf 21 --preset exp'

    if os.path.exists(win_output):
        try:
            os.remove(win_output)
        except:
            pass
    try:
        subprocess.run(run_command, shell=True, cwd=streamlake_dir)
        if not os.path.exists(win_output):
            print("NOT FOUND " + win_output)
            return 2
        md5 = md5sum(Path(win_output)).lower()
        expected_md5 = "151423c03aaff54d71c86a154ace7179"
        if md5 != expected_md5:
            print(
                "MD5 of {} is equal to {}. But should be {}".format(
                    win_output, md5, expected_md5
                )
            )
            return 2
    finally:
        try:
            os.remove(win_output)
        except:
            pass

    return 0


# ---------------------------------------------------------------------------
# Torch-based perceptual metrics (LPIPS, DISTS)
# ---------------------------------------------------------------------------

# Maps CLI name → (pyiqa metric name, extra kwargs, pth path relative to torch_models_path)
_TORCH_METRICS = {
    "lpips_alex":     ("lpips",      {"net": "alex"}, "LPIPS_v0.1_alex-df73285e.pth"),
    "lpips_vgg":      ("lpips",      {"net": "vgg"},  "LPIPS_v0.1_vgg-a78928a0.pth"),
    "lpips_plus":     ("lpips+",     {},               None),
    "lpips_vgg_plus": ("lpips-vgg+", {},               None),
    "dists":          ("dists",      {},               "DISTS_weights-f5e65c96.pth"),
}


def _read_yuv_frame_tensor(f, width, height, pix_fmt):
    """Read one raw YUV frame from an open binary file handle.

    Returns a float32 torch.Tensor of shape [1, 3, H, W] with values in [0, 1]
    after BT.709 full-range YCbCr → RGB conversion.
    """
    import numpy as np
    import torch

    if pix_fmt == "yuv420p":
        y = np.frombuffer(f.read(width * height),      dtype=np.uint8 ).reshape(height,      width)
        u = np.frombuffer(f.read(width * height // 4), dtype=np.uint8 ).reshape(height // 2, width // 2)
        v = np.frombuffer(f.read(width * height // 4), dtype=np.uint8 ).reshape(height // 2, width // 2)
        mv = 255.0
    elif pix_fmt == "yuv420p10le":
        y = np.frombuffer(f.read(width * height * 2),  dtype="<u2").reshape(height,      width)
        u = np.frombuffer(f.read(width * height // 2), dtype="<u2").reshape(height // 2, width // 2)
        v = np.frombuffer(f.read(width * height // 2), dtype="<u2").reshape(height // 2, width // 2)
        mv = 1023.0
    else:
        raise ValueError("Unsupported pix_fmt for torch metrics: {}".format(pix_fmt))

    y  = y.astype(np.float32) / mv
    cb = u.astype(np.float32) / mv - 0.5
    cr = v.astype(np.float32) / mv - 0.5

    # Upsample chroma 4:2:0 → 4:4:4 (nearest-neighbour, matches ffmpeg default)
    cb = np.repeat(np.repeat(cb, 2, axis=0), 2, axis=1)
    cr = np.repeat(np.repeat(cr, 2, axis=0), 2, axis=1)

    # BT.709 full-range YCbCr → RGB
    r = np.clip(y + 1.5748 * cr,               0.0, 1.0)
    g = np.clip(y - 0.1873 * cb - 0.4681 * cr, 0.0, 1.0)
    b = np.clip(y + 1.8556 * cb,               0.0, 1.0)

    return torch.from_numpy(np.stack([r, g, b])).unsqueeze(0)  # [1, 3, H, W]


def _compute_torch_metrics(ref_yuv, dist_yuv, yuv_info, torch_metric_list, torch_models_path):
    """Compute perceptual metrics frame-by-frame on two raw YUV files.

    Returns a dict:
      {
        "lpips_alex":      [per_frame_scores...],
        "lpips_alex_mean": float,
        ...
      }
    """
    import torch
    import pyiqa

    width   = yuv_info["width"]
    height  = yuv_info["height"]
    pix_fmt = yuv_info["pix_fmt"]
    n_frames = yuv_info["length"]

    # Load models
    models = {}
    for name in torch_metric_list:
        pyiqa_name, extra_kwargs, model_file = _TORCH_METRICS[name]
        kwargs = dict(extra_kwargs, as_loss=True, device="cpu")
        if model_file and torch_models_path:
            path = os.path.join(torch_models_path, model_file)
            if os.path.isfile(path):
                kwargs["pretrained_model_path"] = path
        try:
            m = pyiqa.create_metric(pyiqa_name, **kwargs)
            m.eval()
            models[name] = m
        except Exception as e:
            print("[torch_metrics] Cannot load {}: {}".format(name, e))

    if not models:
        return {}

    # Iterate frames
    frame_scores = {n: [] for n in models}
    with open(ref_yuv, "rb") as rf, open(dist_yuv, "rb") as df:
        for _ in range(n_frames):
            rt = _read_yuv_frame_tensor(rf, width, height, pix_fmt)
            dt = _read_yuv_frame_tensor(df, width, height, pix_fmt)
            with torch.no_grad():
                for name, model in models.items():
                    frame_scores[name].append(float(model(rt, dt)))

    result = {}
    for name, scores in frame_scores.items():
        result[name] = scores
        result[name + "_mean"] = sum(scores) / len(scores) if scores else 0.0
    return result


def CalculateMetrics(
    input_yuv,
    yuv_info,
    encoded_filename,
    decoder_cwd,
    decoder_cmd,
    decoder_os,
    decoded_storage,
    base_json_filename,
    result_json_filename,
    metric_list,
    vqmt_threads,
    local_encoded_filename=None,
    torch_metric_list=None,
    torch_models_path=None,
):
    result = False
    decoding_speed_bps = 0.0
    copy_speed_bps = 0.0
    decompress_speed_bps = 0.0
    decoding_speed_fps = 0.0
    _torch_scores = {}
    try:
        # Detect metric results
        if not os.path.isfile(result_json_filename):
            shutil.copy(base_json_filename, result_json_filename)
        try:
            result_json = LoadJsonFromFile(result_json_filename)
        except:
            shutil.copy(base_json_filename, result_json_filename)
            result_json = LoadJsonFromFile(result_json_filename)

        # Skip VQMT re-run if it is already computed and matches requested metrics
        vqmt_already_done = (
            bool(metric_list)
            and "vqmt" in result_json
            and set(metric_list) <= set(result_json.get("metric_list", []))
        )

        if local_encoded_filename is not None:
            shutil.copy(encoded_filename, local_encoded_filename)
            encoded_filename = local_encoded_filename

        gt_encoded_md5 = result_json["launches_data"][0]
        if "encoded_md5" in gt_encoded_md5:
            gt_encoded_md5 = gt_encoded_md5["encoded_md5"][-1]

            md5 = md5sum(Path(encoded_filename))
            if md5 != gt_encoded_md5:
                print(
                    "MD5 of {} is equal to {}. But should be {}".format(
                        encoded_filename, md5, gt_encoded_md5
                    )
                )
                return (
                    result,
                    decoding_speed_bps,
                    copy_speed_bps,
                    decompress_speed_bps,
                    decoding_speed_fps,
                )

        parallel_process = True
        target_metric_list = metric_list
        if decoder_os.lower() == "linux" and not check_wsl_has_linux():
            print("Ubuntu is not installed in WSL")
            return (
                result,
                decoding_speed_bps,
                copy_speed_bps,
                decompress_speed_bps,
                decoding_speed_fps,
            )

        parallel_process = False
        out_yuv = (
            os.path.abspath(
                os.path.join(decoded_storage, os.path.basename(encoded_filename))
            )
            + ".yuv"
        )
        run_command_vqmt = (
            'start /wait cmd /k vqmt -in "{input_yuv}" {pix_fmt} {width}x{height} '
            '-in "{decoded_file}" {pix_fmt} {width}x{height}'.format(
                input_yuv=os.path.join(os.getcwd(), input_yuv),
                width=yuv_info["width"],
                height=yuv_info["height"],
                pix_fmt=yuv_info["pix_fmt"],
                decoded_file=out_yuv,
            )
        )
        run_command_decoder = decoder_cmd.format(
            filename='"{}"'.format(convert_path_to_wsl(encoded_filename)),
            outputname='"{}"'.format(convert_path_to_wsl(out_yuv)),
        )

        if decoder_os.lower() == "linux":
            run_command_decoder = [
                "start",
                "/wait",
                "cmd",
                "/c",
                "wsl",
                "-d",
                "Ubuntu",
                "-u",
                "root",
                "--",
                "bash",
                "-c",
                run_command_decoder,
            ]
        else:
            run_command_decoder = decoder_cmd.format(
                filename='"{}"'.format(encoded_filename),
                outputname='"{}"'.format(out_yuv),
            )
            # Pass as string (not list) — with shell=True on Windows, wrapping in
            # a list triggers list2cmdline which double-quotes the already-quoted paths.

        for metric in target_metric_list:
            run_command_vqmt += " " + metric
        vqmt_json_filename = os.path.join(
            os.getcwd(), "{}.json".format(md5hash(run_command_vqmt))
        )
        run_command_vqmt += ' -no-upscale-uv -json -json_file "{json_filename}" -threads {vqmt_threads} ^& exit'.format(
            json_filename=vqmt_json_filename, vqmt_threads=vqmt_threads
        )

        if not parallel_process:
            decoder_retcode = vqmt_retcode = 1
            try:
                decoder_start_time = time.time()
                decoder_retcode = subprocess.run(
                    run_command_decoder, shell=True, cwd=decoder_cwd
                ).returncode
                decoder_end_time = time.time()
                # Run VQMT sequentially after decoder (out_yuv file is ready)
                if decoder_retcode == 0 and metric_list and not vqmt_already_done:
                    vqmt_retcode = subprocess.run(
                        run_command_vqmt, shell=True, cwd=MSU_VQMT_PATH
                    ).returncode
                elif decoder_retcode == 0:
                    vqmt_retcode = 0  # no VQMT requested, or VQMT already in JSON
                # Compute torch perceptual metrics while out_yuv still exists (before finally deletes it)
                if torch_metric_list and decoder_retcode == 0 and os.path.exists(out_yuv):
                    try:
                        _torch_scores = _compute_torch_metrics(
                            os.path.abspath(os.path.join(os.getcwd(), input_yuv)),
                            out_yuv,
                            yuv_info,
                            torch_metric_list,
                            torch_models_path,
                        )
                    except Exception as _e:
                        print("[torch_metrics] Error: {}".format(_e))
                decoding_speed_bps = (
                    os.path.getsize(os.path.join(os.getcwd(), input_yuv))
                    / (decoder_end_time - decoder_start_time)
                    if (decoder_end_time - decoder_start_time) > 0
                    else 0.0
                )
                decoding_speed_fps = (
                    yuv_info["length"] / (decoder_end_time - decoder_start_time)
                    if (decoder_end_time - decoder_start_time) > 0
                    else 0.0
                )
            finally:
                try:
                    os.remove(out_yuv)
                except:
                    pass
        else:
            p_vqmt = subprocess.Popen(run_command_vqmt, shell=True, cwd=MSU_VQMT_PATH)
            start_time = time.time()
            while True:
                if pipe_name in list(os.listdir(r"\\.\pipe")):
                    break

                if (time.time() - start_time) > 5 * 60:
                    subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=p_vqmt.pid))
                    print("Can't wait! Kill vqmt: {}".format(result_json_filename))
                    return (
                        result,
                        decoding_speed_bps,
                        copy_speed_bps,
                        decompress_speed_bps,
                        decoding_speed_fps,
                    )

                time.sleep(1)

            p_decoder = subprocess.Popen(
                run_command_decoder, shell=True, cwd=decoder_cwd
            )
            try:
                decoder_start_time = time.time()
                decoder_retcode = p_decoder.wait(yuv_info["length"] * WAIT_MULTIPLIER)
                vqmt_retcode = p_vqmt.wait(2 * 60)
                decoder_end_time = time.time()
                decoding_speed_bps = (
                    os.path.getsize(os.path.join(os.getcwd(), input_yuv))
                    / (decoder_end_time - decoder_start_time)
                    if (decoder_end_time - decoder_start_time) > 0
                    else 0.0
                )
                decoding_speed_fps = (
                    yuv_info["length"] / (decoder_end_time - decoder_start_time)
                    if (decoder_end_time - decoder_start_time) > 0
                    else 0.0
                )
            except subprocess.TimeoutExpired:
                # p_vqmt.kill()
                # p_decoder.kill()
                subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=p_vqmt.pid))
                subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=p_decoder.pid))
                print(
                    "Can't wait! Kill vqmt & decoder: {}".format(result_json_filename)
                )
                return (
                    result,
                    decoding_speed_bps,
                    copy_speed_bps,
                    decompress_speed_bps,
                    decoding_speed_fps,
                )
        if not (vqmt_retcode == 0 and decoder_retcode == 0):
            raise Exception(
                "vqmt_retcode: {}\ndecoder_retcode: {}".format(
                    vqmt_retcode, decoder_retcode
                )
            )

        # Save VQMT results
        if metric_list:
            if vqmt_already_done:
                result = True  # VQMT data already loaded in result_json above
            else:
                try:
                    vqmt_scores = LoadJsonFromFile(vqmt_json_filename)
                except:
                    print("Error load json {}".format(vqmt_json_filename))
                    return (
                        result,
                        decoding_speed_bps,
                        copy_speed_bps,
                        decompress_speed_bps,
                        decoding_speed_fps,
                    )
                finally:
                    try:
                        os.remove(vqmt_json_filename)
                    except:
                        print("Can't remove {}".format(vqmt_json_filename))
                if len(vqmt_scores["values"]) == yuv_info["length"]:
                    result_json["vqmt"] = vqmt_scores
                    result_json["metric_list"] = metric_list
                    result = True
                else:
                    print(
                        "Error for {} (file {}): expected len: {}, got {}".format(
                            input_yuv,
                            vqmt_json_filename,
                            yuv_info["length"],
                            len(vqmt_scores["values"]),
                        )
                    )

        # Save torch results
        if _torch_scores:
            result_json["torch_metrics"] = _torch_scores
            result_json["torch_metric_list"] = torch_metric_list
            result = True

        # Write JSON to disk if anything was computed
        if result:
            DumpJsonToFile(result_json, result_json_filename)
    finally:
        try:
            if local_encoded_filename is not None and os.path.exists(
                local_encoded_filename
            ):
                os.remove(local_encoded_filename)
        except:
            print("Can't remove {}".format(local_encoded_filename))

    return (
        result,
        decoding_speed_bps,
        copy_speed_bps,
        decompress_speed_bps,
        decoding_speed_fps,
    )


def delete_files(files):
    for file in files:
        print("Delete {}".format(file))
        try:
            os.remove(file)
        except Exception as e:
            print("Error:", e)


def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve torch models path (default: pth/ subfolder next to this script)
    torch_models_path = args.torch_models_path or os.path.join(script_dir, "pth")

    with open(os.path.join(script_dir, "sheets_meta.json")) as file:
        sheets_info = json.load(file)

    if "credentials_file" in sheets_info and not os.path.isabs(
        sheets_info["credentials_file"]
    ):
        sheets_info["credentials_file"] = os.path.join(
            script_dir, sheets_info["credentials_file"]
        )
    exit_code = 0
    # Инициализируем массивы для сохранения скоростей
    decoding_speed_bps_list = []
    copy_speed_bps_list = []
    decompress_speed_bps_list = []
    decoding_speed_fps_list = []
    md5_speed_bps_list = []

    global PROGRESS, TELEGRAM_NOTIFIER

    # конфигурация
    proxy_kwargs = load_proxy_config(args.proxy_config)

    notifier_config_path = Path(args.notifier_config)
    with open(notifier_config_path) as file:
        notifier_cfg = json.load(file)

    if args.notify_via == "email":
        TELEGRAM_NOTIFIER = EmailNotifier(
            smtp_host=notifier_cfg["smtp_host"],
            smtp_port=notifier_cfg["smtp_port"],
            smtp_ssl=notifier_cfg["smtp_ssl"],
            smtp_user=notifier_cfg["smtp_user"],
            smtp_pass=notifier_cfg["smtp_pass"],
            mail_addr=notifier_cfg["mail_addr"],
            delay=notifier_cfg.get("delay", 900),
        )
    else:
        TELEGRAM_NOTIFIER = TelegramNotifier(
            notifier_cfg["telegram_token"],
            notifier_cfg["telegram_chat_id"],
            request_kwargs=proxy_kwargs,
            spreadsheet_id=sheets_info["spreadsheet_id"]
            if "spreadsheet_id" in sheets_info
            else None,
            sheets_creds_file=sheets_info["credentials_file"]
            if "credentials_file" in sheets_info
            else None,
            status_sheet_name=sheets_info["status_sheet_name"]
            if "status_sheet_name" in sheets_info
            else None,
            log_sheet_name=sheets_info["log_sheet_name"]
            if "log_sheet_name" in sheets_info
            else None,
        )

    sequences_info = json.loads(args.sequences_json.read())
    sequences_info = {s["name"]: s for s in sequences_info}
    codecs_info = json.loads(args.codecs_json.read())
    codecs_info = {c["name"]: c for c in codecs_info}

    if args.bitrates_map:
        bitrates_map = json.loads(args.bitrates_map.read())
        tested_bitrates = bitrates_map[socket.gethostname().lower()]
    else:
        tested_bitrates = args.bitrates

    decoded_storage = args.decoded_storage
    try:
        os.makedirs(decoded_storage, exist_ok=True)
    except:
        decoded_storage = os.path.abspath(".")
        print(
            f"Error while get access to {args.decoded_storage}, change decoded_storage to {decoded_storage}"
        )

    vqmt_threads = args.vqmt_threads
    if vqmt_threads < 0:
        vqmt_threads = max(0, (os.cpu_count() - 2) // args.threads)

    PROGRESS = {
        "done": 0,
        "total": 0,
        "done_frames": 0,
        "total_frames": 0,
        "info": {
            "sequences": set(),
            "codecs": set(),
            "presets": set(),
            "bitrates": set(),
        },
    }

    streamlake_check_passed = False
    if any(["streamlake" in codec for codec in args.codecs]):
        check_result = check_streamlake(
            os.path.dirname(codecs_info["streamlake_v3"]["decoder_json"])
        )
        if check_result != 0:
            if not args.silent:
                if check_result == 1:
                    TELEGRAM_NOTIFIER.error(["Ubuntu is not installed in WSL"])
                else:
                    TELEGRAM_NOTIFIER.error(["Error while checking streamlake"])

            exit(1)

        streamlake_check_passed = True

    for sequence in args.sequences:
        sequence_info = sequences_info[sequence]

        # Copy sequence from remote storage
        local_filename = sequence_info["local_path"]
        run_arguments = []
        for preset in args.presets:
            for codec in args.codecs:
                codec_info = codecs_info[codec]
                decoder_info = LoadJsonFromFile(codec_info["decoder_json"])
                for bitrate in tested_bitrates:
                    encoded_filename = os.path.join(
                        args.result_path,
                        "encoded_streams",
                        sequence,
                        preset,
                        "enc_res_{codec}_{preset}_{seq}_{bitrate}".format(
                            codec=codec, preset=preset, seq=sequence, bitrate=bitrate
                        ),
                    )
                    result_json_filename = os.path.join(
                        args.result_path,
                        args.metrics_path,
                        sequence,
                        preset,
                        "{codec}_{bitrate}.json".format(codec=codec, bitrate=bitrate),
                    )
                    base_json_filename = os.path.join(
                        args.result_path,
                        "results",
                        sequence,
                        preset,
                        "{codec}_{bitrate}.json".format(codec=codec, bitrate=bitrate),
                    )
                    if not os.path.isfile(base_json_filename):
                        continue
                    if not args.force and os.path.isfile(result_json_filename):
                        try:
                            tmp_jf = LoadJsonFromFile(result_json_filename)
                            vqmt_done = (
                                not args.metrics
                                or (
                                    "vqmt" in tmp_jf.keys()
                                    and set(args.metrics) <= set(tmp_jf["metric_list"])
                                    and (
                                        args.vqmt_version is None
                                        or tmp_jf["vqmt"]["generator"]["program"].endswith(
                                            args.vqmt_version
                                        )
                                    )
                                )
                            )
                            torch_done = (
                                not args.torch_metrics
                                or set(args.torch_metrics) <= set(tmp_jf.get("torch_metric_list", []))
                            )
                            if vqmt_done and torch_done:
                                continue
                        except:
                            pass
                    PROGRESS["info"]["codecs"].add(codec)
                    PROGRESS["info"]["presets"].add(preset)
                    PROGRESS["info"]["bitrates"].add(bitrate)
                    PROGRESS["info"]["sequences"].add(sequence)
                    PROGRESS["total"] += 1
                    PROGRESS["total_frames"] += sequence_info["length"]

    for key in PROGRESS["info"]:
        PROGRESS["info"][key] = list(map(str, sorted(list(PROGRESS["info"][key]))))

    if not args.silent and PROGRESS["total"] != 0:
        TELEGRAM_NOTIFIER.start(
            PROGRESS["total"],
            [(key.capitalize(), val) for (key, val) in PROGRESS["info"].items()]
            + [("OS", [system()])],
            [sorted(args.codecs), sorted(args.sequences), PROGRESS["total_frames"]],
        )
        if streamlake_check_passed:
            TELEGRAM_NOTIFIER._send_text("Streamlake test passed", ["info"])

        signals = [
            "SIGABRT",
            "SIGALRM",
            "SIGFPE",
            "SIGILL",
            "SIGINT",
            "SIGIO",
            "SIGPOLL",
            "SIGPROF",
            "SIGPWR",
            "SIGQUIT",
            "SIGSEGV",
            "SIGSTKFLT",
            "SIGSYS",
            "SIGTERM",
            "SIGTRAP",
            "SIGTSTP",
            "SIGTTIN",
            "SIGTTOU",
            "SIGUSR1",
            "SIGUSR2",
            "SIGVTALRM",
            "SIGXCPU",
            "SIGBUS",
        ]

        signals = [getattr(signal, sig) for sig in dir(signal) if sig in signals]

        for sig in signals:
            signal.signal(sig, exit_handler)

        atexit.register(exit_handler)

    if args.local_storage is not None:
        os.makedirs(args.local_storage, exist_ok=True)
        time_file = os.path.join(args.local_storage, "time.json")
        if os.path.exists(time_file):
            with open(time_file, "r", encoding="utf-8") as f:
                file_times = json.load(f)
        else:
            file_times = {}
    else:
        time_file = None
        file_times = {}

    converted_file_times = {}
    for rel_path, timestamp in file_times.items():
        if system().lower() == "windows":
            rel_path = rel_path.replace("/", "\\")
        else:
            rel_path = rel_path.replace("\\", "/")

        key = (
            rel_path
            if os.path.isabs(rel_path)
            else os.path.abspath(os.path.join(args.local_storage, rel_path))
        )
        converted_file_times[key] = timestamp

    file_times = converted_file_times

    for sequence in args.sequences:
        sequence_info = sequences_info[sequence]
        # Copy sequence from remote storage
        local_filename = sequence_info["local_path"]
        sequence_info["remote_path"] = sequence_info["remote_path"].format(
            base_storage=args.sequences_base_storage
        )
        yuv_gt_md5 = sequence_info["md5"]

        if args.local_storage is not None:
            local_storage_yuv_file = os.path.abspath(sequence_info["remote_path"])
            while (
                ":" in local_storage_yuv_file
                or local_storage_yuv_file.startswith("\\")
                or local_storage_yuv_file.startswith("/")
            ):
                local_storage_yuv_file = local_storage_yuv_file[1:]
            local_storage_yuv_file = os.path.join(
                args.local_storage, local_storage_yuv_file
            )

            local_filename = None

        run_arguments = []
        remove_files = []
        for preset in args.presets:
            for codec in args.codecs:
                codec_info = codecs_info[codec]
                decoder_info = LoadJsonFromFile(codec_info["decoder_json"])
                for bitrate in tested_bitrates:
                    encoded_filename = os.path.join(
                        args.result_path,
                        "encoded_streams",
                        sequence,
                        preset,
                        "enc_res_{codec}_{preset}_{seq}_{bitrate}".format(
                            codec=codec, preset=preset, seq=sequence, bitrate=bitrate
                        ),
                    )
                    result_json_filename = os.path.join(
                        args.result_path,
                        args.metrics_path,
                        sequence,
                        preset,
                        "{codec}_{bitrate}.json".format(codec=codec, bitrate=bitrate),
                    )
                    base_json_filename = os.path.join(
                        args.result_path,
                        "results",
                        sequence,
                        preset,
                        "{codec}_{bitrate}.json".format(codec=codec, bitrate=bitrate),
                    )
                    if not os.path.isfile(base_json_filename):
                        continue
                    if not args.force and os.path.isfile(result_json_filename):
                        try:
                            tmp_jf = LoadJsonFromFile(result_json_filename)
                            vqmt_done = (
                                not args.metrics
                                or (
                                    "vqmt" in tmp_jf.keys()
                                    and set(args.metrics) <= set(tmp_jf["metric_list"])
                                    and (
                                        args.vqmt_version is None
                                        or tmp_jf["vqmt"]["generator"]["program"].endswith(
                                            args.vqmt_version
                                        )
                                    )
                                )
                            )
                            torch_done = (
                                not args.torch_metrics
                                or set(args.torch_metrics) <= set(tmp_jf.get("torch_metric_list", []))
                            )
                            if vqmt_done and torch_done:
                                continue
                        except:
                            print("Bad json", result_json_filename)
                    if not os.path.isdir(os.path.dirname(result_json_filename)):
                        os.makedirs(os.path.dirname(result_json_filename))
                    if args.force:
                        shutil.copy(base_json_filename, result_json_filename)

                    local_encoded_filename = os.path.abspath(
                        os.path.join(".", os.path.basename(encoded_filename))
                    )
                    run_arguments.append(
                        [
                            CalculateMetrics,
                            [
                                local_filename
                                if local_filename is not None
                                else local_storage_yuv_file,
                                sequence_info,
                                encoded_filename,
                                os.path.dirname(codec_info["decoder_json"]),
                                decoder_info["command"],
                                decoder_info.get("OS", "Windows"),
                                decoded_storage,
                                base_json_filename,
                                result_json_filename,
                                args.metrics,
                                vqmt_threads,
                                local_encoded_filename,
                                args.torch_metrics,
                                torch_models_path,
                            ],
                        ]
                    )

        if len(run_arguments) == 0:
            continue

        if args.local_storage is not None and local_storage_yuv_file is not None:
            zst_file = os.path.splitext(local_storage_yuv_file)[0] + ".zst"
            remote_zst_file = os.path.splitext(sequence_info["remote_path"])[0] + ".zst"

            total_needed_size = 0
            if not os.path.exists(local_storage_yuv_file):
                total_needed_size += sequence_info["size"]

                if (
                    args.try_zst
                    and not os.path.exists(zst_file)
                    and os.path.exists(remote_zst_file)
                ):
                    try:
                        total_needed_size += os.path.getsize(remote_zst_file)
                    except:
                        pass

            usage = shutil.disk_usage(args.local_storage)
            if usage.free < total_needed_size:

                def remove_file_with_update(file_path):
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    if file_path in file_times:
                        del file_times[file_path]

                for root, _, files in os.walk(args.local_storage):
                    for file in files:
                        file_path = os.path.abspath(os.path.join(root, file))
                        if not (
                            file.endswith(".yuv")
                            or file.endswith(".zst")
                            or file.endswith(".zip")
                            or file.endswith(".y4m")
                        ):
                            continue

                        if file_path not in file_times:
                            remove_file_with_update(file_path)

                usage = shutil.disk_usage(args.local_storage)
                while usage.free < total_needed_size and len(file_times) > 0:
                    sorted_files = sorted(
                        [(not k.endswith(".yuv"), v, k) for k, v in file_times.items()]
                    )

                    for _, _, file_path in sorted_files:
                        if not os.path.exists(file_path):
                            if file_path in file_times:
                                del file_times[file_path]
                        else:
                            remove_file_with_update(file_path)
                            break

                    usage = shutil.disk_usage(args.local_storage)

            use_zst = False
            if not os.path.exists(local_storage_yuv_file):
                if args.try_zst and os.path.exists(remote_zst_file):
                    if not os.path.exists(zst_file):
                        os.makedirs(os.path.dirname(zst_file), exist_ok=True)
                        print(f"Copying {remote_zst_file} to {zst_file}...")
                        zst_copy_start_time = time.time()
                        subprocess.check_call(
                            'start /wait cmd /C copy /Y "%s" "%s"'
                            % (remote_zst_file, zst_file),
                            shell=True,
                        )
                        zst_copy_end_time = time.time()
                        zst_file_size = os.path.getsize(zst_file)
                        zst_copy_speed_bps = (
                            zst_file_size / (zst_copy_end_time - zst_copy_start_time)
                            if (zst_copy_end_time - zst_copy_start_time) > 0
                            else 0.0
                        )
                        copy_speed_bps_list.append(zst_copy_speed_bps)
                        print("ZST file copied")
                    use_zst = True
                    print(f"Decompressing {zst_file} to {local_storage_yuv_file}...")
                    os.makedirs(os.path.dirname(local_storage_yuv_file), exist_ok=True)

                    try:
                        decompress_start_time = time.time()
                        subprocess.check_call(
                            [ZSTD_BIN, "-d", zst_file, "-o", local_storage_yuv_file]
                        )
                        decompress_end_time = time.time()
                        yuv_file_size = os.path.getsize(local_storage_yuv_file)
                        decompress_speed_bps = (
                            yuv_file_size
                            / (decompress_end_time - decompress_start_time)
                            if (decompress_end_time - decompress_start_time) > 0
                            else 0.0
                        )
                        decompress_speed_bps_list.append(decompress_speed_bps)
                        print("File decompressed successfully")

                    except subprocess.CalledProcessError as e:
                        print(f"Error decompressing file: {e}")
                        delete_files(remove_files)
                        continue
                else:
                    if not os.path.exists(sequence_info["remote_path"]):
                        print(f"Cannot find file {sequence_info['remote_path']}")
                        delete_files(remove_files)
                        continue

                    os.makedirs(os.path.dirname(local_storage_yuv_file), exist_ok=True)
                    print(
                        f"Copying {sequence_info['remote_path']} to {local_storage_yuv_file}..."
                    )
                    yuv_copy_start_time = time.time()
                    subprocess.check_call(
                        'start /wait cmd /C copy /Y "%s" "%s"'
                        % (sequence_info["remote_path"], local_storage_yuv_file),
                        shell=True,
                    )
                    yuv_copy_end_time = time.time()
                    yuv_file_size = os.path.getsize(local_storage_yuv_file)
                    yuv_copy_speed_bps = (
                        yuv_file_size / (yuv_copy_end_time - yuv_copy_start_time)
                        if (yuv_copy_end_time - yuv_copy_start_time) > 0
                        else 0.0
                    )
                    copy_speed_bps_list.append(yuv_copy_speed_bps)
                    print("Sequence copied")

                file_times[os.path.abspath(local_storage_yuv_file)] = datetime.now(
                    timezone.utc
                ).timestamp()
                zst_key = os.path.abspath(local_storage_yuv_file).replace(
                    ".yuv", ".zst"
                )

                if use_zst:
                    file_times[zst_key] = file_times[
                        os.path.abspath(local_storage_yuv_file)
                    ]

        if time_file is not None:
            relative_file_times = {}
            for abs_path, timestamp in file_times.items():
                try:
                    rel_path = os.path.relpath(abs_path, args.local_storage)
                    rel_path = rel_path.replace("\\", "/")
                    relative_file_times[rel_path] = timestamp
                except ValueError:
                    # Если путь не может быть сделан относительным (например, на разных дисках в Windows)
                    # Оставляем абсолютный путь, но также нормализуем разделители
                    rel_path = abs_path.replace("\\", "/")
                    relative_file_times[rel_path] = timestamp

            with open(time_file, "w", encoding="utf-8") as f:
                json.dump(relative_file_times, f)

        if local_filename is not None:
            remote_zst_file = os.path.splitext(sequence_info["remote_path"])[0] + ".zst"
            if args.try_zst and os.path.exists(remote_zst_file):
                print(f"Decompressing {remote_zst_file} to {local_filename}...")
                try:
                    decompress_start_time = time.time()
                    subprocess.check_call(
                        [ZSTD_BIN, "-d", remote_zst_file, "-o", local_filename]
                    )
                    decompress_end_time = time.time()
                    yuv_file_size = os.path.getsize(local_filename)
                    decompress_speed_bps = (
                        yuv_file_size / (decompress_end_time - decompress_start_time)
                        if (decompress_end_time - decompress_start_time) > 0
                        else 0.0
                    )
                    decompress_speed_bps_list.append(decompress_speed_bps)
                    print("File decompressed successfully")
                except subprocess.CalledProcessError as e:
                    print(f"Error decompressing file: {e}")
                    delete_files(remove_files)
                    continue
            else:
                if not os.path.exists(sequence_info["remote_path"]):
                    print(f"Cannot find file {sequence_info['remote_path']}")
                    delete_files(remove_files)
                    continue

                os.makedirs(os.path.dirname(local_filename), exist_ok=True)
                print(f"Copying {sequence_info['remote_path']} to {local_filename}...")
                yuv_copy_start_time = time.time()
                subprocess.check_call(
                    'start /wait cmd /C copy /Y "%s" "%s"'
                    % (sequence_info["remote_path"], local_filename),
                    shell=True,
                )
                yuv_copy_end_time = time.time()
                yuv_file_size = os.path.getsize(local_filename)
                yuv_copy_speed_bps = (
                    yuv_file_size / (yuv_copy_end_time - yuv_copy_start_time)
                    if (yuv_copy_end_time - yuv_copy_start_time) > 0
                    else 0.0
                )
                copy_speed_bps_list.append(yuv_copy_speed_bps)
                print("Sequence copied")

            remove_files.append(local_filename)

        md5_start_time = time.time()
        if local_filename is not None:
            yuv_md5 = md5sum(Path(local_filename))
            md5_file_size = os.path.getsize(local_filename)
        else:
            yuv_md5 = md5sum(Path(local_storage_yuv_file))
            md5_file_size = os.path.getsize(local_storage_yuv_file)

        md5_end_time = time.time()
        md5_speed_bps = (
            md5_file_size / (md5_end_time - md5_start_time)
            if (md5_end_time - md5_start_time) > 0
            else 0.0
        )
        md5_speed_bps_list.append(md5_speed_bps)

        if yuv_md5 != yuv_gt_md5:
            print(
                "MD5 of {} is equal to {}. But should be {}".format(
                    local_filename
                    if local_filename is not None
                    else local_storage_yuv_file,
                    yuv_md5,
                    yuv_gt_md5,
                )
            )
            exit_code = 1
            delete_files(remove_files)
            continue

        widgets = [
            sequence,
            ": ",
            progressbar.Percentage(),
            " (",
            progressbar.SimpleProgress(),
            ")",
            " ",
            progressbar.Bar(),
            " ",
            progressbar.Timer(),
            " ",
            progressbar.AdaptiveETA(),
        ]
        bar = progressbar.ProgressBar(widgets=widgets, redirect_stdout=True)
        bad_wsl_reported = False

        with Pool(processes=args.threads) as pool:
            workers = pool.imap_unordered(StarFunc, run_arguments)
            for arg in bar(run_arguments):
                result = workers.next()
                PROGRESS["done"] += 1
                PROGRESS["done_frames"] += arg[1][1]["length"]

                # Собираем скорости из результата
                if (
                    result is not None
                    and isinstance(result, tuple)
                    and len(result) == 5
                ):
                    done, decoding_bps, copy_bps, decompress_bps, decoding_fps = result
                    if done:
                        if decoding_bps > 0:
                            decoding_speed_bps_list.append(decoding_bps)
                        if copy_bps > 0:
                            copy_speed_bps_list.append(copy_bps)
                        if decompress_bps > 0:
                            decompress_speed_bps_list.append(decompress_bps)
                        if decoding_fps > 0:
                            decoding_speed_fps_list.append(decoding_fps)
                    else:
                        exit_code = 1

                try:
                    if not args.silent:
                        if (
                            not bad_wsl_reported
                            and arg[1][5].lower() == "linux"
                            and not check_wsl_has_linux()
                        ):
                            bad_wsl_reported = True
                            TELEGRAM_NOTIFIER.error(["Ubuntu is not installed in WSL"])

                        TELEGRAM_NOTIFIER.progress(
                            PROGRESS["done"],
                            PROGRESS["total"],
                            PROGRESS["done_frames"],
                            PROGRESS["total_frames"],
                            ignore_delay=PROGRESS["done"] == PROGRESS["total"],
                        )
                except Exception as e:
                    print("Error occurred while sending progress", e)

        delete_files(remove_files)

    # Сохраняем скорости в файл, если указан
    if args.output_speeds_file:
        speeds_data = {
            "decoding_speed_bps": decoding_speed_bps_list,
            "copy_speed_bps": copy_speed_bps_list,
            "decompress_speed_bps": decompress_speed_bps_list,
            "decoding_speed_fps": decoding_speed_fps_list,
            "md5_speed_bps": md5_speed_bps_list,
        }
        try:
            with open(args.output_speeds_file, "w", encoding="utf-8") as f:
                json.dump(speeds_data, f, indent=2)
            print(f"Speeds saved to {args.output_speeds_file}")
        except Exception as e:
            print(f"Error saving speeds to {args.output_speeds_file}: {e}")

    if not args.silent:
        try:
            TELEGRAM_NOTIFIER.finish(
                PROGRESS["done"],
                PROGRESS["total"],
                PROGRESS["done_frames"],
                PROGRESS["total_frames"],
                PROGRESS["info"]["codecs"],
                PROGRESS["info"]["sequences"],
                PROGRESS["info"]["presets"],
                PROGRESS["info"]["bitrates"],
            )
            TELEGRAM_NOTIFIER.stop()
        except Exception as e:
            print("Error occurred while finishing", e)

    print("Program has been successfully completed")
    sys.stdout.flush()
    exit(exit_code)


def exit_handler(signal_num=None, frame=None):
    global TELEGRAM_NOTIFIER, PROGRESS

    print(f"Exiting due to signal {signal_num}")

    try:
        TELEGRAM_NOTIFIER.progress(
            PROGRESS["done"],
            PROGRESS["total"],
            PROGRESS["done_frames"],
            PROGRESS["total_frames"],
            ignore_delay=True,
        )

        TELEGRAM_NOTIFIER.finish(
            PROGRESS["done"],
            PROGRESS["total"],
            PROGRESS["done_frames"],
            PROGRESS["total_frames"],
            PROGRESS["info"]["codecs"],
            PROGRESS["info"]["sequences"],
            PROGRESS["info"]["presets"],
            PROGRESS["info"]["bitrates"],
            send_log=False,
        )
        TELEGRAM_NOTIFIER.stop()
    except Exception as e:
        print("Error occurred while finishing", e)
    sys.stdout.flush()

    if signal_num is not None:
        signal.signal(signal_num, signal.SIG_DFL)
        signal.raise_signal(signal_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--result-path",
        type=str,
        default=os.path.join(
            "\\\\titan", "Codec Comparison", "HEVC_2018", "launches_data"
        ),
        help="JSON file containing info about test sequences",
    )
    parser.add_argument(
        "--sequences-json",
        default=os.path.join("..", "sequences", "info.json"),
        type=argparse.FileType("rt", encoding="UTF-8"),
        help="JSON file containing info about test sequences",
    )
    parser.add_argument(
        "--sequences_base_storage",
        type=str,
        required=True,
        help="Name of storage with sequences",
    )
    parser.add_argument(
        "--codecs-json",
        default=os.path.join("..", "codecs", "info.json"),
        type=argparse.FileType("rt", encoding="UTF-8"),
        help="JSON file containing info about codecs",
    )
    parser.add_argument(
        "-c",
        "--codecs",
        required=True,
        type=str,
        nargs="+",
        help="Tested encoder names",
    )
    parser.add_argument(
        "-p",
        "--presets",
        default=[
            "mv_fast",
            "mv_universal",
            "mv_ripping",
        ],
        type=str,
        nargs="+",
        help="Tested presets",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        default=[
            "-metr psnr over Y,U,V",
            "-metr ssim_precise over Y,U,V",
            "-metr msssim over Y,U,V",
            "-metr vmaf over Y,U,V -set model_preset=vmaf_v061",
            "-metr vmaf over Y,U,V -set model_preset=vmaf_v062",
            "-metr vmaf over Y,U,V -set model_preset=vmaf_v063",
            "-metr vmaf over Y,U,V -set model_preset=vmaf_v061_neg",
        ],
        type=str,
        nargs="*",
        help="Metrics passed to MSU VQMT (pass --metrics with no args to disable VQMT entirely)",
    )
    parser.add_argument(
        "-b",
        "--bitrates",
        default=["6000"],
        type=str,
        nargs="+",
        help="Tested encoder names",
    )
    parser.add_argument("--bitrates-map", type=open, help="JSON with bitrates map")
    parser.add_argument(
        "--metrics-path",
        type=str,
        help="PATH to metric results",
        default="results-metrics",
    )
    parser.add_argument(
        "-s",
        "--sequences",
        type=str,
        default=[],
        nargs="+",
        help="Tested sequences names",
    )
    parser.add_argument("-t", "--threads", type=int, default=1, help="Threads")
    parser.add_argument("--vqmt_threads", type=int, default=-1, help="VQMT threads")
    parser.add_argument("--force", dest="force", action="store_true")
    parser.add_argument(
        "--vqmt_version", type=str, default=None, help="Vqmt version to control"
    )
    parser.add_argument(
        "--silent", action="store_true", help="Disables telegram notifications"
    )
    parser.add_argument(
        "--local_storage",
        type=str,
        default=None,
        help="Path for local storage of source YUVs",
    )
    parser.add_argument(
        "--decoded_storage",
        type=str,
        default="D:\\",
        help="Path for local decoded YUVs",
    )
    parser.add_argument(
        "--try_zip",
        action="store_true",
        help="Try use zip file if there is no source YUV",
    )
    parser.add_argument(
        "--try_zst",
        action="store_true",
        help="Try use zst file if there is no source YUV",
    )
    parser.add_argument(
        "--proxy-config",
        type=str,
        default=None,
        help="Path to proxy configuration JSON file (only used with --notify-via telegram)",
    )
    parser.add_argument(
        "--notify-via",
        choices=["telegram", "email"],
        default="telegram",
        help="Способ отправки уведомлений: telegram (по умолчанию) или email через SMTP",
    )
    parser.add_argument(
        "--notifier-config",
        type=str,
        required=True,
        help="Path to notifier_config.json with Telegram and SMTP credentials",
    )
    parser.add_argument(
        "--output-speeds-file",
        type=str,
        default=None,
        help="Path to JSON file for saving speed arrays (decoding_speed_bps, copy_speed_bps, decompress_speed_bps, decoding_speed_fps)",
    )
    parser.add_argument(
        "--torch-metrics",
        type=str,
        nargs="*",
        default=[],
        choices=list(_TORCH_METRICS.keys()),
        help="Torch-based perceptual metrics to compute frame by frame (lpips_alex, lpips_vgg, lpips_plus, lpips_vgg_plus, dists)",
    )
    parser.add_argument(
        "--torch-models-path",
        type=str,
        default=None,
        help="Path to directory with model weight subfolders (default: ../subjects/ relative to script)",
    )

    parser.set_defaults(force=False)
    args = parser.parse_args()
    args.try_zst = args.try_zst or args.try_zip
    main(args)
