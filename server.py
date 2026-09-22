from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from color_engine.palette import (
    apply_palette_profile,
    build_palette_profile,
    load_palette_profile,
    render_palette_profile,
    save_palette_profile,
)
from color_engine.corrections import (
    estimate_low_chroma_cast,
    reduce_warm_cast,
    resolve_low_frequency_strengths,
)
from color_engine.report import write_seam_report
from color_engine.seams import analyze_placement_seams
from experiments.structure_recolor.engine import decompose_tile, joint_grayscale_normalize, recolor_tile

import toolkit_core as core

try:
    import cv2
except Exception:
    cv2 = None


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
ASSET_ROOT = ROOT / "assets"
INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "outputs"
PREVIEW_DIR = OUTPUT_DIR / "_previews"
HISTORY_FILE = OUTPUT_DIR / "manifest_history.json"
STITCH_HISTORY_FILE = OUTPUT_DIR / "stitch_history.json"
DEFAULT_PORT = core.PORT_MAP["xiangliu-grid"]
APP_NAME = "相柳网格"
APP_NAME_EN = "Xiangliu Grid"
APP_VERSION = "0.6.0"

# --------------------------------------------------------------------------
# 输入上限（SERIES-SPEC §7 / S5）
#
# 旧版对这些参数完全不设防：overlap >= 块边长 时步长会被静默压成 1，
# 推导出上百万个图块，接口直接把服务拖死。现在越界一律返回 400 + 中文说明。
# --------------------------------------------------------------------------

MAX_LONG_EDGE = 8192          # 切块边长
MAX_OVERLAP = 1024            # 重叠像素
MAX_TARGET_PIECES = 4096      # 目标块数
MAX_GRID_SIDE = 256           # 行列裁切的最大行数 / 列数
MAX_TILES_PER_JOB = 20000     # 单次切分最多产出多少图块
MAX_CANVAS_PIXELS = 4_000_000_000
MAX_RESIZE_SCALE = 16.0       # 分辨率导出的最大倍数
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024   # 单次上传 2 GB

MAX_STRUCTURE_RECOLOR_TILES = 64

for directory in (INPUT_DIR, OUTPUT_DIR, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", name, flags=re.UNICODE)
    return name.strip("._") or f"image_{int(time.time())}"


def resolve_image_path(value: str) -> Path:
    raw = unquote(value or "").strip().strip('"')
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / raw).resolve()
    return path


def choose_file(title: str, filetypes: list[tuple[str, str]]) -> str:
    if sys.platform.startswith("win"):
        return choose_file_windows(title, filetypes)
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askopenfilename(title=title, filetypes=filetypes)
    finally:
        root.destroy()


def choose_folder(title: str) -> str:
    if sys.platform.startswith("win"):
        return choose_folder_windows(title)
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askdirectory(title=title)
    finally:
        root.destroy()


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def run_powershell_dialog(command: str) -> str:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    return completed.stdout.strip()


def clean_dialog_output(value: str, suffixes: list[str] | None = None) -> str:
    raw = (value or "").strip().strip('"')
    if not raw:
        return ""
    for line in raw.splitlines():
        candidate = line.strip().strip('"')
        if candidate and Path(candidate).exists():
            return candidate
    if suffixes:
        lowered = raw.lower()
        best_end = None
        for suffix in sorted({s.lower() for s in suffixes if s}, key=len, reverse=True):
            idx = lowered.find(suffix)
            if idx >= 0:
                end = idx + len(suffix)
                if best_end is None or end < best_end:
                    best_end = end
        if best_end:
            return raw[:best_end].strip().strip('"')
    return raw


def choose_file_windows(title: str, filetypes: list[tuple[str, str]]) -> str:
    filters = []
    suffixes = []
    for label, pattern in filetypes:
        parts = [part for part in pattern.split() if part]
        suffixes.extend(part.replace("*", "") for part in parts if part.startswith("*."))
        filter_pattern = ";".join(parts)
        filters.append(f"{label} ({filter_pattern})|{filter_pattern}")
    filter_text = "|".join(filters) if filters else "All files (*.*)|*.*"
    command = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$f=New-Object System.Windows.Forms.Form;"
        "$f.TopMost=$true;$f.ShowInTaskbar=$false;$f.WindowState='Minimized';$f.Show();"
        "$d=New-Object System.Windows.Forms.OpenFileDialog;"
        f"$d.Title={powershell_quote(title)};"
        f"$d.Filter={powershell_quote(filter_text)};"
        "if($d.ShowDialog($f) -eq [System.Windows.Forms.DialogResult]::OK)"
        "{Write-Output $d.FileName};"
        "$f.Close();"
    )
    return clean_dialog_output(run_powershell_dialog(command), suffixes)


def choose_folder_windows(title: str) -> str:
    command = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$f=New-Object System.Windows.Forms.Form;"
        "$f.TopMost=$true;$f.ShowInTaskbar=$false;$f.WindowState='Minimized';$f.Show();"
        "$d=New-Object System.Windows.Forms.OpenFileDialog;"
        f"$d.Title={powershell_quote(title)};"
        "$d.Filter='Folders|*.folder';"
        "$d.CheckFileExists=$false;"
        "$d.CheckPathExists=$true;"
        "$d.ValidateNames=$false;"
        "$d.FileName='选择此文件夹';"
        "if($d.ShowDialog($f) -eq [System.Windows.Forms.DialogResult]::OK)"
        "{if([System.IO.Directory]::Exists($d.FileName)){Write-Output $d.FileName}"
        "else{Write-Output ([System.IO.Path]::GetDirectoryName($d.FileName))}};"
        "$f.Close();"
    )
    return clean_dialog_output(run_powershell_dialog(command))


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    core.apply_cors(handler)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    try:
        handler.wfile.write(data)
    except (BrokenPipeError, ConnectionResetError):
        pass


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(items: list[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(items[:30], f, ensure_ascii=False, indent=2)


def remember_manifest(manifest_path: Path, tiles_dir: Path | None = None, job_name: str = "") -> None:
    items = load_history()
    manifest_path = manifest_path.resolve()
    items = [item for item in items if item.get("manifest_json") != str(manifest_path)]
    item = {
        "job_name": job_name or manifest_path.parent.name,
        "manifest_json": str(manifest_path),
        "tiles_dir": str(tiles_dir.resolve()) if tiles_dir else "",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    items.insert(0, item)
    save_history(items)


def list_manifest_history() -> list[dict]:
    items = load_history()
    known = {item.get("manifest_json") for item in items}
    for manifest in sorted(OUTPUT_DIR.glob("**/tiles_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        resolved = str(manifest.resolve())
        if resolved in known:
            continue
        items.append(
            {
                "job_name": manifest.parent.name,
                "manifest_json": resolved,
                "tiles_dir": str((manifest.parent / "tiles").resolve()),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(manifest.stat().st_mtime)),
            }
        )
    return items[:30]


STITCH_HISTORY_FIELDS = (
    "manifest_path",
    "restored_dir",
    "out_name",
    "output_dir",
    "mode",
    "selected_tiles",
    "blend_mode",
    "color_mode",
    "seam_strength",
    "low_freq_strength",
    "low_freq_mode",
    "yellow_reduction",
    "color_tiles",
    "trim_padding",
    "feather_px",
    "reference_path",
    "palette_reference_paths",
    "palette_profile_path",
    "palette_strength",
    "saturation_gain",
    "neutral_protection",
    "skin_protection",
    "gold_protection",
    "ink_lightness",
    "ink_expand_px",
    "seam_diagnostics",
)


def load_stitch_history() -> list[dict]:
    if not STITCH_HISTORY_FILE.exists():
        return []
    try:
        with STITCH_HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_stitch_history(items: list[dict]) -> None:
    STITCH_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STITCH_HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(items[:10], f, ensure_ascii=False, indent=2)


def remember_stitch_history(payload: dict, result: dict) -> None:
    settings = {key: payload.get(key) for key in STITCH_HISTORY_FIELDS if key in payload}
    signature = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    items = [item for item in load_stitch_history() if item.get("signature") != signature]
    manifest = Path(str(settings.get("manifest_path") or ""))
    item = {
        "job_name": settings.get("out_name") or manifest.parent.name or "拼合记录",
        "settings": settings,
        "out_path": str(result.get("out_path", "")),
        "canvas_width": int(result.get("canvas_width", 0)),
        "canvas_height": int(result.get("canvas_height", 0)),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "signature": signature,
    }
    items.insert(0, item)
    save_stitch_history(items)


def read_json(handler: BaseHTTPRequestHandler) -> dict:
    """读取 JSON 请求体，带 1 MB 体积上限（SERIES-SPEC §7 / S3）。"""
    return core.read_json(handler)


def clamp_int(
    payload: dict, key: str, default: int, low: int, high: int, label: str
) -> int:
    """取一个整数参数并夹在 [low, high] 内；越界抛带中文说明的 ValidationError。"""
    raw = payload.get(key, default)
    if isinstance(raw, bool) or raw is None or raw == "":
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise core.ValidationError(
            f"{label}必须是整数。", field=key, detail=f"value={raw!r}")
    if value < low or value > high:
        raise core.ValidationError(
            f"{label}必须在 {low} 到 {high} 之间。", field=key, detail=f"{key}={value}")
    return value


def clamp_float(
    payload: dict, key: str, default: float, low: float, high: float, label: str
) -> float:
    """取一个浮点参数并夹在 [low, high] 内。"""
    raw = payload.get(key, default)
    if isinstance(raw, bool) or raw is None or raw == "":
        raw = default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise core.ValidationError(
            f"{label}必须是数字。", field=key, detail=f"value={raw!r}")
    if not math.isfinite(value):
        raise core.ValidationError(f"{label}不是有效数字。", field=key, detail=f"{key}={raw!r}")
    if value < low or value > high:
        raise core.ValidationError(
            f"{label}必须在 {low} 到 {high} 之间。", field=key, detail=f"{key}={value}")
    return value


def check_tile_budget(plan: dict) -> None:
    """切图前的总量校验：块数、画布像素，防止一个请求把服务拖死。"""
    tiles = plan.get("tiles")
    count = len(tiles) if isinstance(tiles, list) else int(plan.get("tile_count", 0) or 0)
    if count > MAX_TILES_PER_JOB:
        raise core.ValidationError(
            f"这次会切出 {count} 个图块，超过上限 {MAX_TILES_PER_JOB}。"
            "请增大块尺寸、减少目标块数，或降低重叠。",
            field="target_pieces", detail=f"tiles={count}")
    canvas = plan.get("canvas") or {}
    pixels = int(canvas.get("width", 0) or 0) * int(canvas.get("height", 0) or 0)
    if pixels > MAX_CANVAS_PIXELS:
        raise core.ValidationError(
            "工作画布像素过多（超过 40 亿），请减小块尺寸或目标块数。",
            field="long_edge", detail=f"pixels={pixels}")


def check_overlap(long_edge: int, overlap: int) -> None:
    """重叠必须小于块边长。

    旧版这里不校验：overlap >= 块边长 时步长会被静默压成 1，
    于是一张 2048x2048 的图配 块尺寸256/重叠256 就能推导出上百万个图块，
    接口实测超时，服务被拖死。
    """
    if overlap >= long_edge:
        raise core.ValidationError(
            "重叠像素必须小于块边长。", field="overlap",
            detail=f"long_edge={long_edge}, overlap={overlap}")


def image_info(path: Path) -> dict:
    with Image.open(path) as img:
        return {
            "path": str(path),
            "filename": path.name,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format,
        }


def preview_path_for(path: Path, max_size: int = 2400) -> Path:
    key = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return PREVIEW_DIR / f"{key}_{max_size}.jpg"


def ensure_preview(path: Path, max_size: int = 2400) -> Path:
    out = preview_path_for(path, max_size)
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        img.save(out, "JPEG", quality=88, optimize=True)
    return out


def choose_grid(width: int, height: int, target: int, strict_count: bool = False) -> tuple[int, int]:
    target = max(1, int(target))
    tolerance = max(1, min(8, round(target * 0.16)))
    aspect = width / height
    best: tuple[float, int, int] | None = None
    for rows in range(1, target + 8):
        for cols in range(1, target + 8):
            pieces = rows * cols
            if strict_count:
                if pieces != target:
                    continue
            else:
                if pieces < max(1, target - tolerance) or pieces > target + tolerance:
                    continue
            tile_aspect = (width / cols) / (height / rows)
            story_bias = 0.0 if tile_aspect >= 0.85 else 0.35
            count_weight = 2.0 if strict_count else 1.0
            score = abs(pieces - target) / target * count_weight + abs(math.log(tile_aspect)) * 0.18 + story_bias
            candidate = (score, rows, cols)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        cols = max(1, round(math.sqrt(target * aspect)))
        rows = max(1, math.ceil(target / cols))
        return rows, cols
    return best[1], best[2]


def fixed_tile_size(long_edge: int) -> tuple[int, int]:
    return long_edge, long_edge


def fixed_coverage(cols: int, rows: int, tile_w: int, tile_h: int, overlap: int) -> tuple[int, int]:
    stride_w = max(1, tile_w - overlap)
    stride_h = max(1, tile_h - overlap)
    return tile_w + (cols - 1) * stride_w, tile_h + (rows - 1) * stride_h


def choose_fixed_grid(
    width: int,
    height: int,
    target: int,
    tile_w: int,
    tile_h: int,
    overlap: int,
    strict_count: bool = False,
) -> tuple[int, int]:
    target = max(1, int(target))
    tolerance = max(1, min(8, round(target * 0.16)))
    source_aspect = width / height
    best: tuple[float, int, int] | None = None
    max_n = target + 12
    for rows in range(1, max_n + 1):
        for cols in range(1, max_n + 1):
            pieces = rows * cols
            if strict_count:
                if pieces != target:
                    continue
            else:
                if pieces < max(1, target - tolerance) or pieces > target + tolerance:
                    continue
            cover_w, cover_h = fixed_coverage(cols, rows, tile_w, tile_h, overlap)
            if cover_w < width or cover_h < height:
                continue
            cover_aspect = cover_w / cover_h
            count_score = abs(pieces - target) / target
            aspect_score = abs(math.log(cover_aspect / source_aspect))
            blank_score = abs(cover_aspect / source_aspect - 1) * 0.2
            score = count_score * 0.65 + aspect_score * 0.35 + blank_score
            candidate = (score, rows, cols)
            if best is None or candidate < best:
                best = candidate
    if best:
        return best[1], best[2]
    stride_w = max(1, tile_w - overlap)
    stride_h = max(1, tile_h - overlap)
    min_cols = max(1, math.ceil((width - tile_w) / stride_w) + 1)
    min_rows = max(1, math.ceil((height - tile_h) / stride_h) + 1)
    return min_rows, min_cols


def build_plan(
    path: Path,
    target: int,
    long_edge: int,
    overlap: int,
    allow_upscale: bool = False,
    strict_count: bool = False,
    grid_shift_x: int = 0,
    grid_shift_y: int = 0,
) -> dict:
    info = image_info(path)
    tile_w, tile_h = fixed_tile_size(long_edge)
    stride_w = max(1, tile_w - overlap)
    stride_h = max(1, tile_h - overlap)
    grid_shift_x = int(grid_shift_x)
    grid_shift_y = int(grid_shift_y)
    offset_x = (-grid_shift_x) % stride_w if stride_w else 0
    offset_y = (-grid_shift_y) % stride_h if stride_h else 0
    cols = max(1, math.ceil(max(0, offset_x + info["width"] - tile_w) / stride_w) + 1)
    rows = max(1, math.ceil(max(0, offset_y + info["height"] - tile_h) / stride_h) + 1)
    cover_w, cover_h = fixed_coverage(cols, rows, tile_w, tile_h, overlap)
    scale = 1.0
    scaled_image_w = info["width"]
    scaled_image_h = info["height"]
    original_overlap = overlap
    tiles = []
    for row in range(rows):
        for col in range(cols):
            tile_x = col * stride_w
            tile_y = row * stride_h
            core_left = 0 if col == 0 else overlap
            core_top = 0 if row == 0 else overlap
            core_right = tile_w if col == cols - 1 else tile_w - overlap
            core_bottom = tile_h if row == rows - 1 else tile_h - overlap
            image_left = offset_x
            image_top = offset_y
            image_right = offset_x + scaled_image_w
            image_bottom = offset_y + scaled_image_h
            intersect_left = max(tile_x, image_left)
            intersect_top = max(tile_y, image_top)
            intersect_right = min(tile_x + tile_w, image_right)
            intersect_bottom = min(tile_y + tile_h, image_bottom)
            if intersect_right > intersect_left and intersect_bottom > intersect_top:
                x0 = max(0, round(intersect_left - offset_x))
                y0 = max(0, round(intersect_top - offset_y))
                x1 = min(info["width"], round(intersect_right - offset_x))
                y1 = min(info["height"], round(intersect_bottom - offset_y))
            else:
                x0 = y0 = x1 = y1 = 0
            tile_id = f"R{row + 1:02d}_C{col + 1:02d}"
            tiles.append(
                {
                    "tile_id": tile_id,
                    "row": row + 1,
                    "col": col + 1,
                    "core": [x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
                    "context": [x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
                    "scaled_context": [tile_w, tile_h],
                    "tile_rect": [tile_x, tile_y, tile_w, tile_h],
                    "source_rect": [x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
                    "source_paste_rect": [
                        round(intersect_left - tile_x) if intersect_right > intersect_left else 0,
                        round(intersect_top - tile_y) if intersect_bottom > intersect_top else 0,
                        round(intersect_right - intersect_left) if intersect_right > intersect_left else 0,
                        round(intersect_bottom - intersect_top) if intersect_bottom > intersect_top else 0,
                    ],
                    "core_output_rect": [
                        core_left,
                        core_top,
                        max(0, core_right - core_left),
                        max(0, core_bottom - core_top),
                    ],
                    "paste_rect": [
                        tile_x + core_left,
                        tile_y + core_top,
                        max(0, core_right - core_left),
                        max(0, core_bottom - core_top),
                    ],
                }
            )
    return {
        **info,
        "layout_mode": "fixed_3x4",
        "target_pieces": target,
        "strict_count": strict_count,
        "rows": rows,
        "cols": cols,
        "pieces": rows * cols,
        "long_edge": long_edge,
        "tile_width": tile_w,
        "tile_height": tile_h,
        "stride_width": stride_w,
        "stride_height": stride_h,
        "allow_upscale": allow_upscale,
        "overlap_output_px": overlap,
        "overlap_original_px": original_overlap,
        "scale": scale,
        "scale_percent": round(scale * 100, 2),
        "will_scale": False,
        "recommended_source_width": cover_w,
        "recommended_source_height": cover_h,
        "scaled_image_width": scaled_image_w,
        "scaled_image_height": scaled_image_h,
        "canvas_padding_x": offset_x,
        "canvas_padding_y": offset_y,
        "grid_shift_x": grid_shift_x,
        "grid_shift_y": grid_shift_y,
        "canvas_padding_right": cover_w - offset_x - scaled_image_w,
        "canvas_padding_bottom": cover_h - offset_y - scaled_image_h,
        "core_tile_width": tile_w,
        "core_tile_height": tile_h,
        "core_output_width": tile_w,
        "core_output_height": tile_h,
        "max_tile_output_width": max(tile["scaled_context"][0] for tile in tiles) if tiles else 0,
        "max_tile_output_height": max(tile["scaled_context"][1] for tile in tiles) if tiles else 0,
        "working_width": cover_w,
        "working_height": cover_h,
        "tiles": tiles,
    }


def write_manifest(plan: dict, out_dir: Path) -> tuple[Path, Path]:
    json_path = out_dir / "tiles_manifest.json"
    csv_path = out_dir / "tiles_manifest.csv"
    for tile in plan["tiles"]:
        core = tile["core"]
        tile["output_file"] = f"{tile['tile_id']}_x{core[0]}_y{core[1]}_v001.png"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    fields = [
        "tile_id",
        "row",
        "col",
        "core_x",
        "core_y",
        "core_w",
        "core_h",
        "context_x",
        "context_y",
        "context_w",
        "context_h",
        "scaled_w",
        "scaled_h",
        "output_file",
        "status",
        "owner",
        "version",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for tile in plan["tiles"]:
            core = tile["core"]
            context = tile["context"]
            scaled = tile["scaled_context"]
            output_file = tile["output_file"]
            writer.writerow(
                {
                    "tile_id": tile["tile_id"],
                    "row": tile["row"],
                    "col": tile["col"],
                    "core_x": core[0],
                    "core_y": core[1],
                    "core_w": core[2],
                    "core_h": core[3],
                    "context_x": context[0],
                    "context_y": context[1],
                    "context_w": context[2],
                    "context_h": context[3],
                    "scaled_w": scaled[0],
                    "scaled_h": scaled[1],
                    "output_file": output_file,
                    "status": "todo",
                    "owner": "",
                    "version": "v001",
                    "notes": "",
                }
            )
    return json_path, csv_path


def draw_locator_map(plan: dict, source_path: Path, out_dir: Path) -> Path:
    out_path = out_dir / "tile_locator_map.jpg"
    preview_scale = min(3200 / plan["working_width"], 1800 / plan["working_height"], 1.0)
    canvas_w = max(1, round(plan["working_width"] * preview_scale))
    canvas_h = max(1, round(plan["working_height"] * preview_scale))
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 242, 235))
    with Image.open(source_path) as img:
        img = img.convert("RGB")
        source_w = max(1, round(plan["width"] * preview_scale))
        source_h = max(1, round(plan["height"] * preview_scale))
        img = img.resize((source_w, source_h), Image.Resampling.LANCZOS)
        paste_x = round(plan["canvas_padding_x"] * preview_scale)
        paste_y = round(plan["canvas_padding_y"] * preview_scale)
        canvas.paste(img, (paste_x, paste_y))
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()
    for tile in plan["tiles"]:
        x, y, w, h = tile["tile_rect"]
        box = (
            round(x * preview_scale),
            round(y * preview_scale),
            round((x + w) * preview_scale),
            round((y + h) * preview_scale),
        )
        draw.rectangle(box, fill=(31, 111, 104, 24))
        dash = max(5, round(14 * preview_scale))
        gap = max(4, round(8 * preview_scale))
        for xx in range(box[0], box[2], dash + gap):
            draw.line((xx, box[1], min(xx + dash, box[2]), box[1]), fill=(255, 255, 255, 235), width=2)
            draw.line((xx, box[3], min(xx + dash, box[2]), box[3]), fill=(255, 255, 255, 235), width=2)
        for yy in range(box[1], box[3], dash + gap):
            draw.line((box[0], yy, box[0], min(yy + dash, box[3])), fill=(255, 255, 255, 235), width=2)
            draw.line((box[2], yy, box[2], min(yy + dash, box[3])), fill=(255, 255, 255, 235), width=2)
        label = tile.get("output_file") or tile["tile_id"]
        text_box = draw.textbbox((0, 0), label, font=font)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        tx = box[0] + 6
        ty = box[1] + 6
        if tx + tw + 6 > box[2]:
            label = tile["tile_id"]
            text_box = draw.textbbox((0, 0), label, font=font)
            tw = text_box[2] - text_box[0]
            th = text_box[3] - text_box[1]
        draw.rectangle((tx - 3, ty - 2, tx + tw + 3, ty + th + 2), fill=(0, 0, 0, 150))
        draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)
    canvas.save(out_path, "JPEG", quality=90, optimize=True)
    return out_path


def split_image(
    path: Path,
    target: int,
    long_edge: int,
    overlap: int,
    job_name: str,
    output_base: Path | None = None,
    allow_upscale: bool = False,
    strict_count: bool = False,
    grid_shift_x: int = 0,
    grid_shift_y: int = 0,
) -> dict:
    plan = build_plan(path, target, long_edge, overlap, allow_upscale, strict_count, grid_shift_x, grid_shift_y)
    base_dir = output_base if output_base else OUTPUT_DIR
    out_dir = base_dir / safe_name(job_name or f"{path.stem}_{plan['pieces']}tiles")
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as img:
        for tile in plan["tiles"]:
            core = tile["core"]
            output_file = f"{tile['tile_id']}_x{core[0]}_y{core[1]}_v001.png"
            if plan.get("layout_mode") == "fixed_3x4":
                tile_canvas = Image.new("RGB", tuple(tile["scaled_context"]), (245, 242, 235))
                source = tile["source_rect"]
                paste = tile["source_paste_rect"]
                if source[2] > 0 and source[3] > 0 and paste[2] > 0 and paste[3] > 0:
                    crop_box = (
                        source[0],
                        source[1],
                        source[0] + source[2],
                        source[1] + source[3],
                    )
                    cropped = img.crop(crop_box).convert("RGB")
                    if cropped.size != (paste[2], paste[3]):
                        raise RuntimeError(
                            f"{tile['tile_id']} 裁切尺寸 {cropped.size} 与粘贴区域 {(paste[2], paste[3])} 不一致"
                        )
                    tile_canvas.paste(cropped, (paste[0], paste[1]))
                tile_canvas.save(tiles_dir / output_file, "PNG")
            else:
                context = tile["context"]
                crop_box = (
                    context[0],
                    context[1],
                    context[0] + context[2],
                    context[1] + context[3],
                )
                cropped = img.crop(crop_box).convert("RGB")
                scaled_size = tile["scaled_context"]
                if cropped.size != tuple(scaled_size):
                    cropped = cropped.resize(tuple(scaled_size), Image.Resampling.LANCZOS)
                cropped.save(tiles_dir / output_file, "PNG")
            tile["output_file"] = output_file
    manifest_json, manifest_csv = write_manifest(plan, out_dir)
    locator_map = draw_locator_map(plan, path, out_dir)
    return {
        "out_dir": str(out_dir),
        "tiles_dir": str(tiles_dir),
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
        "locator_map": str(locator_map),
        "locator_preview_url": f"/api/preview?path={quote(str(locator_map))}",
        "plan": plan,
    }


def build_grid_plan(path: Path, rows: int, cols: int) -> dict:
    info = image_info(path)
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    tiles = []
    for row in range(rows):
        y0 = round(info["height"] * row / rows)
        y1 = round(info["height"] * (row + 1) / rows)
        for col in range(cols):
            x0 = round(info["width"] * col / cols)
            x1 = round(info["width"] * (col + 1) / cols)
            w = max(0, x1 - x0)
            h = max(0, y1 - y0)
            tile_id = f"R{row + 1:02d}_C{col + 1:02d}"
            tiles.append(
                {
                    "tile_id": tile_id,
                    "row": row + 1,
                    "col": col + 1,
                    "core": [x0, y0, w, h],
                    "context": [x0, y0, w, h],
                    "scaled_context": [w, h],
                    "tile_rect": [x0, y0, w, h],
                    "source_rect": [x0, y0, w, h],
                    "source_paste_rect": [0, 0, w, h],
                    "core_output_rect": [0, 0, w, h],
                    "paste_rect": [x0, y0, w, h],
                }
            )
    return {
        **info,
        "layout_mode": "manual_grid",
        "target_pieces": rows * cols,
        "strict_count": True,
        "rows": rows,
        "cols": cols,
        "pieces": rows * cols,
        "long_edge": 0,
        "tile_width": max(tile["scaled_context"][0] for tile in tiles) if tiles else 0,
        "tile_height": max(tile["scaled_context"][1] for tile in tiles) if tiles else 0,
        "stride_width": 0,
        "stride_height": 0,
        "allow_upscale": False,
        "overlap_output_px": 0,
        "overlap_original_px": 0,
        "scale": 1.0,
        "scale_percent": 100,
        "will_scale": False,
        "recommended_source_width": info["width"],
        "recommended_source_height": info["height"],
        "scaled_image_width": info["width"],
        "scaled_image_height": info["height"],
        "canvas_padding_x": 0,
        "canvas_padding_y": 0,
        "canvas_padding_right": 0,
        "canvas_padding_bottom": 0,
        "core_tile_width": max(tile["scaled_context"][0] for tile in tiles) if tiles else 0,
        "core_tile_height": max(tile["scaled_context"][1] for tile in tiles) if tiles else 0,
        "core_output_width": max(tile["scaled_context"][0] for tile in tiles) if tiles else 0,
        "core_output_height": max(tile["scaled_context"][1] for tile in tiles) if tiles else 0,
        "max_tile_output_width": max(tile["scaled_context"][0] for tile in tiles) if tiles else 0,
        "max_tile_output_height": max(tile["scaled_context"][1] for tile in tiles) if tiles else 0,
        "working_width": info["width"],
        "working_height": info["height"],
        "tiles": tiles,
    }


def split_image_grid(
    path: Path,
    rows: int,
    cols: int,
    job_name: str,
    output_base: Path | None = None,
) -> dict:
    plan = build_grid_plan(path, rows, cols)
    base_dir = output_base if output_base else OUTPUT_DIR
    out_dir = base_dir / safe_name(job_name or f"{path.stem}_{plan['rows']}x{plan['cols']}")
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as img:
        for tile in plan["tiles"]:
            core = tile["core"]
            output_file = f"{tile['tile_id']}_x{core[0]}_y{core[1]}_v001.png"
            crop_box = (core[0], core[1], core[0] + core[2], core[1] + core[3])
            img.crop(crop_box).convert("RGB").save(tiles_dir / output_file, "PNG")
            tile["output_file"] = output_file
    manifest_json, manifest_csv = write_manifest(plan, out_dir)
    locator_map = draw_locator_map(plan, path, out_dir)
    return {
        "out_dir": str(out_dir),
        "tiles_dir": str(tiles_dir),
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
        "locator_map": str(locator_map),
        "locator_preview_url": f"/api/preview?path={quote(str(locator_map))}",
        "plan": plan,
    }


def resize_export(
    path: Path,
    edge_mode: str,
    target_px: int,
    scale_mode: str,
    scale_factor: float,
    output_format: str,
    quality: int,
    allow_upscale: bool,
    job_name: str,
    output_base: Path | None = None,
    out_name: str = "",
) -> dict:
    target_px = max(1, int(target_px))
    scale_mode = scale_mode if scale_mode in {"edge", "factor"} else "edge"
    scale_factor = max(0.01, min(16.0, float(scale_factor or 1.0)))
    quality = max(1, min(100, int(quality)))
    edge_mode = edge_mode if edge_mode in {"long", "short"} else "long"
    output_format = (output_format or "png").lower()
    if output_format not in {"png", "jpg", "jpeg", "webp", "tif", "tiff"}:
        output_format = "png"

    with Image.open(path) as img:
        source_w, source_h = img.size
        source_long = max(source_w, source_h)
        source_short = min(source_w, source_h)
        if scale_mode == "factor":
            scale = scale_factor
        else:
            basis = source_long if edge_mode == "long" else source_short
            scale = target_px / basis
        out_w = max(1, round(source_w * scale))
        out_h = max(1, round(source_h * scale))
        resized = img.copy()
        if (out_w, out_h) != img.size:
            resized = resized.resize((out_w, out_h), Image.Resampling.LANCZOS)

        ext = "jpg" if output_format == "jpeg" else output_format
        save_format = {"jpg": "JPEG", "tif": "TIFF"}.get(ext, ext.upper())
        base_dir = output_base if output_base else OUTPUT_DIR
        name_suffix = f"{scale_factor:g}x" if scale_mode == "factor" else f"{edge_mode}_{target_px}px"
        out_dir = base_dir / safe_name(job_name or f"{path.stem}_{name_suffix}")
        out_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"{path.stem}_{name_suffix}.{ext}"
        out_path = out_dir / safe_name(out_name or default_name)
        if out_path.suffix.lower() != f".{ext}":
            out_path = out_path.with_suffix(f".{ext}")

        save_kwargs = {}
        if save_format in {"JPEG", "WEBP"}:
            save_kwargs["quality"] = quality
        if save_format == "JPEG":
            save_kwargs["optimize"] = True
            save_img = resized.convert("RGB")
        else:
            save_img = resized
        save_img.save(out_path, save_format, **save_kwargs)

    return {
        "out_dir": str(out_dir),
        "out_path": str(out_path),
        "preview_url": f"/api/preview?path={quote(str(out_path))}",
        "source_width": source_w,
        "source_height": source_h,
        "width": out_w,
        "height": out_h,
        "edge_mode": edge_mode,
        "target_px": target_px,
        "scale_mode": scale_mode,
        "scale_factor": scale_factor,
        "scale": scale,
        "format": ext,
        "quality": quality,
        "allow_upscale": True,
    }


def find_tile_file(restored_dir: Path, output_file: str, tile_id: str) -> Path | None:
    exact = restored_dir / output_file if output_file else None
    if exact and exact.is_file():
        return exact
    matches = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.webp"):
        matches.extend(restored_dir.glob(f"*{tile_id}*{ext[1:]}"))
    matches = sorted(matches)
    return matches[-1] if matches else None


def parse_selected_tiles(raw: str | list[str] | None) -> set[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        selected = raw
    else:
        selected = re.split(r"[\s,;，；]+", raw.strip())
    cleaned = {item.strip().upper() for item in selected if item.strip()}
    return cleaned or None


def parse_tile_ids(raw: str | list[str] | None) -> list[str]:
    selected = parse_selected_tiles(raw)
    return sorted(selected) if selected else []


def parse_paths(raw: str | list[str] | None) -> list[Path]:
    if not raw:
        return []
    values = raw if isinstance(raw, list) else re.split(r"[;；\n]+", raw)
    return [resolve_image_path(value) for value in values if str(value).strip()]


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tile_lookup(plan: dict) -> dict[str, dict]:
    return {tile["tile_id"].upper(): tile for tile in plan["tiles"]}


def next_patch_run_dir(manifest_path: Path, output_base: Path | None = None, name_hint: str = "") -> Path:
    root = output_base if output_base else manifest_path.parent / "patch_runs"
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for item in root.glob("run_*"):
        match = re.match(r"run_(\d+)", item.name)
        if match:
            existing.append(int(match.group(1)))
    run_no = max(existing, default=0) + 1
    suffix = safe_name(name_hint) if name_hint else ""
    name = f"run_{run_no:03d}" + (f"_{suffix}" if suffix else "")
    return root / name


def copy_tiles(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
            target = dst_dir / path.name
            with path.open("rb") as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)


def build_patch_region(plan: dict, tiles_dir: Path, candidate_ids: list[str]) -> tuple[Image.Image, dict, list[dict]]:
    lookup = tile_lookup(plan)
    missing_ids = [tile_id for tile_id in candidate_ids if tile_id.upper() not in lookup]
    if missing_ids:
        raise RuntimeError(f"Unknown candidate tiles: {', '.join(missing_ids)}")
    selected = [lookup[tile_id.upper()] for tile_id in candidate_ids]
    if not selected:
        raise RuntimeError("Please select at least one candidate tile.")
    min_x = min(tile["tile_rect"][0] for tile in selected)
    min_y = min(tile["tile_rect"][1] for tile in selected)
    max_x = max(tile["tile_rect"][0] + tile["tile_rect"][2] for tile in selected)
    max_y = max(tile["tile_rect"][1] + tile["tile_rect"][3] for tile in selected)
    canvas = Image.new("RGB", (max_x - min_x, max_y - min_y), (245, 242, 235))
    loaded = []
    for tile in selected:
        tile_file = find_tile_file(tiles_dir, tile.get("output_file", ""), tile["tile_id"])
        if tile_file is None:
            raise RuntimeError(f"Missing tile file for {tile['tile_id']} in {tiles_dir}")
        with Image.open(tile_file) as img:
            tile_img = img.convert("RGB")
            expected = (tile["tile_rect"][2], tile["tile_rect"][3])
            if tile_img.size != expected:
                tile_img = tile_img.resize(expected, Image.Resampling.LANCZOS)
            canvas.paste(tile_img, (tile["tile_rect"][0] - min_x, tile["tile_rect"][1] - min_y))
        loaded.append({"tile_id": tile["tile_id"], "path": str(tile_file)})
    region = {"x": min_x, "y": min_y, "width": canvas.width, "height": canvas.height}
    return canvas, region, loaded


def cv_gray(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def cv_template_match(region: Image.Image, patch: Image.Image) -> dict:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Please install opencv-python for Patch Align.")
    if patch.width > region.width or patch.height > region.height:
        raise RuntimeError("Patch image is larger than the selected tile region.")
    region_gray = cv_gray(region)
    patch_gray = cv_gray(patch)
    gray_scores = cv2.matchTemplate(region_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
    _, gray_score, _, gray_loc = cv2.minMaxLoc(gray_scores)
    region_edge = cv2.Canny(region_gray, 80, 160)
    patch_edge = cv2.Canny(patch_gray, 80, 160)
    if np.count_nonzero(patch_edge) > 12:
        edge_scores = cv2.matchTemplate(region_edge, patch_edge, cv2.TM_CCOEFF_NORMED)
        _, edge_score, _, edge_loc = cv2.minMaxLoc(edge_scores)
    else:
        edge_score, edge_loc = -1.0, gray_loc
    if edge_score > gray_score + 0.04:
        x, y = edge_loc
        method = "edge"
        score = float(edge_score)
    else:
        x, y = gray_loc
        method = "gray"
        score = float(gray_score)
    return {"x": int(x), "y": int(y), "score": round(score, 4), "method": method}


def draw_patch_preview(region: Image.Image, patch: Image.Image, match: dict, out_path: Path) -> Path:
    preview = region.convert("RGBA")
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    patch_rgba = patch.convert("RGBA")
    patch_rgba.putalpha(138)
    overlay.alpha_composite(patch_rgba, (match["x"], match["y"]))
    preview = Image.alpha_composite(preview, overlay)
    draw = ImageDraw.Draw(preview, "RGBA")
    box = (match["x"], match["y"], match["x"] + patch.width, match["y"] + patch.height)
    draw.rectangle(box, outline=(255, 0, 0, 255), width=5)
    label = f"{match['method']} score {match['score']}"
    font = ImageFont.load_default()
    text_box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((box[0], max(0, box[1] - 22), box[0] + text_box[2] + 10, max(22, box[1])), fill=(0, 0, 0, 170))
    draw.text((box[0] + 5, max(3, box[1] - 18)), label, fill=(255, 255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.convert("RGB").save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


def make_tile_selector_map(manifest_path: Path, tiles_dir: Path, max_w: int = 2600, max_h: int = 1700) -> dict:
    plan = load_manifest(manifest_path)
    row_by_y = {value: index + 1 for index, value in enumerate(sorted({tile["tile_rect"][1] for tile in plan["tiles"]}))}
    col_by_x = {value: index + 1 for index, value in enumerate(sorted({tile["tile_rect"][0] for tile in plan["tiles"]}))}

    def tile_grid_position(tile: dict) -> tuple[int, int]:
        if tile.get("row") is not None and tile.get("col") is not None:
            return int(tile["row"]), int(tile["col"])
        match = re.fullmatch(r"R0*(\d+)_C0*(\d+)", str(tile.get("tile_id", "")), flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
        x, y = tile["tile_rect"][:2]
        return row_by_y[y], col_by_x[x]

    scale = min(max_w / plan["working_width"], max_h / plan["working_height"], 1.0)
    canvas_w = max(1, round(plan["working_width"] * scale))
    canvas_h = max(1, round(plan["working_height"] * scale))
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 242, 235))
    loaded = []
    for tile in plan["tiles"]:
        tile_file = find_tile_file(tiles_dir, tile.get("output_file", ""), tile["tile_id"])
        if tile_file is None:
            continue
        with Image.open(tile_file) as img:
            tile_img = img.convert("RGB")
            tile_img.thumbnail(
                (max(1, round(tile["tile_rect"][2] * scale)), max(1, round(tile["tile_rect"][3] * scale))),
                Image.Resampling.LANCZOS,
            )
            canvas.paste(tile_img, (round(tile["tile_rect"][0] * scale), round(tile["tile_rect"][1] * scale)))
        loaded.append(tile["tile_id"])
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()
    tiles = []
    for tile in plan["tiles"]:
        row, col = tile_grid_position(tile)
        x, y, w, h = tile["tile_rect"]
        box = (
            round(x * scale),
            round(y * scale),
            round((x + w) * scale),
            round((y + h) * scale),
        )
        draw.rectangle(box, outline=(255, 255, 255, 230), width=2)
        draw.rectangle(box, fill=(127, 29, 29, 20))
        label = tile["tile_id"]
        draw.rectangle((box[0] + 4, box[1] + 4, box[0] + 78, box[1] + 22), fill=(0, 0, 0, 150))
        draw.text((box[0] + 8, box[1] + 7), label, fill=(255, 255, 255, 255), font=font)
        tiles.append({
            "tile_id": tile["tile_id"],
            "row": row,
            "col": col,
            "x": box[0],
            "y": box[1],
            "w": max(1, box[2] - box[0]),
            "h": max(1, box[3] - box[1]),
        })
    out_dir = manifest_path.parent / "patch_previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tile_selector_map.jpg"
    canvas.save(out_path, "JPEG", quality=90, optimize=True)
    return {
        "preview_path": str(out_path),
        "preview_url": f"/api/preview?path={quote(str(out_path))}",
        "width": canvas_w,
        "height": canvas_h,
        "scale": scale,
        "tiles": tiles,
        "loaded_tiles": loaded,
    }


def patch_align(
    manifest_path: Path,
    tiles_dir: Path,
    patch_path: Path,
    candidate_ids: list[str],
    output_base: Path | None = None,
    run_name: str = "",
    apply_patch: bool = False,
    min_score: float = 0.28,
) -> dict:
    plan = load_manifest(manifest_path)
    region_img, region, loaded_tiles = build_patch_region(plan, tiles_dir, candidate_ids)
    with Image.open(patch_path) as patch_src:
        patch_img = patch_src.convert("RGB")
    match = cv_template_match(region_img, patch_img)
    run_dir = next_patch_run_dir(manifest_path, output_base, run_name) if apply_patch else (manifest_path.parent / "patch_previews")
    preview_path = draw_patch_preview(region_img, patch_img, match, run_dir / "patch_preview.jpg")
    affected_tiles = []
    output_tiles_dir = ""
    record_path = ""
    if apply_patch:
        if match["score"] < min_score:
            raise RuntimeError(f"Patch confidence too low: {match['score']}. Preview only; please choose a tighter tile region.")
        output_tiles = run_dir / "tiles"
        copy_tiles(tiles_dir, output_tiles)
        region_img.paste(patch_img, (match["x"], match["y"]))
        lookup = tile_lookup(plan)
        for tile_id in candidate_ids:
            tile = lookup[tile_id.upper()]
            tx = tile["tile_rect"][0] - region["x"]
            ty = tile["tile_rect"][1] - region["y"]
            box = (tx, ty, tx + tile["tile_rect"][2], ty + tile["tile_rect"][3])
            updated_tile = region_img.crop(box)
            out_file = tile.get("output_file") or f"{tile['tile_id']}.png"
            updated_tile.save(output_tiles / out_file, "PNG")
            affected_tiles.append(tile["tile_id"])
        output_tiles_dir = str(output_tiles)
        record = {
            "manifest_path": str(manifest_path),
            "source_tiles_dir": str(tiles_dir),
            "patch_image": str(patch_path),
            "candidate_tiles": candidate_ids,
            "affected_tiles": affected_tiles,
            "region": region,
            "match": match,
            "output_tiles_dir": output_tiles_dir,
            "loaded_tiles": loaded_tiles,
        }
        record_path_obj = run_dir / "patch_record.json"
        with record_path_obj.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        record_path = str(record_path_obj)
    return {
        "run_dir": str(run_dir),
        "preview_path": str(preview_path),
        "preview_url": f"/api/preview?path={quote(str(preview_path))}",
        "region": region,
        "candidate_tiles": candidate_ids,
        "match": match,
        "patch_size": [patch_img.width, patch_img.height],
        "output_tiles_dir": output_tiles_dir,
        "record_path": record_path,
        "affected_tiles": affected_tiles,
        "opencv_available": cv2 is not None,
    }


def rgb_stats(img: Image.Image) -> tuple[list[float], list[float]]:
    stat = ImageStat.Stat(img.convert("RGB"))
    mean = list(stat.mean[:3])
    std = [max(1.0, value) for value in stat.stddev[:3]]
    return mean, std


def match_color_stats(img: Image.Image, target_mean: list[float], target_std: list[float]) -> Image.Image:
    src_mean, src_std = rgb_stats(img)
    channels = img.convert("RGB").split()
    adjusted = []
    for idx, channel in enumerate(channels):
        gain = target_std[idx] / max(1.0, src_std[idx])
        bias = target_mean[idx] - src_mean[idx] * gain
        adjusted.append(channel.point(lambda px, gain=gain, bias=bias: max(0, min(255, int(px * gain + bias)))))
    return Image.merge("RGB", adjusted)


def lab_stats(img: Image.Image) -> tuple[list[float], list[float]]:
    stat = ImageStat.Stat(img.convert("RGB").convert("LAB"))
    mean = list(stat.mean[:3])
    std = [max(1.0, value) for value in stat.stddev[:3]]
    return mean, std


def match_lab_reinhard(img: Image.Image, target_mean: list[float], target_std: list[float]) -> Image.Image:
    lab = img.convert("RGB").convert("LAB")
    src_mean, src_std = lab_stats(img)
    channels = lab.split()
    adjusted = []
    for idx, channel in enumerate(channels):
        gain = target_std[idx] / max(1.0, src_std[idx])
        bias = target_mean[idx] - src_mean[idx] * gain
        adjusted.append(channel.point(lambda px, gain=gain, bias=bias: max(0, min(255, int(px * gain + bias)))))
    return Image.merge("LAB", adjusted).convert("RGB")


def match_tile_color(img: Image.Image, reference: Image.Image, color_mode: str) -> Image.Image:
    if color_mode == "lab_reinhard":
        return match_lab_reinhard(img, *lab_stats(reference))
    if color_mode == "rgb_stats":
        return match_color_stats(img, *rgb_stats(reference))
    return img


def rounded_rgb_mean(img: Image.Image) -> list[int]:
    return [round(value) for value in rgb_stats(img)[0]]


def make_color_comparison(samples: list[dict], out_dir: Path) -> tuple[Path | None, list[dict]]:
    if not samples:
        return None, []
    shown = samples[:12]
    thumb_w = 520
    thumb_h = 340
    label_h = 56
    gap = 16
    cols = 3
    row_w = cols * thumb_w + (cols - 1) * gap
    row_h = thumb_h + label_h + gap
    canvas = Image.new("RGB", (row_w, len(shown) * row_h + 8), (24, 25, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    item_dir = out_dir / "color_compare_tiles"
    item_dir.mkdir(parents=True, exist_ok=True)
    items = []

    def fit(img: Image.Image) -> Image.Image:
        tile = Image.new("RGB", (thumb_w, thumb_h), (15, 16, 18))
        copy = img.convert("RGB").copy()
        copy.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (thumb_w - copy.width) // 2
        y = (thumb_h - copy.height) // 2
        tile.paste(copy, (x, y))
        return tile

    for row, sample in enumerate(shown):
        y = row * row_h + 8
        columns = [
            ("before", sample["before"]),
            ("after", sample["after"]),
            ("reference", sample["reference"]),
        ]
        item_canvas = Image.new("RGB", (row_w, thumb_h + label_h), (24, 25, 28))
        item_draw = ImageDraw.Draw(item_canvas)
        for col, (label, img) in enumerate(columns):
            x = col * (thumb_w + gap)
            fitted = fit(img)
            canvas.paste(fitted, (x, y + label_h))
            item_canvas.paste(fitted, (x, label_h))
            draw.text((x + 6, y + 4), f"{sample['tile_id']} {label}", fill=(240, 238, 230), font=font)
            item_draw.text((x + 6, 4), f"{sample['tile_id']} {label}", fill=(240, 238, 230), font=font)
            if label == "before":
                mean = sample["before_mean"]
            elif label == "after":
                mean = sample["after_mean"]
            else:
                mean = sample["reference_mean"]
            draw.text((x + 6, y + 20), f"RGB {mean}", fill=(170, 170, 165), font=font)
            item_draw.text((x + 6, 20), f"RGB {mean}", fill=(170, 170, 165), font=font)
        item_path = item_dir / f"{sample['tile_id']}.jpg"
        item_canvas.save(item_path, "JPEG", quality=92, optimize=True)
        items.append(
            {
                "tile_id": sample["tile_id"],
                "path": str(item_path),
                "preview_url": f"/api/preview?path={quote(str(item_path))}",
                "before_rgb_mean": sample["before_mean"],
                "after_rgb_mean": sample["after_mean"],
                "reference_rgb_mean": sample["reference_mean"],
                "delta_rgb_mean": [sample["after_mean"][i] - sample["before_mean"][i] for i in range(3)],
            }
        )
    out = out_dir / "color_compare.jpg"
    canvas.save(out, "JPEG", quality=92, optimize=True)
    return out, items


def comparison_thumbnail(image: Image.Image) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail((520, 340), Image.Resampling.LANCZOS)
    return copy


def source_reference_for_tile(plan: dict, tile: dict, source_img: Image.Image | None) -> Image.Image | None:
    if source_img is None:
        return None
    rect = tile.get("source_rect") or tile.get("context") or [0, 0, 0, 0]
    if rect[2] <= 0 or rect[3] <= 0:
        return None
    crop = source_img.crop((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])).convert("RGB")
    expected = tuple(tile.get("scaled_context", [crop.width, crop.height]))
    if crop.size != expected:
        crop = crop.resize(expected, Image.Resampling.LANCZOS)
    return crop


SEAM_BALANCE_PRESETS = {
    "light":    {"strip": 20, "inset": 80, "regularize": 0.50, "max_bias": 10.0, "feather": 80},
    "standard": {"strip": 20, "inset": 80, "regularize": 0.35, "max_bias": 18.0, "feather": 80},
    "strong":   {"strip": 20, "inset": 80, "regularize": 0.25, "max_bias": 24.0, "feather": 120},
    "palette_residual": {"strip": 20, "inset": 80, "regularize": 0.85, "max_bias": 6.0, "feather": 20},
}


def strip_median_rgb(image: Image.Image, side: str, strip_px: int, inset_px: int) -> np.ndarray:
    width, height = image.size
    x0, y0, x1, y1 = 0, 0, width, height
    if side == "left":
        x1 = min(strip_px, width)
    elif side == "right":
        x0 = max(0, width - strip_px)
    elif side == "top":
        y1 = min(strip_px, height)
    elif side == "bottom":
        y0 = max(0, height - strip_px)
    if side in {"left", "right"} and height > inset_px * 2:
        y0, y1 = inset_px, height - inset_px
    if side in {"top", "bottom"} and width > inset_px * 2:
        x0, x1 = inset_px, width - inset_px
    arr = np.asarray(image.crop((x0, y0, x1, y1)), dtype=np.float32)
    return np.median(arr.reshape(-1, 3), axis=0)


def solve_seam_biases(
    tile_images: list[tuple[str, Image.Image, dict]],
    rows: int,
    cols: int,
    strip_px: int,
    inset_px: int,
    regularize: float,
    allowed_pairs: set[tuple[str, str]] | None = None,
    fixed_zero_ids: set[str] | None = None,
) -> dict[str, np.ndarray]:
    by_pos = {(int(t[2].get("row", 1)), int(t[2].get("col", 1))): t for t in tile_images}
    all_ids = [t[0] for t in tile_images]
    fixed_zero_ids = fixed_zero_ids or set()
    ids = [tid for tid in all_ids if tid not in fixed_zero_ids]
    index = {tid: i for i, tid in enumerate(ids)}
    equations: list[tuple[str, str]] = []
    targets: list[np.ndarray] = []
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            item = by_pos.get((row, col))
            if item is None:
                continue
            right = by_pos.get((row, col + 1))
            if right is not None and (allowed_pairs is None or (item[0], right[0]) in allowed_pairs):
                mean_a = strip_median_rgb(item[1], "right", strip_px, inset_px)
                mean_b = strip_median_rgb(right[1], "left", strip_px, inset_px)
                equations.append((item[0], right[0]))
                targets.append(mean_b - mean_a)
            bottom = by_pos.get((row + 1, col))
            if bottom is not None and (allowed_pairs is None or (item[0], bottom[0]) in allowed_pairs):
                mean_a = strip_median_rgb(item[1], "bottom", strip_px, inset_px)
                mean_b = strip_median_rgb(bottom[1], "top", strip_px, inset_px)
                equations.append((item[0], bottom[0]))
                targets.append(mean_b - mean_a)
    n = len(ids)
    if n == 0:
        return {tid: np.zeros(3, dtype=np.float32) for tid in all_ids}
    mat = np.zeros((len(equations) + n + 1, n), dtype=np.float64)
    rhs = np.zeros((len(equations) + n + 1, 3), dtype=np.float64)
    for i, ((lid, rid), target) in enumerate(zip(equations, targets)):
        if lid in index:
            mat[i, index[lid]] = 1.0
        if rid in index:
            mat[i, index[rid]] = -1.0
        rhs[i] = target
    for off in range(n):
        mat[len(equations) + off, off] = regularize
    if not fixed_zero_ids:
        mat[-1, :] = 1.0
    solution, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
    return {
        tid: solution[index[tid]].astype(np.float32) if tid in index else np.zeros(3, dtype=np.float32)
        for tid in all_ids
    }


def apply_tile_bias(image: Image.Image, bias: np.ndarray, max_abs: float) -> Image.Image:
    bias = np.clip(bias, -max_abs, max_abs)
    arr = np.asarray(image, dtype=np.float32) + bias.reshape(1, 1, 3)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def low_frequency_correct_placements(
    placements: list[tuple[str, int, int, Image.Image, dict]],
    plan: dict,
    strength: float = 0.5,
    mode: str = "auto",
    small_width: int = 600,
) -> list[tuple[str, int, int, Image.Image, dict]]:
    """逐块低频校正：先用缩略图估算低频校正场，再放大校正场应用到每块全分辨率原图。
    校正场是平滑的低频场，相邻块的校正值在边界处几乎一致，不会引入新接缝。
    原图全分辨率细节完全保留，只修正大面积明暗/色温漂移。"""
    lightness_strength, temperature_strength = resolve_low_frequency_strengths(mode, strength)
    if (lightness_strength <= 0 and temperature_strength <= 0) or not placements:
        return placements
    # Use the real placement extent. With trim_padding disabled, the last tile can
    # extend beyond the source width/height into the manifest's padded work area.
    canvas_w = max(int(plan["width"]), max(px + img.width for _, px, _, img, _ in placements))
    canvas_h = max(int(plan["height"]), max(py + img.height for _, _, py, img, _ in placements))
    scale = small_width / canvas_w
    small_w = small_width
    small_h = max(1, round(canvas_h * scale))

    # 1. 用缩略图快速拼一个小图（用于估算低频场）
    weight_sum = np.zeros((small_h, small_w), dtype=np.float32)
    color_sum = np.zeros((small_h, small_w, 3), dtype=np.float32)
    for tid, px, py, img, tile in placements:
        tw, th = img.size
        stw = max(1, round(tw * scale))
        sth = max(1, round(th * scale))
        small_tile = img.convert("RGB").resize((stw, sth), Image.Resampling.LANCZOS)
        stx = round(px * scale)
        sty = round(py * scale)
        arr = np.asarray(small_tile, dtype=np.float32)
        h_s, w_s = arr.shape[:2]
        if sty + h_s > small_h:
            h_s = small_h - sty
            arr = arr[:h_s]
        if stx + w_s > small_w:
            w_s = small_w - stx
            arr = arr[:, :w_s]
        weight_sum[sty:sty + h_s, stx:stx + w_s] += 1.0
        color_sum[sty:sty + h_s, stx:stx + w_s] += arr
    result_small = color_sum / np.maximum(weight_sum[:, :, None], 1.0)
    small_img = Image.fromarray(np.clip(result_small, 0, 255).astype(np.uint8), "RGB")

    # 2. 在小图上算低频校正场（LAB 空间，大半径高斯模糊）
    small_lab = small_img.convert("LAB")
    radius = max(10, small_w // 4)
    correction_fields_full = []
    for index, ch in enumerate(small_lab.split()):
        arr = np.asarray(ch, dtype=np.float32)
        blurred = Image.fromarray(arr.astype(np.uint8), "L").filter(
            ImageFilter.GaussianBlur(radius=radius)
        )
        blurred_arr = np.asarray(blurred, dtype=np.float32)
        global_mean = float(blurred_arr.mean())
        channel_strength = lightness_strength if index == 0 else temperature_strength
        correction_small = (blurred_arr - global_mean) * channel_strength
        # 放大校正场到全尺寸（校正场是平滑的，放大不会丢信息）
        corr_img = Image.fromarray(correction_small, "F")
        corr_full = corr_img.resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        correction_fields_full.append(np.asarray(corr_full, dtype=np.float32))

    # 3. 把校正场应用到每块全分辨率原图
    new_placements = []
    for tid, px, py, img, tile in placements:
        tw, th = img.size
        lab = img.convert("RGB").convert("LAB")
        channels = lab.split()
        corrected_channels = []
        for idx, ch in enumerate(channels):
            ch_arr = np.asarray(ch, dtype=np.float32)
            corr_region = correction_fields_full[idx][py:py + th, px:px + tw]
            if corr_region.shape != ch_arr.shape:
                corr_region = corr_region[:ch_arr.shape[0], :ch_arr.shape[1]]
            corrected = np.clip(ch_arr - corr_region, 0, 255)
            corrected_channels.append(Image.fromarray(corrected.astype(np.uint8), "L"))
        corrected_lab = Image.merge("LAB", corrected_channels)
        corrected_rgb = corrected_lab.convert("RGB")
        new_placements.append((tid, px, py, corrected_rgb, tile))
    return new_placements


def feather_mask(size: tuple[int, int], tile: dict, plan: dict, feather_px: int) -> Image.Image:
    width, height = size
    feather_px = max(0, min(feather_px, width, height))
    if feather_px <= 0:
        return Image.new("L", size, 255)
    row = int(tile.get("row", 1))
    col = int(tile.get("col", 1))
    rows = int(plan.get("rows", 1))
    cols = int(plan.get("cols", 1))
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    mask = np.full((height, width), 255.0, dtype=np.float32)
    denom = max(1, feather_px - 1)
    if col > 1:
        left = np.clip(255.0 * x / denom, 0, 255)
        mask = np.minimum(mask, left[np.newaxis, :])
    if col < cols:
        right = np.clip(255.0 * (width - 1 - x) / denom, 0, 255)
        mask = np.minimum(mask, right[np.newaxis, :])
    if row > 1:
        top = np.clip(255.0 * y / denom, 0, 255)
        mask = np.minimum(mask, top[:, np.newaxis])
    if row < rows:
        bottom = np.clip(255.0 * (height - 1 - y) / denom, 0, 255)
        mask = np.minimum(mask, bottom[:, np.newaxis])
    return Image.fromarray(mask.astype(np.uint8), "L")


def apply_structure_recolor_placements(
    placements: list[tuple[str, int, int, Image.Image, dict]],
    palette_profile: dict,
    palette_strength: float,
    ink_lightness: float,
    saturation_gain: float = 1.0,
    ink_expand_px: int = 2,
    preserve_local_hue: bool = True,
) -> tuple[list[tuple[str, int, int, Image.Image, dict]], dict]:
    if len(placements) > MAX_STRUCTURE_RECOLOR_TILES:
        raise ValueError(
            f"统一灰阶与墨线实验最多处理 {MAX_STRUCTURE_RECOLOR_TILES} 块，请先用‘选择图块’限定范围"
        )
    if not placements:
        return [], {"tiles": {}, "ink_lightness": ink_lightness}
    min_x = min(item[1] for item in placements)
    min_y = min(item[2] for item in placements)
    max_x = max(item[1] + item[3].width for item in placements)
    max_y = max(item[2] + item[3].height for item in placements)
    layers = {tile_id: decompose_tile(image) for tile_id, _, _, image, _ in placements}
    light_placements = [
        (tile_id, x - min_x, y - min_y, layers[tile_id].structure_l)
        for tile_id, x, y, _, _ in placements
    ]
    normalized, report = joint_grayscale_normalize(
        light_placements,
        (max_x - min_x, max_y - min_y),
        overlap=20,
        ink_lightness=ink_lightness,
        ink_expand_px=ink_expand_px,
    )
    lights = {item[0]: item[3] for item in normalized}
    processed = [
        (tile_id, x, y, recolor_tile(layers[tile_id], lights[tile_id], palette_profile, palette_strength, saturation_gain, preserve_local_hue=preserve_local_hue), tile)
        for tile_id, x, y, _, tile in placements
    ]
    return processed, report


def validate_structure_recolor_selection(plan: dict, selected_tiles: set[str] | None) -> None:
    if not selected_tiles:
        raise ValueError(f"统一灰阶与墨线实验必须先用‘选择图块’指定相邻 4–{MAX_STRUCTURE_RECOLOR_TILES} 块")
    if len(selected_tiles) < 4:
        raise ValueError("统一灰阶与墨线实验至少选择 4 块相邻图块")
    if len(selected_tiles) > MAX_STRUCTURE_RECOLOR_TILES:
        raise ValueError(f"统一灰阶与墨线实验最多处理 {MAX_STRUCTURE_RECOLOR_TILES} 块")
    by_id = {tile["tile_id"].upper(): tile for tile in plan.get("tiles", [])}
    missing = sorted(selected_tiles - set(by_id))
    if missing:
        raise ValueError(f"选择图块不存在：{', '.join(missing)}")
    chosen = [by_id[tile_id] for tile_id in selected_tiles]
    rows = sorted({int(tile["row"]) for tile in chosen})
    cols = sorted({int(tile["col"]) for tile in chosen})
    expected = {(row, col) for row in range(rows[0], rows[-1] + 1) for col in range(cols[0], cols[-1] + 1)}
    actual = {(int(tile["row"]), int(tile["col"])) for tile in chosen}
    if actual != expected:
        raise ValueError("统一灰阶与墨线实验要求选择连续的矩形相邻图块")


def stitch_tiles(
    manifest_path: Path,
    restored_dir: Path,
    out_name: str,
    output_dir: Path | None = None,
    mode: str = "full",
    selected_tiles: set[str] | None = None,
    blend_mode: str = "feather",
    color_mode: str = "none",
    feather_px: int | None = None,
    reference_path: Path | None = None,
    color_tiles: set[str] | None = None,
    trim_padding: bool = False,
    seam_balance: bool = False,
    seam_strength: str = "standard",
    low_freq_strength: float = 0.0,
    low_freq_mode: str = "auto",
    yellow_reduction: float = 0.0,
    palette_profile_path: Path | None = None,
    palette_reference_paths: list[Path] | None = None,
    palette_strength: float = 0.85,
    saturation_gain: float = 1.0,
    neutral_protection: float = 0.9,
    skin_protection: float = 0.55,
    gold_protection: float = 0.7,
    seam_diagnostics: bool = True,
    structure_recolor: bool = False,
    ink_lightness: float = 0.55,
    ink_expand_px: int = 1,
    preserve_local_hue: bool = True,
) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    out_dir = output_dir if output_dir else manifest_path.parent / "stitched"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = mode if mode in {"local", "full", "complete"} else "full"
    blend_mode = blend_mode if blend_mode in {"overwrite", "feather"} else "feather"
    legacy_color_modes = {"source_reference", "custom_reference", "global", "reference"}
    if color_mode in legacy_color_modes:
        color_mode = "rgb_stats"
    color_mode = color_mode if color_mode in {"none", "rgb_stats", "lab_reinhard", "seam_balance", "palette_family"} else "none"
    if color_mode == "palette_family":
        # palette_family = 两阶段流水线：
        #   第一阶段 全部图块 去色 → 联合灰阶/墨线归一化 → 18 色网格矫正（色卡档位矫正，明度统一消明度缝）
        #   第二阶段 palette_residual 接缝平衡（只修残余）+ 可选低频颜色场校正
        structure_recolor = True
        if color_tiles is not None:
            raise ValueError("两阶段流水线作用于全部图块（联合灰阶/墨线归一化需要整片），请清空「调色图块」")
        if selected_tiles:
            validate_structure_recolor_selection(plan, selected_tiles)
    elif structure_recolor:
        raise ValueError("统一灰阶与墨线只支持标准色谱迁移模式")
    if color_mode == "seam_balance":
        seam_balance = True
    if color_mode == "palette_family":
        seam_balance = True
    feather_px = int(feather_px if feather_px is not None else plan.get("overlap_output_px", 20))
    if seam_balance:
        initial_preset_name = "palette_residual" if color_mode == "palette_family" else seam_strength
        preset = SEAM_BALANCE_PRESETS.get(initial_preset_name, SEAM_BALANCE_PRESETS["standard"])
        feather_px = max(feather_px, preset["feather"])
    missing = []
    placements = []
    pasted = 0
    color_matched = 0
    color_tile_ids = []
    color_changes = []
    color_samples = []
    scale = float(plan["scale"])
    source_img = None
    reference_source = "none"
    reference_whole_stats = None
    palette_profile = None
    palette_json_path = None
    palette_preview_path = None
    palette_reference_preview = None
    palette_reference_mean = None
    substrate_cast_before = None
    substrate_cast_after = None
    palette_reference_paths = palette_reference_paths or ([] if reference_path is None else [reference_path])
    if color_mode == "palette_family":
        if palette_profile_path:
            palette_profile = load_palette_profile(palette_profile_path)
            palette_json_path = palette_profile_path
            reference_source = str(palette_profile_path)
        elif palette_reference_paths:
            reference_images = []
            try:
                for item in palette_reference_paths:
                    reference_images.append(Image.open(item).convert("RGB"))
                palette_profile = build_palette_profile(reference_images)
                palette_json_path = save_palette_profile(palette_profile, out_dir / "standard_palette.json")
                reference_source = "; ".join(str(item) for item in palette_reference_paths)
            finally:
                for image in reference_images:
                    image.close()
        else:
            # 兜底：未给色卡时用 manifest 记录的源图作为参考（走自动白平衡提取）
            fallback = resolve_image_path(str(plan.get("path", "")))
            if not fallback.exists():
                raise RuntimeError("标准色谱模式需要选择色谱 JSON，或至少一张标准色彩图")
            reference_images = [Image.open(fallback).convert("RGB")]
            try:
                palette_profile = build_palette_profile(reference_images)
                palette_json_path = save_palette_profile(palette_profile, out_dir / "standard_palette.json")
                reference_source = str(fallback)
            finally:
                for image in reference_images:
                    image.close()
        palette_preview_path = render_palette_profile(palette_profile, out_dir / "standard_palette.png")
        if palette_reference_paths:
            with Image.open(palette_reference_paths[0]) as preview_source:
                palette_reference_preview = preview_source.convert("RGB").copy()
                palette_reference_preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
            palette_reference_mean = rounded_rgb_mean(palette_reference_preview)
        elif palette_preview_path.exists():
            # 用色卡 JSON 时：comparison 的 reference 直接显示色板预览
            with Image.open(palette_preview_path) as preview_source:
                palette_reference_preview = preview_source.convert("RGB").copy()
                palette_reference_preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
            palette_reference_mean = rounded_rgb_mean(palette_reference_preview)
    if color_mode not in {"none", "palette_family", "seam_balance"}:
        try:
            source_path = reference_path if reference_path else resolve_image_path(plan.get("path", ""))
            if source_path.exists():
                source_img = Image.open(source_path).convert("RGB")
                if reference_path:
                    reference_whole_stats = source_img.copy()
                    reference_source = str(source_path)
                else:
                    reference_source = "source_image_regions"
        except Exception:
            source_img = None
    for tile in plan["tiles"]:
        if selected_tiles and tile["tile_id"].upper() not in selected_tiles:
            continue
        tile_file = find_tile_file(restored_dir, tile.get("output_file", ""), tile["tile_id"])
        if tile_file is None:
            missing.append(tile["tile_id"])
            continue
        with Image.open(tile_file) as restored:
            restored = restored.convert("RGB")
            if plan.get("layout_mode") == "fixed_3x4":
                tile_rect = tile["tile_rect"]
                expected_size = (tile_rect[2], tile_rect[3])
                if restored.size != expected_size:
                    raise RuntimeError(
                        f"{tile['tile_id']} 尺寸为 {restored.size}，应为 {expected_size}"
                    )
                if trim_padding:
                    sp = tile.get("source_paste_rect", [0, 0, tile_rect[2], tile_rect[3]])
                    if sp[2] > 0 and sp[3] > 0 and (sp[2] < tile_rect[2] or sp[3] < tile_rect[3]):
                        core_img = restored.crop((sp[0], sp[1], sp[0] + sp[2], sp[1] + sp[3])).copy()
                    else:
                        core_img = restored.copy()
                    paste_x = tile_rect[0]
                    paste_y = tile_rect[1]
                else:
                    core_img = restored.copy()
                    paste_x = tile_rect[0]
                    paste_y = tile_rect[1]
            else:
                core = tile["core"]
                context = tile["context"]
                left = round((core[0] - context[0]) * scale)
                top = round((core[1] - context[1]) * scale)
                right = left + round(core[2] * scale)
                bottom = top + round(core[3] * scale)
                core_img = restored.crop((left, top, right, bottom)).copy()
                paste_x = round(core[0] * scale)
                paste_y = round(core[1] * scale)
            if color_mode == "palette_family" and not structure_recolor and (color_tiles is None or tile["tile_id"].upper() in color_tiles):
                before_img = core_img.copy()
                before_mean = rounded_rgb_mean(before_img)
                core_img = apply_palette_profile(
                    core_img,
                    palette_profile,
                    palette_strength,
                    neutral_protection,
                    skin_protection,
                    gold_protection,
                )
                after_mean = rounded_rgb_mean(core_img)
                color_matched += 1
                color_tile_ids.append(tile["tile_id"])
                color_changes.append({
                    "tile_id": tile["tile_id"], "before_rgb_mean": before_mean,
                    "after_rgb_mean": after_mean,
                    "delta_rgb_mean": [after_mean[i] - before_mean[i] for i in range(3)],
                })
                if len(color_samples) < 12:
                    color_samples.append({
                        "tile_id": tile["tile_id"], "before": comparison_thumbnail(before_img),
                        "after": comparison_thumbnail(core_img),
                        "reference": comparison_thumbnail(palette_reference_preview) if palette_reference_preview else comparison_thumbnail(core_img),
                        "before_mean": before_mean, "after_mean": after_mean,
                        "reference_mean": palette_reference_mean or after_mean,
                    })
            elif color_mode not in {"none", "seam_balance"} and (color_tiles is None or tile["tile_id"].upper() in color_tiles):
                reference = reference_whole_stats or source_reference_for_tile(plan, tile, source_img)
                if reference is not None:
                    before_img = core_img.copy()
                    before_mean = rounded_rgb_mean(before_img)
                    reference_mean = rounded_rgb_mean(reference)
                    core_img = match_tile_color(core_img, reference, color_mode)
                    after_mean = rounded_rgb_mean(core_img)
                    color_matched += 1
                    color_tile_ids.append(tile["tile_id"])
                    color_changes.append(
                        {
                            "tile_id": tile["tile_id"],
                            "before_rgb_mean": before_mean,
                            "after_rgb_mean": after_mean,
                            "reference_rgb_mean": reference_mean,
                            "delta_rgb_mean": [after_mean[i] - before_mean[i] for i in range(3)],
                        }
                    )
                    if len(color_samples) < 12:
                        color_samples.append(
                            {
                                "tile_id": tile["tile_id"],
                                "before": comparison_thumbnail(before_img),
                                "after": comparison_thumbnail(core_img),
                                "reference": comparison_thumbnail(reference),
                                "before_mean": before_mean,
                                "after_mean": after_mean,
                                "reference_mean": reference_mean,
                            }
                        )
            placements.append((tile["tile_id"], paste_x, paste_y, core_img, tile))
    if source_img is not None:
        source_img.close()
    if color_mode == "palette_family" and yellow_reduction > 0 and placements:
        substrate_cast_before = estimate_low_chroma_cast(item[3] for item in placements)
    structure_recolor_report = None
    if structure_recolor and placements:
        structure_before = {
            tid: (comparison_thumbnail(img), rounded_rgb_mean(img))
            for tid, _, _, img, _ in placements
        }
        placements, structure_recolor_report = apply_structure_recolor_placements(
            placements, palette_profile, palette_strength, ink_lightness, saturation_gain, ink_expand_px,
            preserve_local_hue,
        )
        color_matched = len(placements)
        color_tile_ids = [item[0] for item in placements]
        for tid, _, _, img, _ in placements:
            before_thumb, before_mean = structure_before[tid]
            after_mean = rounded_rgb_mean(img)
            color_changes.append({
                "tile_id": tid,
                "before_rgb_mean": before_mean,
                "after_rgb_mean": after_mean,
                "delta_rgb_mean": [after_mean[i] - before_mean[i] for i in range(3)],
            })
            color_samples.append({
                "tile_id": tid,
                "before": before_thumb,
                "after": comparison_thumbnail(img),
                "reference": comparison_thumbnail(palette_reference_preview) if palette_reference_preview else comparison_thumbnail(img),
                "before_mean": before_mean,
                "after_mean": after_mean,
                "reference_mean": palette_reference_mean or after_mean,
            })
    seam_bias_report = []
    diagnostic_before = (
        analyze_placement_seams(placements, int(plan.get("rows", 1)), int(plan.get("cols", 1)))
        if seam_balance and placements else []
    )
    if seam_balance and placements:
        preset_name = "palette_residual" if color_mode == "palette_family" else seam_strength
        preset = SEAM_BALANCE_PRESETS.get(preset_name, SEAM_BALANCE_PRESETS["standard"])
        allowed_pairs = {
            (item["first"], item["second"]) for item in diagnostic_before if item.get("safe_for_color_balance")
        }
        tile_images_for_solve = [(p[0], p[3], p[4]) for p in placements]
        biases = solve_seam_biases(
            tile_images_for_solve,
            int(plan.get("rows", 1)),
            int(plan.get("cols", 1)),
            preset["strip"],
            preset["inset"],
            preset["regularize"],
            allowed_pairs,
            ({p[0] for p in placements if color_tiles is not None and p[0].upper() not in color_tiles}
             if color_mode == "palette_family" else None),
        )
        new_placements = []
        for tid, px, py, img, t in placements:
            bias = biases.get(tid, np.zeros(3, dtype=np.float32))
            apply_bias = color_mode != "palette_family" or color_tiles is None or tid.upper() in color_tiles
            if apply_bias:
                img = apply_tile_bias(img, bias, preset["max_bias"])
            else:
                bias = np.zeros(3, dtype=np.float32)
            seam_bias_report.append({
                "tile_id": tid,
                "bias": [round(float(x), 2) for x in np.clip(bias, -preset["max_bias"], preset["max_bias"])],
            })
            new_placements.append((tid, px, py, img, t))
        placements = new_placements
        if color_mode != "palette_family":
            color_matched = len(placements)
            color_tile_ids = [p[0] for p in placements]
            color_mode = "seam_balance"
    if low_freq_strength > 0 and low_freq_mode != "off" and placements:
        placements = low_frequency_correct_placements(placements, plan, low_freq_strength, low_freq_mode)
    if substrate_cast_before is not None and placements:
        substrate_cast_after = estimate_low_chroma_cast(item[3] for item in placements)
        placements = [
            (tid, px, py, reduce_warm_cast(img, substrate_cast_before, substrate_cast_after, yellow_reduction), tile)
            for tid, px, py, img, tile in placements
        ]
    seam_report = None
    seam_results = []
    if seam_diagnostics and placements:
        seam_results = analyze_placement_seams(placements, int(plan.get("rows", 1)), int(plan.get("cols", 1)))
        seam_report = write_seam_report(seam_results, out_dir)
    if mode == "complete" and missing:
        raise RuntimeError(f"完整拼合缺少 {len(missing)} 块：{', '.join(missing[:12])}")
    if not placements:
        raise RuntimeError("没有找到可拼合的图块，请检查文件夹或编号")
    if mode == "local":
        min_x = min(item[1] for item in placements)
        min_y = min(item[2] for item in placements)
        max_x = max(item[1] + item[3].width for item in placements)
        max_y = max(item[2] + item[3].height for item in placements)
        canvas = Image.new("RGB", (max_x - min_x, max_y - min_y), (245, 242, 235))
        offset_x = min_x
        offset_y = min_y
    else:
        if trim_padding:
            canvas = Image.new("RGB", (int(plan["width"]), int(plan["height"])), (245, 242, 235))
        else:
            canvas = Image.new("RGB", (plan["working_width"], plan["working_height"]), (245, 242, 235))
        offset_x = 0
        offset_y = 0
    if blend_mode == "feather":
        canvas_w, canvas_h = canvas.size
        weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        color_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        for _, paste_x, paste_y, core_img, tile in placements:
            mask = feather_mask(core_img.size, tile, plan, feather_px)
            mask_arr = np.asarray(mask, dtype=np.float32)
            color_arr = np.asarray(core_img, dtype=np.float32)
            tx = paste_x - offset_x
            ty = paste_y - offset_y
            h, w = color_arr.shape[:2]
            weight_sum[ty:ty + h, tx:tx + w] += mask_arr
            color_sum[ty:ty + h, tx:tx + w] += color_arr * mask_arr[:, :, np.newaxis]
            pasted += 1
        weight_safe = np.maximum(weight_sum, 1.0)
        result_arr = color_sum / weight_safe[:, :, np.newaxis]
        uncovered = weight_sum <= 0
        result_arr[uncovered] = (245, 242, 235)
        canvas = Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8), "RGB")
    else:
        for _, paste_x, paste_y, core_img, tile in placements:
            canvas.paste(core_img, (paste_x - offset_x, paste_y - offset_y))
            pasted += 1
    out_path = out_dir / safe_name(out_name or "stitched_result.png")
    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        out_path = out_path.with_suffix(".png")
    canvas.save(out_path)
    color_compare_path, color_compare_items = make_color_comparison(color_samples, out_dir)
    return {
        "out_path": str(out_path),
        "preview_url": f"/api/preview?path={quote(str(out_path))}",
        "mode": mode,
        "pasted": pasted,
        "missing": missing,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "canvas_width": canvas.width,
        "canvas_height": canvas.height,
        "blend_mode": blend_mode,
        "color_mode": color_mode,
        "color_reference": reference_source,
        "color_applied": color_matched > 0,
        "color_matched_tiles": color_matched,
        "color_tile_ids": color_tile_ids,
        "color_changes": color_changes[:24],
        "color_compare_path": str(color_compare_path) if color_compare_path else "",
        "color_compare_preview_url": f"/api/preview?path={quote(str(color_compare_path))}" if color_compare_path else "",
        "color_compare_items": color_compare_items,
        "feather_px": feather_px,
        "trim_padding": trim_padding,
        "seam_balance": seam_balance,
        "seam_strength": seam_strength,
        "seam_bias_report": seam_bias_report[:48],
        "low_freq_strength": low_freq_strength,
        "low_freq_mode": low_freq_mode,
        "yellow_reduction": yellow_reduction,
        "substrate_cast_before": substrate_cast_before.tolist() if substrate_cast_before is not None else None,
        "substrate_cast_after": substrate_cast_after.tolist() if substrate_cast_after is not None else None,
        "palette_profile_path": str(palette_json_path) if palette_json_path else "",
        "palette_preview_path": str(palette_preview_path) if palette_preview_path else "",
        "palette_preview_url": f"/api/preview?path={quote(str(palette_preview_path))}" if palette_preview_path else "",
        "palette_strength": palette_strength,
        "structure_recolor": structure_recolor,
        "ink_lightness": ink_lightness,
        "structure_recolor_report": structure_recolor_report,
        "seam_diagnostics": seam_results,
        "seam_report": seam_report,
        "tile_ids": [item[0] for item in placements],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_OPTIONS(self) -> None:  # noqa: N802
        core.handle_options(self)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                core.api_ok(self, core.health_payload(
                    "xiangliu-grid", APP_VERSION, self.server.server_address[1],
                    name=APP_NAME, nameEn=APP_NAME_EN,
                ))
                return
            if parsed.path == "/":
                self.serve_file(core.safe_join(WEB_ROOT, "index.html"))
                return

            if parsed.path == "/api/preview":
                qs = parse_qs(parsed.query)
                path = resolve_image_path(qs.get("path", [""])[0])
                preview = ensure_preview(path)
                self.serve_file(preview)
                return

            if parsed.path == "/api/history":
                core.api_ok(self, {
                    "items": list_manifest_history(),
                    "stitch_items": load_stitch_history(),
                })
                return

            # 下面两个前缀必须走真路径包含校验。
            # 旧版这里是 (ROOT / path).resolve() + "ROOT in parents"，
            # 结果 /outputs/../server.py 能把源码读出去。
            if parsed.path.startswith("/outputs/"):
                self.serve_file(core.safe_join(OUTPUT_DIR, parsed.path[len("/outputs/"):]))
                return
            if parsed.path.startswith("/assets/"):
                self.serve_file(core.safe_join(ASSET_ROOT, parsed.path[len("/assets/"):]))
                return

            rel = parsed.path.lstrip("/") or "index.html"
            if core.serve_static(self, WEB_ROOT, rel):
                return
            raise core.NotFoundError(f"找不到页面：{parsed.path}", detail=parsed.path)
        except Exception as exc:
            core.api_exception(self, exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/pick-file":
                payload = read_json(self)
                kind = payload.get("kind", "image")
                if kind == "manifest":
                    picked = choose_file("选择 tiles_manifest.json", [("JSON", "*.json"), ("All files", "*.*")])
                elif kind == "palette":
                    picked = choose_file("选择标准色谱 JSON", [("JSON", "*.json"), ("All files", "*.*")])
                else:
                    picked = choose_file(
                        "选择壁画原图",
                        [
                            ("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.webp *.bmp"),
                            ("All files", "*.*"),
                        ],
                    )
                json_response(self, {"path": picked})
                return
            if parsed.path == "/api/pick-folder":
                payload = read_json(self)
                title = payload.get("title", "选择文件夹")
                picked = choose_folder(title)
                json_response(self, {"path": picked})
                return
            if parsed.path == "/api/upload":
                filename = core.safe_filename(
                    self.headers.get("X-Filename", ""),
                    fallback=f"upload_{int(time.time())}.png",
                )
                try:
                    length = int(self.headers.get("Content-Length", "0") or 0)
                except ValueError:
                    raise core.ValidationError("Content-Length 不合法。")
                if length <= 0:
                    raise core.ValidationError("上传内容为空。")
                if length > MAX_UPLOAD_BYTES:
                    raise core.PayloadTooLargeError(
                        f"图片过大，单次上传上限 {MAX_UPLOAD_BYTES // (1024 ** 3)} GB。",
                        detail=f"content-length={length}")
                # 重名不覆盖：自动加 _1 / _2 后缀
                out = core.unique_path(INPUT_DIR, filename)
                with out.open("wb") as f:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)
                json_response(self, {"ok": True, "path": str(out), **image_info(out)})
                return
            payload = read_json(self)
            if parsed.path == "/api/inspect":
                path = resolve_image_path(payload.get("path", ""))
                info = image_info(path)
                json_response(self, {**info, "preview_url": f"/api/preview?path={quote(str(path))}"})
                return
            if parsed.path == "/api/palette-profile":
                paths = parse_paths(payload.get("reference_paths"))
                if not paths:
                    raise RuntimeError("请至少选择一张标准色彩图")
                output_raw = payload.get("output_dir", "")
                output = resolve_image_path(output_raw) if output_raw else OUTPUT_DIR / "palettes"
                output.mkdir(parents=True, exist_ok=True)
                white_balance = bool(payload.get("white_balance", True))
                images = []
                try:
                    images = [Image.open(path).convert("RGB") for path in paths]
                    profile = build_palette_profile(images, white_balance=white_balance)
                finally:
                    for image in images:
                        image.close()
                json_path = save_palette_profile(profile, output / "standard_palette.json")
                preview_path = render_palette_profile(profile, output / "standard_palette.png")
                json_response(self, {
                    "profile_path": str(json_path),
                    "preview_path": str(preview_path),
                    "preview_url": f"/api/preview?path={quote(str(preview_path))}",
                    "reference_count": len(paths),
                    "swatches": profile.get("swatches", []),
                })
                return
            if parsed.path == "/api/plan":
                path = resolve_image_path(payload.get("path", ""))
                long_edge = clamp_int(payload, "long_edge", 2048, 256, MAX_LONG_EDGE, "块边长")
                overlap = clamp_int(payload, "overlap", 20, 0, MAX_OVERLAP, "重叠像素")
                check_overlap(long_edge, overlap)
                plan = build_plan(
                    path,
                    clamp_int(payload, "target_pieces", 50, 1, MAX_TARGET_PIECES, "目标块数"),
                    long_edge,
                    overlap,
                    bool(payload.get("allow_upscale", False)),
                    bool(payload.get("strict_count", False)),
                    clamp_int(payload, "grid_shift_x", 0, -100000, 100000, "水平分割线平移"),
                    clamp_int(payload, "grid_shift_y", 0, -100000, 100000, "垂直分割线平移"),
                )
                check_tile_budget(plan)
                plan.pop("tiles")
                json_response(self, plan)
                return
            if parsed.path == "/api/grid-plan":
                path = resolve_image_path(payload.get("path", ""))
                plan = build_grid_plan(
                    path,
                    clamp_int(payload, "rows", 1, 1, MAX_GRID_SIDE, "行数"),
                    clamp_int(payload, "cols", 3, 1, MAX_GRID_SIDE, "列数"),
                )
                check_tile_budget(plan)
                plan.pop("tiles")
                json_response(self, plan)
                return
            if parsed.path == "/api/split":
                path = resolve_image_path(payload.get("path", ""))
                output_base_raw = payload.get("output_base", "")
                output_base = resolve_image_path(output_base_raw) if output_base_raw else None
                long_edge = clamp_int(payload, "long_edge", 2048, 256, MAX_LONG_EDGE, "块边长")
                overlap = clamp_int(payload, "overlap", 20, 0, MAX_OVERLAP, "重叠像素")
                check_overlap(long_edge, overlap)
                result = split_image(
                    path,
                    clamp_int(payload, "target_pieces", 50, 1, MAX_TARGET_PIECES, "目标块数"),
                    long_edge,
                    overlap,
                    payload.get("job_name", ""),
                    output_base,
                    bool(payload.get("allow_upscale", False)),
                    bool(payload.get("strict_count", False)),
                    clamp_int(payload, "grid_shift_x", 0, -100000, 100000, "水平分割线平移"),
                    clamp_int(payload, "grid_shift_y", 0, -100000, 100000, "垂直分割线平移"),
                )
                result["plan"].pop("tiles", None)
                remember_manifest(Path(result["manifest_json"]), Path(result["tiles_dir"]), payload.get("job_name", ""))
                json_response(self, result)
                return
            if parsed.path == "/api/split-grid":
                path = resolve_image_path(payload.get("path", ""))
                output_base_raw = payload.get("output_base", "")
                output_base = resolve_image_path(output_base_raw) if output_base_raw else None
                result = split_image_grid(
                    path,
                    clamp_int(payload, "rows", 1, 1, MAX_GRID_SIDE, "行数"),
                    clamp_int(payload, "cols", 3, 1, MAX_GRID_SIDE, "列数"),
                    payload.get("job_name", ""),
                    output_base,
                )
                result["plan"].pop("tiles", None)
                remember_manifest(Path(result["manifest_json"]), Path(result["tiles_dir"]), payload.get("job_name", ""))
                json_response(self, result)
                return
            if parsed.path == "/api/resize-export":
                path = resolve_image_path(payload.get("path", ""))
                output_base_raw = payload.get("output_base", "")
                output_base = resolve_image_path(output_base_raw) if output_base_raw else None
                result = resize_export(
                    path,
                    payload.get("edge_mode", "long"),
                    clamp_int(payload, "target_px", 2048, 64, 65536, "目标边长"),
                    payload.get("scale_mode", "edge"),
                    clamp_float(payload, "scale_factor", 1.0, 0.05, MAX_RESIZE_SCALE, "缩放倍数"),
                    payload.get("format", "png"),
                    clamp_int(payload, "quality", 92, 1, 100, "图片质量"),
                    bool(payload.get("allow_upscale", False)),
                    payload.get("job_name", ""),
                    output_base,
                    payload.get("out_name", ""),
                )
                json_response(self, result)
                return
            if parsed.path in {"/api/patch-align-preview", "/api/patch-align-apply"}:
                manifest = resolve_image_path(payload.get("manifest_path", ""))
                tiles_dir = resolve_image_path(payload.get("tiles_dir", ""))
                patch_path = resolve_image_path(payload.get("patch_path", ""))
                output_base_raw = payload.get("output_base", "")
                output_base = resolve_image_path(output_base_raw) if output_base_raw else None
                candidate_ids = parse_tile_ids(payload.get("candidate_tiles", ""))
                result = patch_align(
                    manifest,
                    tiles_dir,
                    patch_path,
                    candidate_ids,
                    output_base,
                    payload.get("run_name", ""),
                    parsed.path == "/api/patch-align-apply",
                    float(payload.get("min_score", 0.28)),
                )
                json_response(self, result)
                return
            if parsed.path == "/api/patch-selector-map":
                manifest = resolve_image_path(payload.get("manifest_path", ""))
                tiles_dir = resolve_image_path(payload.get("tiles_dir", ""))
                json_response(self, make_tile_selector_map(manifest, tiles_dir))
                return
            if parsed.path == "/api/stitch":
                manifest = resolve_image_path(payload.get("manifest_path", ""))
                restored_dir = resolve_image_path(payload.get("restored_dir", ""))
                output_dir_raw = payload.get("output_dir", "")
                output_dir = resolve_image_path(output_dir_raw) if output_dir_raw else None
                reference_path_raw = payload.get("reference_path", "")
                reference_path = resolve_image_path(reference_path_raw) if reference_path_raw else None
                palette_path_raw = payload.get("palette_profile_path", "")
                palette_path = resolve_image_path(palette_path_raw) if palette_path_raw else None
                result = stitch_tiles(
                    manifest,
                    restored_dir,
                    payload.get("out_name", "stitched_result.png"),
                    output_dir,
                    payload.get("mode", "full"),
                    parse_selected_tiles(payload.get("selected_tiles")),
                    payload.get("blend_mode", "feather"),
                    payload.get("color_mode", "none"),
                    int(payload.get("feather_px", 20)),
                    reference_path,
                    parse_selected_tiles(payload.get("color_tiles")),
                    bool(payload.get("trim_padding", False)),
                    bool(payload.get("seam_balance", False)),
                    payload.get("seam_strength", "standard"),
                    float(payload.get("low_freq_strength", 0.0)),
                    payload.get("low_freq_mode", "auto"),
                    float(payload.get("yellow_reduction", 0.0)),
                    palette_path,
                    parse_paths(payload.get("palette_reference_paths")),
                    float(payload.get("palette_strength", 0.85)),
                    float(payload.get("saturation_gain", 1.0)),
                    float(payload.get("neutral_protection", 0.9)),
                    float(payload.get("skin_protection", 0.55)),
                    float(payload.get("gold_protection", 0.7)),
                    bool(payload.get("seam_diagnostics", True)),
                    bool(payload.get("structure_recolor", False)),
                    float(payload.get("ink_lightness", 0.55)),
                    int(payload.get("ink_expand_px", 1)),
                    bool(payload.get("preserve_local_hue", True)),
                )
                remember_stitch_history(payload, result)
                json_response(self, result)
                return
            json_response(self, {"error": {"code": "NOT_FOUND", "message": "找不到接口。"}}, 404)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            core.api_exception(self, exc)

    def serve_file(self, path: Path) -> None:
        """分块发送文件（SERIES-SPEC §7 / S6），不再把大图整份读进内存。"""
        core.stream_file(self, path)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    core.print_banner(APP_NAME, APP_NAME_EN, APP_VERSION, port)
    print(f"  输出目录：{OUTPUT_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
