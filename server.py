from __future__ import annotations

import csv
import hashlib
import json
import math
import mimetypes
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

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
DEFAULT_PORT = 8765
APP_NAME = "Xiangliu Grid"
APP_VERSION = "0.4.1"

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
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


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


def read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


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
) -> dict:
    info = image_info(path)
    tile_w, tile_h = fixed_tile_size(long_edge)
    stride_w = max(1, tile_w - overlap)
    stride_h = max(1, tile_h - overlap)
    cols = max(1, math.ceil(max(0, info["width"] - tile_w) / stride_w) + 1)
    rows = max(1, math.ceil(max(0, info["height"] - tile_h) / stride_h) + 1)
    cover_w, cover_h = fixed_coverage(cols, rows, tile_w, tile_h, overlap)
    scale = 1.0
    scaled_image_w = info["width"]
    scaled_image_h = info["height"]
    offset_x = 0
    offset_y = 0
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
) -> dict:
    plan = build_plan(path, target, long_edge, overlap, allow_upscale, strict_count)
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
) -> dict[str, np.ndarray]:
    by_pos = {(int(t[2].get("row", 1)), int(t[2].get("col", 1))): t for t in tile_images}
    ids = [t[0] for t in tile_images]
    index = {tid: i for i, tid in enumerate(ids)}
    equations: list[tuple[str, str]] = []
    targets: list[np.ndarray] = []
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            item = by_pos.get((row, col))
            if item is None:
                continue
            right = by_pos.get((row, col + 1))
            if right is not None:
                mean_a = strip_median_rgb(item[1], "right", strip_px, inset_px)
                mean_b = strip_median_rgb(right[1], "left", strip_px, inset_px)
                equations.append((item[0], right[0]))
                targets.append(mean_b - mean_a)
            bottom = by_pos.get((row + 1, col))
            if bottom is not None:
                mean_a = strip_median_rgb(item[1], "bottom", strip_px, inset_px)
                mean_b = strip_median_rgb(bottom[1], "top", strip_px, inset_px)
                equations.append((item[0], bottom[0]))
                targets.append(mean_b - mean_a)
    n = len(ids)
    mat = np.zeros((len(equations) + n + 1, n), dtype=np.float64)
    rhs = np.zeros((len(equations) + n + 1, 3), dtype=np.float64)
    for i, ((lid, rid), target) in enumerate(zip(equations, targets)):
        mat[i, index[lid]] = 1.0
        mat[i, index[rid]] = -1.0
        rhs[i] = target
    for off in range(n):
        mat[len(equations) + off, off] = regularize
    mat[-1, :] = 1.0
    solution, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
    return {tid: solution[index[tid]].astype(np.float32) for tid in ids}


def apply_tile_bias(image: Image.Image, bias: np.ndarray, max_abs: float) -> Image.Image:
    bias = np.clip(bias, -max_abs, max_abs)
    arr = np.asarray(image, dtype=np.float32) + bias.reshape(1, 1, 3)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def low_frequency_correct_placements(
    placements: list[tuple[str, int, int, Image.Image, dict]],
    plan: dict,
    strength: float = 0.5,
    small_width: int = 600,
) -> list[tuple[str, int, int, Image.Image, dict]]:
    """逐块低频校正：先用缩略图估算低频校正场，再放大校正场应用到每块全分辨率原图。
    校正场是平滑的低频场，相邻块的校正值在边界处几乎一致，不会引入新接缝。
    原图全分辨率细节完全保留，只修正大面积明暗/色温漂移。"""
    if strength <= 0 or not placements:
        return placements
    canvas_w = int(plan["width"])
    canvas_h = int(plan["height"])
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
    for ch in small_lab.split():
        arr = np.asarray(ch, dtype=np.float32)
        blurred = Image.fromarray(arr.astype(np.uint8), "L").filter(
            ImageFilter.GaussianBlur(radius=radius)
        )
        blurred_arr = np.asarray(blurred, dtype=np.float32)
        global_mean = float(blurred_arr.mean())
        correction_small = (blurred_arr - global_mean) * strength
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
    color_mode = color_mode if color_mode in {"none", "rgb_stats", "lab_reinhard", "seam_balance"} else "none"
    if color_mode == "seam_balance":
        seam_balance = True
    feather_px = int(feather_px if feather_px is not None else plan.get("overlap_output_px", 20))
    if seam_balance:
        preset = SEAM_BALANCE_PRESETS.get(seam_strength, SEAM_BALANCE_PRESETS["standard"])
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
    if color_mode != "none":
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
            if color_mode not in {"none", "seam_balance"} and (color_tiles is None or tile["tile_id"].upper() in color_tiles):
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
                                "before": before_img,
                                "after": core_img.copy(),
                                "reference": reference.copy(),
                                "before_mean": before_mean,
                                "after_mean": after_mean,
                                "reference_mean": reference_mean,
                            }
                        )
            placements.append((tile["tile_id"], paste_x, paste_y, core_img, tile))
    if source_img is not None:
        source_img.close()
    seam_bias_report = []
    if seam_balance and placements:
        preset = SEAM_BALANCE_PRESETS.get(seam_strength, SEAM_BALANCE_PRESETS["standard"])
        tile_images_for_solve = [(p[0], p[3], p[4]) for p in placements]
        biases = solve_seam_biases(
            tile_images_for_solve,
            int(plan.get("rows", 1)),
            int(plan.get("cols", 1)),
            preset["strip"],
            preset["inset"],
            preset["regularize"],
        )
        new_placements = []
        for tid, px, py, img, t in placements:
            bias = biases.get(tid, np.zeros(3, dtype=np.float32))
            img = apply_tile_bias(img, bias, preset["max_bias"])
            seam_bias_report.append({
                "tile_id": tid,
                "bias": [round(float(x), 2) for x in np.clip(bias, -preset["max_bias"], preset["max_bias"])],
            })
            new_placements.append((tid, px, py, img, t))
        placements = new_placements
        color_matched = len(placements)
        color_tile_ids = [p[0] for p in placements]
        color_mode = "seam_balance"
    if low_freq_strength > 0 and placements:
        placements = low_frequency_correct_placements(placements, plan, low_freq_strength)
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
        "tile_ids": [item[0] for item in placements],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file(WEB_ROOT / "index.html")
            return
        if parsed.path == "/api/preview":
            qs = parse_qs(parsed.query)
            path = resolve_image_path(qs.get("path", [""])[0])
            try:
                preview = ensure_preview(path)
                self.serve_file(preview)
            except Exception as exc:
                json_response(self, {"error": str(exc)}, 500)
            return
        if parsed.path == "/api/history":
            json_response(self, {"items": list_manifest_history()})
            return
        if parsed.path.startswith("/outputs/"):
            target = (ROOT / parsed.path.lstrip("/")).resolve()
            if ROOT in target.parents:
                self.serve_file(target)
            else:
                json_response(self, {"error": "Forbidden"}, 403)
            return
        if parsed.path.startswith("/assets/"):
            target = (ROOT / parsed.path.lstrip("/")).resolve()
            if ASSET_ROOT == target.parent or ASSET_ROOT in target.parents:
                self.serve_file(target)
            else:
                json_response(self, {"error": "Forbidden"}, 403)
            return
        self.serve_file(WEB_ROOT / parsed.path.lstrip("/"))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/pick-file":
                payload = read_json(self)
                kind = payload.get("kind", "image")
                if kind == "manifest":
                    picked = choose_file("选择 tiles_manifest.json", [("JSON", "*.json"), ("All files", "*.*")])
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
                filename = safe_name(self.headers.get("X-Filename", f"upload_{int(time.time())}.png"))
                length = int(self.headers.get("Content-Length", "0"))
                out = INPUT_DIR / filename
                with out.open("wb") as f:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)
                json_response(self, {"path": str(out), **image_info(out)})
                return
            payload = read_json(self)
            if parsed.path == "/api/inspect":
                path = resolve_image_path(payload.get("path", ""))
                info = image_info(path)
                json_response(self, {**info, "preview_url": f"/api/preview?path={quote(str(path))}"})
                return
            if parsed.path == "/api/plan":
                path = resolve_image_path(payload.get("path", ""))
                plan = build_plan(
                    path,
                    int(payload.get("target_pieces", 50)),
                    int(payload.get("long_edge", 2048)),
                    int(payload.get("overlap", 20)),
                    bool(payload.get("allow_upscale", False)),
                    bool(payload.get("strict_count", False)),
                )
                plan.pop("tiles")
                json_response(self, plan)
                return
            if parsed.path == "/api/grid-plan":
                path = resolve_image_path(payload.get("path", ""))
                plan = build_grid_plan(
                    path,
                    int(payload.get("rows", 1)),
                    int(payload.get("cols", 3)),
                )
                plan.pop("tiles")
                json_response(self, plan)
                return
            if parsed.path == "/api/split":
                path = resolve_image_path(payload.get("path", ""))
                output_base_raw = payload.get("output_base", "")
                output_base = resolve_image_path(output_base_raw) if output_base_raw else None
                result = split_image(
                    path,
                    int(payload.get("target_pieces", 50)),
                    int(payload.get("long_edge", 2048)),
                    int(payload.get("overlap", 20)),
                    payload.get("job_name", ""),
                    output_base,
                    bool(payload.get("allow_upscale", False)),
                    bool(payload.get("strict_count", False)),
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
                    int(payload.get("rows", 1)),
                    int(payload.get("cols", 3)),
                    payload.get("job_name", ""),
                    output_base,
                )
                result["plan"].pop("tiles", None)
                remember_manifest(Path(result["manifest_json"]), Path(result["tiles_dir"]), payload.get("job_name", ""))
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
                )
                json_response(self, result)
                return
            json_response(self, {"error": "Unknown endpoint"}, 404)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, 500)

    def serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            json_response(self, {"error": "Not found"}, 404)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"{APP_NAME} v{APP_VERSION} running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
