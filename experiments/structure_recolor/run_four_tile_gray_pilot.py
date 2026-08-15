from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from color_engine.color_space import oklab_to_rgb
from color_engine.palette import build_palette_profile
from color_engine.report import write_seam_report
from color_engine.seams import analyze_placement_seams
from experiments.structure_recolor.engine import decompose_tile, joint_grayscale_normalize, recolor_tile


DEFAULT_SOURCE = Path(r"E:\Onedrive\真老师的奇妙幻境\合作系列\山西大云禅院壁画修复\第二阶段-0812开始\9张还原性创作\9张还原性创作")
DEFAULT_OUTPUT = ROOT / "validation" / "four_tile_gray_pilot"
STRIDE = 2028
TILES = [
    ("R01_C07", "R01_C07_x12168_y0_v001.png", 0, 0, 1, 1),
    ("R01_C08", "R01_C08_x14196_y0_v001.png", STRIDE, 0, 1, 2),
    ("R02_C07", "R02_C07_x12168_y2028_v001.png", 0, STRIDE, 2, 1),
    ("R02_C08", "R02_C08_x14196_y2028_v001.png", STRIDE, STRIDE, 2, 2),
]
STANDARD = "R01_C09_x16224_y0_v001.png"


def mosaic(images: dict[str, Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (4076, 4076), (238, 232, 220))
    for tile_id, _, x, y, _, _ in TILES:
        canvas.paste(images[tile_id], (x, y))
    return canvas


def comparison(before: Image.Image, after: Image.Image, labels: tuple[str, str]) -> Image.Image:
    items = []
    for image in (before, after):
        copy = image.copy()
        copy.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
        items.append(copy)
    canvas = Image.new("RGB", (items[0].width + items[1].width, max(i.height for i in items) + 50), (25, 25, 25))
    canvas.paste(items[0], (0, 50)); canvas.paste(items[1], (items[0].width, 50))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 17), labels[0], fill=(245, 242, 235))
    draw.text((items[0].width + 12, 17), labels[1], fill=(245, 242, 235))
    return canvas


def placements(images):
    return [(tile_id, x, y, images[tile_id], {"row": row, "col": col})
            for tile_id, _, x, y, row, col in TILES]


def main(source: Path, output: Path, ink_lightness: float = 0.48) -> None:
    total_started = time.perf_counter()
    for name in ("gray_before", "gray_after", "recolored"):
        (output / name).mkdir(parents=True, exist_ok=True)
    read_started = time.perf_counter()
    originals, layers, gray_before = {}, {}, {}
    for tile_id, filename, *_ in TILES:
        with Image.open(source / filename) as image:
            originals[tile_id] = image.convert("RGB").copy()
        layers[tile_id] = decompose_tile(originals[tile_id])
        gray_before[tile_id] = layers[tile_id].structure_preview
        gray_before[tile_id].save(output / "gray_before" / f"{tile_id}.png")
    with Image.open(source / STANDARD) as image:
        profile = build_palette_profile([image.convert("RGB")])
    read_seconds = time.perf_counter() - read_started
    gray_started = time.perf_counter()
    light_placements = [(tile_id, x, y, layers[tile_id].structure_l) for tile_id, _, x, y, _, _ in TILES]
    normalized, gray_report = joint_grayscale_normalize(
        light_placements, (4076, 4076), overlap=20, ink_lightness=ink_lightness
    )
    gray_seconds = time.perf_counter() - gray_started
    normalized_by_id = {item[0]: item[3] for item in normalized}
    gray_after, recolored = {}, {}
    recolor_started = time.perf_counter()
    for tile_id, *_ in TILES:
        light = normalized_by_id[tile_id]
        neutral = oklab_to_rgb(np.stack((light, np.zeros_like(light), np.zeros_like(light)), axis=-1))
        gray_after[tile_id] = Image.fromarray(neutral, "RGB")
        gray_after[tile_id].save(output / "gray_after" / f"{tile_id}.png")
        recolored[tile_id] = recolor_tile(layers[tile_id], light, profile, strength=0.88)
        recolored[tile_id].save(output / "recolored" / f"{tile_id}.png")
    recolor_seconds = time.perf_counter() - recolor_started

    report_started = time.perf_counter()
    original_mosaic, gray_before_mosaic = mosaic(originals), mosaic(gray_before)
    gray_after_mosaic, recolored_mosaic = mosaic(gray_after), mosaic(recolored)
    comparison(gray_before_mosaic, gray_after_mosaic, ("GRAY BEFORE", "GRAY AFTER")).save(output / "gray_comparison.jpg", quality=94)
    comparison(original_mosaic, recolored_mosaic, ("COLOR BEFORE", "COLOR AFTER")).save(output / "color_comparison.jpg", quality=94)
    before_seams = analyze_placement_seams(placements(gray_before), 2, 2)
    after_seams = analyze_placement_seams(placements(gray_after), 2, 2)
    color_after_seams = analyze_placement_seams(placements(recolored), 2, 2)
    seam_report = write_seam_report(color_after_seams, output)
    timing = {
        "read_and_decompose_seconds": round(read_seconds, 3),
        "gray_and_ink_seconds": round(gray_seconds, 3),
        "recolor_and_tile_save_seconds": round(recolor_seconds, 3),
    }
    metrics = {"parameters": {"ink_lightness": ink_lightness}, "timing": timing,
               "gray_normalization": gray_report, "gray_before_seams": before_seams,
               "gray_after_seams": after_seams, "color_after_seams": color_after_seams,
               "seam_report": seam_report}
    timing["report_and_comparison_seconds"] = round(time.perf_counter() - report_started, 3)
    timing["total_seconds"] = round(time.perf_counter() - total_started, 3)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "ink_lightness": ink_lightness,
                      "timing": timing, "tiles": gray_report["tiles"],
                      "seams": seam_report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ink-lightness", type=float, default=0.48,
                        help="墨线目标明度，0.25–0.65；数值越大越淡")
    args = parser.parse_args()
    main(args.source, args.output, args.ink_lightness)
