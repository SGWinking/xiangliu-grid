from __future__ import annotations

import json
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from color_engine.color_space import oklab_to_rgb
from color_engine.palette import build_palette_profile, render_palette_profile, save_palette_profile
from color_engine.report import write_seam_report
from color_engine.seams import analyze_placement_seams
from experiments.structure_recolor.engine import decompose_tile, joint_low_frequency_normalize, recolor_tile


DEFAULT_SOURCE = Path(r"E:\Onedrive\真老师的奇妙幻境\合作系列\山西大云禅院壁画修复\第二阶段-0812开始\9张还原性创作\9张还原性创作")
DEFAULT_OUTPUT = ROOT / "validation" / "structure_recolor_pilot"
STRIDE = 2028
TILES = [
    ("R01_C07", "R01_C07_x12168_y0_v001.png", 0, 0, 1, 1),
    ("R01_C08", "R01_C08_x14196_y0_v001.png", STRIDE, 0, 1, 2),
    ("R01_C09", "R01_C09_x16224_y0_v001.png", STRIDE * 2, 0, 1, 3),
    ("R02_C07", "R02_C07_x12168_y2028_v001.png", 0, STRIDE, 2, 1),
    ("R02_C08", "R02_C08_x14196_y2028_v001.png", STRIDE, STRIDE, 2, 2),
    ("R02_C09", "R02_C09_x16224_y2028_v001.png", STRIDE * 2, STRIDE, 2, 3),
]


def stitch(images: dict[str, Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (6104, 4076), (238, 232, 220))
    for tile_id, _, x, y, _, _ in TILES:
        canvas.paste(images[tile_id], (x, y))
    return canvas


def save_preview(image: Image.Image, path: Path, max_width: int = 2200) -> Path:
    preview = image.copy()
    preview.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
    preview.save(path, "JPEG", quality=92)
    return path


def make_comparison(original: Image.Image, structure: Image.Image, result: Image.Image) -> Image.Image:
    previews = []
    for image in (original, structure, result):
        copy = image.copy()
        copy.thumbnail((1100, 760), Image.Resampling.LANCZOS)
        previews.append(copy)
    width = sum(item.width for item in previews)
    height = max(item.height for item in previews) + 55
    canvas = Image.new("RGB", (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x = 0
    for label, image in zip(("ORIGINAL", "STRUCTURE + NORMALIZED L", "RECOLORED"), previews):
        draw.text((x + 10, 15), label, fill=(245, 242, 235), font=font)
        canvas.paste(image, (x, 48))
        x += image.width
    return canvas


def make_focus_comparison(original: Image.Image, result: Image.Image) -> Image.Image:
    # 对应用户截图中两个最明显的矩形接缝区域。
    regions = [(650, 1400, 1900, 3450), (3500, 1600, 4950, 3450)]
    panels = []
    for index, box in enumerate(regions, 1):
        before = original.crop(box)
        after = result.crop(box)
        pair = Image.new("RGB", (before.width * 2, before.height + 48), (28, 28, 28))
        pair.paste(before, (0, 48))
        pair.paste(after, (before.width, 48))
        draw = ImageDraw.Draw(pair)
        draw.text((12, 16), f"REGION {index} BEFORE", fill=(245, 242, 235))
        draw.text((before.width + 12, 16), f"REGION {index} AFTER", fill=(245, 242, 235))
        pair.thumbnail((2200, 900), Image.Resampling.LANCZOS)
        panels.append(pair)
    canvas = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), (28, 28, 28))
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    return canvas


def seam_metrics(placements):
    return analyze_placement_seams(placements, 2, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> None:
    for name in ("structure", "normalized_structure", "recolored"):
        (output / name).mkdir(parents=True, exist_ok=True)
    originals, layers = {}, {}
    for tile_id, filename, *_ in TILES:
        with Image.open(source / filename) as source_image:
            originals[tile_id] = source_image.convert("RGB").copy()
        layers[tile_id] = decompose_tile(originals[tile_id])
        layers[tile_id].structure_preview.save(output / "structure" / f"{tile_id}.png")

    profile = build_palette_profile([originals["R01_C09"]])
    save_palette_profile(profile, output / "standard_palette.json")
    render_palette_profile(profile, output / "standard_palette.png")

    light_placements = [
        (tile_id, x, y, layers[tile_id].structure_l) for tile_id, _, x, y, _, _ in TILES
    ]
    normalized, normalization_report = joint_low_frequency_normalize(light_placements, (6104, 4076), overlap=20)
    normalized_by_id = {item[0]: item[3] for item in normalized}
    normalized_images, recolored = {}, {}
    for tile_id, *_ in TILES:
        light = normalized_by_id[tile_id]
        neutral = oklab_to_rgb(np.stack((light, np.zeros_like(light), np.zeros_like(light)), axis=-1))
        normalized_images[tile_id] = Image.fromarray(neutral, "RGB")
        normalized_images[tile_id].save(output / "normalized_structure" / f"{tile_id}.png")
        recolored[tile_id] = recolor_tile(layers[tile_id], light, profile, strength=0.88)
        recolored[tile_id].save(output / "recolored" / f"{tile_id}.png")

    original_mosaic = stitch(originals)
    structure_mosaic = stitch(normalized_images)
    recolored_mosaic = stitch(recolored)
    save_preview(original_mosaic, output / "original_mosaic.jpg")
    save_preview(structure_mosaic, output / "normalized_structure_mosaic.jpg")
    save_preview(recolored_mosaic, output / "recolored_mosaic.jpg")
    make_comparison(original_mosaic, structure_mosaic, recolored_mosaic).save(output / "comparison.jpg", quality=92)
    make_focus_comparison(original_mosaic, recolored_mosaic).save(output / "focus_comparison.jpg", quality=94)

    def as_placements(images):
        return [
            (tile_id, x, y, images[tile_id], {"row": row, "col": col})
            for tile_id, _, x, y, row, col in TILES
        ]

    before_seams = seam_metrics(as_placements(originals))
    after_seams = seam_metrics(as_placements(recolored))
    seam_report = write_seam_report(after_seams, output)
    (output / "pilot_metrics.json").write_text(json.dumps({
        "standard_reference": str(source / TILES[2][1]),
        "standard_reference_sha256": sha256(source / TILES[2][1]),
        "normalization": normalization_report,
        "before_seams": before_seams,
        "after_seams": after_seams,
        "seam_report": seam_report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "biases": normalization_report["biases"],
                      "seams": seam_report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.source, args.output)
