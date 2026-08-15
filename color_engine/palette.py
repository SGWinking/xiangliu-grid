from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color_space import oklab_to_rgb, rgb_to_oklab


def _sample_image(image: Image.Image, max_edge: int = 640) -> np.ndarray:
    copy = image.convert("RGB").copy()
    copy.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return np.asarray(copy, dtype=np.uint8).reshape(-1, 3)


def _semantic_name(hue: float) -> str:
    degree = (math.degrees(hue) + 360.0) % 360.0
    if degree < 25 or degree >= 345:
        return "朱红"
    if degree < 70:
        return "土黄 / 暖褐"
    if degree < 155:
        return "石绿"
    if degree < 215:
        return "青色"
    if degree < 285:
        return "石青"
    return "紫褐"


def build_palette_profile(images: Iterable[Image.Image], bins: int = 24) -> dict:
    images = list(images)
    if not images:
        raise ValueError("至少需要一张标准色彩图")
    rgb = np.concatenate([_sample_image(image) for image in images], axis=0)
    lab = rgb_to_oklab(rgb)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.mod(np.arctan2(lab[:, 2], lab[:, 1]), 2 * np.pi)
    # 灰尘、纸底和黑线不参与有彩色色族统计，但会单独记录中性色范围。
    colorful = chroma >= 0.025
    indices = np.floor(hue / (2 * np.pi) * bins).astype(int) % bins
    families = []
    for index in range(bins):
        mask = colorful & (indices == index)
        count = int(mask.sum())
        if count:
            weights = np.maximum(chroma[mask], 0.01)
            mean_a = float(np.average(lab[mask, 1], weights=weights))
            mean_b = float(np.average(lab[mask, 2], weights=weights))
            mean_hue = float(np.mod(np.arctan2(mean_b, mean_a), 2 * np.pi))
            family = {
                "index": index,
                "name": _semantic_name(mean_hue),
                "count": count,
                "hue": round(mean_hue, 8),
                "chroma_median": round(float(np.median(chroma[mask])), 8),
                "chroma_q25": round(float(np.quantile(chroma[mask], 0.25)), 8),
                "chroma_q75": round(float(np.quantile(chroma[mask], 0.75)), 8),
                "lightness_median": round(float(np.median(lab[mask, 0])), 8),
            }
        else:
            family = {"index": index, "name": "", "count": 0, "hue": 0.0, "chroma_median": 0.0,
                      "chroma_q25": 0.0, "chroma_q75": 0.0, "lightness_median": 0.0}
        families.append(family)
    # 色卡按色相铺开，避免大面积墙底色把红、青、蓝等小面积颜料挤出预览。
    populated = sorted((f for f in families if f["count"]), key=lambda f: f["hue"])
    swatches = []
    for family in populated:
        preview_lab = np.array([[[0.68, math.cos(family["hue"]) * family["chroma_median"],
                                  math.sin(family["hue"]) * family["chroma_median"]]]])
        swatches.append({**family, "rgb": oklab_to_rgb(preview_lab)[0, 0].tolist()})
    return {
        "version": 1,
        "color_space": "OKLab",
        "bins": bins,
        "reference_count": len(images),
        "neutral_chroma_threshold": 0.025,
        "families": families,
        "swatches": swatches,
    }


def save_palette_profile(profile: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_palette_profile(path: Path) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("version") != 1 or profile.get("color_space") != "OKLab":
        raise ValueError("不支持的标准色谱文件")
    return profile


def render_palette_profile(profile: dict, path: Path) -> Path:
    swatches = profile.get("swatches", [])
    width, item_h = 760, 74
    canvas = Image.new("RGB", (width, max(150, 74 + item_h * len(swatches))), (238, 232, 220))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((24, 20), "Xiangliu Grid Standard Palette / OKLab", fill=(35, 38, 40), font=font)
    for row, swatch in enumerate(swatches):
        y = 58 + row * item_h
        rgb = tuple(swatch["rgb"])
        draw.rounded_rectangle((24, y, 148, y + 54), radius=8, fill=rgb, outline=(70, 70, 70))
        draw.text((170, y + 8), f"{swatch['name']}  RGB {rgb}", fill=(35, 38, 40), font=font)
        draw.text((170, y + 29), f"hue-bin {swatch['index']:02d}  pixels {swatch['count']}",
                  fill=(90, 88, 82), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _nearest_family_map(profile: dict) -> list[dict | None]:
    bins = int(profile["bins"])
    populated = [f for f in profile["families"] if f["count"]]
    result = []
    for index in range(bins):
        if not populated:
            result.append(None)
            continue
        result.append(min(populated, key=lambda f: min((f["index"] - index) % bins, (index - f["index"]) % bins)))
    return result


def apply_palette_profile(
    image: Image.Image,
    profile: dict,
    strength: float = 0.85,
    neutral_protection: float = 0.9,
    skin_protection: float = 0.55,
    gold_protection: float = 0.7,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    output = np.empty_like(rgb)
    chunk_rows = 128
    bins = int(profile["bins"])
    hist_edges = np.linspace(0.0, 0.36, 513, dtype=np.float32)
    histograms = np.zeros((bins, 512), dtype=np.int64)
    for y0 in range(0, rgb.shape[0], chunk_rows):
        y1 = min(rgb.shape[0], y0 + chunk_rows)
        lab = rgb_to_oklab(rgb[y0:y1])
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        hue = np.mod(np.arctan2(lab[..., 2], lab[..., 1]), 2 * np.pi)
        indices = np.floor(hue / (2 * np.pi) * bins).astype(np.int16) % bins
        for index in range(bins):
            values = chroma[indices == index]
            if values.size:
                histograms[index] += np.histogram(values, bins=hist_edges)[0]
    source_medians = np.zeros(bins, dtype=np.float32)
    centers = (hist_edges[:-1] + hist_edges[1:]) * 0.5
    for index, histogram in enumerate(histograms):
        total = int(histogram.sum())
        if total:
            source_medians[index] = centers[min(int(np.searchsorted(np.cumsum(histogram), (total + 1) // 2)), 511)]
    for y0 in range(0, rgb.shape[0], chunk_rows):
        y1 = min(rgb.shape[0], y0 + chunk_rows)
        output[y0:y1] = _apply_palette_chunk(
            rgb[y0:y1], profile, strength, neutral_protection, skin_protection, gold_protection,
            source_medians,
        )
    return Image.fromarray(output, "RGB")


def _apply_palette_chunk(
    rgb: np.ndarray,
    profile: dict,
    strength: float,
    neutral_protection: float,
    skin_protection: float,
    gold_protection: float,
    source_medians: np.ndarray,
) -> np.ndarray:
    lab = rgb_to_oklab(rgb)
    light = lab[..., 0]
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.mod(np.arctan2(b, a), 2 * np.pi)
    bins = int(profile["bins"])
    indices = np.floor(hue / (2 * np.pi) * bins).astype(int) % bins
    family_map = _nearest_family_map(profile)
    out_a, out_b = a.copy(), b.copy()
    base_weight = np.full(chroma.shape, np.clip(strength, 0.0, 1.0), dtype=np.float32)
    neutral_threshold = float(profile.get("neutral_chroma_threshold", 0.025))
    neutral_mix = np.clip((chroma - neutral_threshold * 0.55) / max(neutral_threshold, 1e-6), 0.0, 1.0)
    base_weight *= 1.0 - np.clip(neutral_protection, 0.0, 1.0) * (1.0 - neutral_mix)
    degrees = np.mod(np.degrees(hue), 360.0)
    skin = (degrees >= 25) & (degrees <= 75) & (light >= 0.48) & (chroma >= 0.025) & (chroma <= 0.16)
    gold = (degrees >= 70) & (degrees <= 115) & (light >= 0.52) & (chroma >= 0.07)
    base_weight[skin] *= 1.0 - np.clip(skin_protection, 0.0, 1.0)
    base_weight[gold] *= 1.0 - np.clip(gold_protection, 0.0, 1.0)
    for index, family in enumerate(family_map):
        if family is None:
            continue
        mask = indices == index
        if not np.any(mask):
            continue
        # 保留目标内部的浓淡起伏，只把该色族的中心拉向标准图。
        source_med = float(source_medians[index])
        target_med = float(family["chroma_median"])
        scale = np.clip(target_med / max(source_med, 0.02), 0.55, 1.8)
        desired_c = np.clip(chroma[mask] * scale, 0.0, 0.32)
        desired_a = np.cos(float(family["hue"])) * desired_c
        desired_b = np.sin(float(family["hue"])) * desired_c
        weight = base_weight[mask]
        out_a[mask] = a[mask] * (1.0 - weight) + desired_a * weight
        out_b[mask] = b[mask] * (1.0 - weight) + desired_b * weight
    result = np.stack((light, out_a, out_b), axis=-1)
    return oklab_to_rgb(result)
