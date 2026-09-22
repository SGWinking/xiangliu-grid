from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color_space import oklab_to_rgb, rgb_to_oklab

NEUTRAL_CHROMA_DEFAULT = 0.025
BINS_DEFAULT = 24
# 三档明度的档名（支持中英文）
LEVEL_NAMES = {"浅色": "浅色", "light": "浅色", "浅": "浅色",
               "基准色": "基准色", "base": "基准色", "基准": "基准色",
               "深色": "深色", "dark": "深色", "深": "深色"}


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


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = str(hex_str).lstrip("#")
    if len(hex_str) != 6:
        raise ValueError(f"无法解析色值: {hex_str}")
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def _entry_rgb(entry: dict) -> tuple[int, int, int]:
    if isinstance(entry.get("rgb"), (list, tuple)) and len(entry["rgb"]) >= 3:
        return tuple(int(v) for v in entry["rgb"][:3])
    if entry.get("hex"):
        return _hex_to_rgb(entry["hex"])
    raise ValueError(f"色卡条目缺少 rgb/hex：{entry.get('name', '?')}")


def _parse_entries(entries: list[dict]) -> list[dict]:
    """把色卡条目（hex/rgb）转成 OKLab 统计，返回带 L/a/b/hue/chroma 的条目。"""
    parsed = []
    for entry in entries:
        rgb = np.asarray(_entry_rgb(entry), dtype=np.float32).reshape(1, 1, 3)
        L, a, b = rgb_to_oklab(rgb)[0, 0]
        parsed.append({
            **entry,
            "L": float(L),
            "a": float(a),
            "b": float(b),
            "chroma": float(np.hypot(a, b)),
            "hue": float(np.mod(np.arctan2(b, a), 2 * np.pi)),
        })
    return parsed


def _levels_from_entries(members: list[dict], family_hue: float, bins: int = BINS_DEFAULT) -> list[dict]:
    """把族内各条目按 level 分档（每档记录 lightness/chroma/hue，供 18 色网格矫正）；
    没有 level 字段时按明度三等分。"""
    levels = []
    for m in members:
        level_raw = str(m.get("level") or "")
        level = LEVEL_NAMES.get(level_raw) or LEVEL_NAMES.get(level_raw.lower(), "")
        if level:
            levels.append({"level": level, "lightness": m["L"], "chroma": m["chroma"], "hue": m["hue"]})
    if len(levels) >= 2:
        return levels
    # 无 level：按明度三等分，浅=高 L、深=低 L
    lights = sorted(m["L"] for m in members)
    lo, hi = lights[0], lights[-1]
    if hi - lo < 1e-4:
        return [{"level": "基准色", "lightness": (lo + hi) / 2,
                 "chroma": _quantile([m["chroma"] for m in members], 0.5),
                 "hue": float(np.median([m["hue"] for m in members]))}]
    mid1 = lo + (hi - lo) / 3
    mid2 = lo + 2 * (hi - lo) / 3
    buckets = {"浅色": [], "基准色": [], "深色": []}
    for m in members:
        if m["L"] >= mid2:
            buckets["浅色"].append(m)
        elif m["L"] >= mid1:
            buckets["基准色"].append(m)
        else:
            buckets["深色"].append(m)
    return [
        {"level": name, "lightness": _quantile([m["L"] for m in group], 0.5),
         "chroma": _quantile([m["chroma"] for m in group], 0.5),
         "hue": float(np.median([m["hue"] for m in group]))}
        for name, group in buckets.items() if group
    ]


def _profile_from_entries(entries: list[dict], bins: int = BINS_DEFAULT,
                          neutral_chroma_threshold: float = NEUTRAL_CHROMA_DEFAULT,
                          cave: str = "", period: str = "") -> dict:
    """把任意色卡条目列表转成工具内部 profile。
    - 有 family 字段：按族聚合，锚点色相取"基准色"（缺省取族内第一条）。
    - 无 family 字段：按色相分桶聚合。
    - 白/黑与低彩度条目（chroma 低于阈值×0.4）不进有彩色色族，记为 neutrals。
    """
    parsed = _parse_entries(entries)
    families = [
        {"index": i, "name": "", "count": 0, "hue": 0.0, "chroma_median": 0.0,
         "chroma_q25": 0.0, "chroma_q75": 0.0, "lightness_median": 0.0, "levels": []}
        for i in range(bins)
    ]
    neutrals = []

    by_family: dict[str, list[dict]] = {}
    for p in parsed:
        fam = str(p.get("family") or "").strip()
        is_neutral_family = fam in {"白", "黑", "灰", "white", "black", "gray", "grey"}
        if is_neutral_family or p["chroma"] < neutral_chroma_threshold * 0.4:
            neutrals.append(p)
            continue
        key = fam if fam else f"__bin{p['hue']:.3f}"
        by_family.setdefault(key, []).append(p)

    for key, members in by_family.items():
        fam = key if not key.startswith("__bin") else ""
        if fam:
            base = next((m for m in members if str(m.get("level") or "").lower() in
                         {"基准色", "base", "基准", "标准色"}), members[0])
            anchor_hue = base["hue"]
        else:
            anchor_hue = float(np.median([m["hue"] for m in members]))
        bin_index = int(math.floor(anchor_hue / (2 * np.pi) * bins)) % bins
        chromas = [m["chroma"] for m in members]
        lights = [m["L"] for m in members]
        families[bin_index] = {
            "index": bin_index,
            "name": fam or _semantic_name(anchor_hue),
            "count": len(members),
            "hue": round(anchor_hue, 8),
            "chroma_median": round(_quantile(chromas, 0.5), 8),
            "chroma_q25": round(_quantile(chromas, 0.25), 8),
            "chroma_q75": round(_quantile(chromas, 0.75), 8),
            "lightness_median": round(_quantile(lights, 0.5), 8),
            "levels": _levels_from_entries(members, anchor_hue, bins),
        }

    return {
        "version": 1,
        "color_space": "OKLab",
        "bins": bins,
        "reference_count": len(entries),
        "neutral_chroma_threshold": neutral_chroma_threshold,
        "cave": cave,
        "period": period,
        "source": "结构化色卡/色卡列表转换",
        "families": families,
        "swatches": _swatches_from_families(families, parsed),
        "neutrals": [
            {"name": n.get("name", ""), "hex": str(n.get("hex", "")), "L": round(n["L"], 6),
             "chroma": round(n["chroma"], 6), "level": n.get("level", "")}
            for n in sorted(neutrals, key=lambda x: x["L"], reverse=True)
        ],
    }


def _swatches_from_families(families: list[dict], parsed: list[dict]) -> list[dict]:
    swatches = []
    for f in families:
        if not f["count"]:
            continue
        name = f["name"]
        candidate = next((m for m in parsed if m.get("family") == name and str(m.get("level") or "").lower()
                          in {"基准色", "base", "基准", "标准色"}), None)
        if candidate is None:
            candidate = next((m for m in parsed if m.get("family") == name), None)
        if candidate is None:
            candidate = next((m for m in parsed if abs(m["hue"] - f["hue"]) < 0.2), None)
        rgb = _entry_rgb(candidate) if candidate else (0, 0, 0)
        swatches.append({**f, "rgb": list(rgb)})
    return swatches


def build_palette_profile(images: Iterable[Image.Image], bins: int = 24,
                          white_balance: bool = True) -> dict:
    images = list(images)
    if not images:
        raise ValueError("至少需要一张标准色彩图")
    rgb = np.concatenate([_sample_image(image) for image in images], axis=0)
    lab = rgb_to_oklab(rgb)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    # 白平衡：从彩度最低的像素（纸底/灰）估计全局色偏并扣除，
    # 避免标准图本身偏暖导致各族 hue 偏暖。用低彩度分位而非固定阈值，更鲁棒。
    cast_a, cast_b = 0.0, 0.0
    if white_balance:
        n = len(chroma)
        k = min(n, max(32, int(n * 0.10)))
        if k >= 32:
            low_idx = np.argpartition(chroma, k)[:k]
            cast_a = float(np.median(lab[low_idx, 1]))
            cast_b = float(np.median(lab[low_idx, 2]))
            lab = lab.copy()
            lab[:, 1] -= cast_a
            lab[:, 2] -= cast_b
            chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.mod(np.arctan2(lab[:, 2], lab[:, 1]), 2 * np.pi)
    # 灰尘、纸底和黑线不参与有彩色色族统计，但会单独记录中性色范围。
    colorful = chroma >= NEUTRAL_CHROMA_DEFAULT
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
            # 族内按明度三等分，保留三档 levels（供上色时按明度插值色相/彩度）
            order = np.argsort(lab[mask, 0])
            sorted_l = lab[mask, 0][order]
            sorted_c = chroma[mask][order]
            sorted_h = hue[mask][order]
            n = len(sorted_l)
            third = max(1, n // 3)
            groups = [
                ("浅色", slice(max(0, n - third), n)),      # 高 L
                ("基准色", slice(max(0, n // 2 - third // 2), min(n, n // 2 + third - third // 2))),
                ("深色", slice(0, third)),                  # 低 L
            ]
            levels = []
            for level_name, sl in groups:
                if sl.start >= sl.stop:
                    continue
                levels.append({
                    "level": level_name,
                    "lightness": round(float(np.median(sorted_l[sl])), 8),
                    "chroma": round(float(np.median(sorted_c[sl])), 8),
                    "hue": round(float(np.median(sorted_h[sl])), 8),
                })
            family = {
                "index": index,
                "name": _semantic_name(mean_hue),
                "count": count,
                "hue": round(mean_hue, 8),
                "chroma_median": round(float(np.median(chroma[mask])), 8),
                "chroma_q25": round(float(np.quantile(chroma[mask], 0.25)), 8),
                "chroma_q75": round(float(np.quantile(chroma[mask], 0.75)), 8),
                "lightness_median": round(float(np.median(lab[mask, 0])), 8),
                "levels": levels,
            }
        else:
            family = {"index": index, "name": "", "count": 0, "hue": 0.0, "chroma_median": 0.0,
                      "chroma_q25": 0.0, "chroma_q75": 0.0, "lightness_median": 0.0, "levels": []}
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
        "neutral_chroma_threshold": NEUTRAL_CHROMA_DEFAULT,
        "cast": [round(cast_a, 8), round(cast_b, 8)],
        "white_balance": white_balance,
        "families": families,
        "swatches": swatches,
    }


def save_palette_profile(profile: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_palette_profile(path: Path) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    # 1) 工具自家生成的格式
    if profile.get("version") == 1 and profile.get("color_space") == "OKLab":
        return profile
    # 2) 结构化色卡：{cave, period, colors:[{family, level, name, hex}]}
    if isinstance(profile.get("colors"), list) and profile["colors"]:
        return _profile_from_entries(
            profile["colors"],
            cave=str(profile.get("cave") or ""),
            period=str(profile.get("period") or ""),
        )
    # 3) 简单 swatch 列表：{swatches:[{name, rgb|hex}]}
    if isinstance(profile.get("swatches"), list) and profile["swatches"]:
        return _profile_from_entries(profile["swatches"])
    raise ValueError(
        "色卡格式无法识别：请用「生成标准色谱」按钮生成，"
        "或提供含 colors（family/level/name/hex）或 swatches（name/rgb）字段的 JSON"
    )


def _load_cjk_font(size: int = 16):
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_palette_profile(profile: dict, path: Path) -> Path:
    """渲染完整色板：每个有彩色族的三档（浅/基准/深）+ 全部中性色（白/黑/灰）。"""
    families = [f for f in profile.get("families", []) if f["count"]]
    neutrals = profile.get("neutrals", [])
    items: list[tuple[str, tuple[int, int, int]]] = []
    for f in families:
        name = f["name"]
        levels = f.get("levels")
        if levels:
            for lv in sorted(levels, key=lambda x: float(x["lightness"]), reverse=True):
                L = float(lv["lightness"])
                c = float(lv["chroma"])
                h = float(lv.get("hue", f["hue"]))
                lab = np.array([[[L, c * math.cos(h), c * math.sin(h)]]], dtype=np.float32)
                rgb = tuple(int(v) for v in oklab_to_rgb(lab)[0, 0])
                items.append((f"{name}·{lv['level']}", rgb))
        else:
            lab = np.array([[[0.68, math.cos(f["hue"]) * f["chroma_median"],
                              math.sin(f["hue"]) * f["chroma_median"]]]])
            rgb = tuple(int(v) for v in oklab_to_rgb(lab)[0, 0])
            items.append((name, rgb))
    for n in neutrals:
        try:
            rgb = _hex_to_rgb(str(n.get("hex", "")))
        except Exception:
            rgb = (0, 0, 0)
        items.append((f"{n['name']}（中性）", rgb))
    total = len(items)
    width, item_h = 760, 74
    canvas = Image.new("RGB", (width, max(150, 74 + item_h * total)), (238, 232, 220))
    draw = ImageDraw.Draw(canvas)
    font = _load_cjk_font(16)
    draw.text((24, 20), "Xiangliu Grid Standard Palette / OKLab", fill=(35, 38, 40), font=font)
    for row, (label, rgb) in enumerate(items):
        y = 58 + row * item_h
        draw.rounded_rectangle((24, y, 148, y + 54), radius=8, fill=rgb, outline=(70, 70, 70))
        draw.text((170, y + 8), f"{label}  RGB {rgb}", fill=(35, 38, 40), font=font)
        draw.text((170, y + 29), f"swatch {row + 1:02d} / {total}", fill=(90, 88, 82), font=font)
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


def level_chroma(family: dict, light) -> np.ndarray:
    """三档明度插值：按像素明度 L 在族 levels（浅→深）之间线性插值目标彩度。
    无 levels 时回退到族 chroma_median。"""
    return level_hue_chroma(family, light)[1]


def level_hue_chroma(family: dict, light) -> tuple[np.ndarray, np.ndarray]:
    """18 色网格矫正核心：按像素明度 L 在族 levels（浅/基准/深）之间插值，
    返回 (目标色相, 目标彩度)。色相先展开到族基准附近避免 0/360 环绕。
    无 levels 时回退到族 hue + chroma_median。"""
    levels = family.get("levels")
    light = np.asarray(light, dtype=np.float32)
    if not levels or len(levels) < 2:
        hue0 = float(family.get("hue", 0.0))
        chroma0 = float(family.get("chroma_median", 0.03))
        return np.full(light.shape, hue0, dtype=np.float32), np.full(light.shape, chroma0, dtype=np.float32)
    ls = sorted(levels, key=lambda x: float(x["lightness"]), reverse=True)  # 浅→深
    base = float(ls[0]["hue"])

    def wrap(h):
        # 把 h 展开到 base 附近（避免 0/360 环绕造成错误插值）
        return base + ((h - base + np.pi) % (2 * np.pi) - np.pi)

    out_h = np.empty_like(light)
    out_c = np.empty_like(light)
    h_hi, c_hi = wrap(float(ls[0]["hue"])), float(ls[0]["chroma"])
    out_h[light >= float(ls[0]["lightness"])] = h_hi
    out_c[light >= float(ls[0]["lightness"])] = c_hi
    for i in range(len(ls) - 1):
        l_hi, c_hi = float(ls[i]["lightness"]), float(ls[i]["chroma"])
        l_lo, c_lo = float(ls[i + 1]["lightness"]), float(ls[i + 1]["chroma"])
        h_hi, h_lo = wrap(float(ls[i]["hue"])), wrap(float(ls[i + 1]["hue"]))
        span = max(l_hi - l_lo, 1e-6)
        frac = np.clip((light - l_lo) / span, 0.0, 1.0)
        mask = (light < l_hi) & (light >= l_lo)
        out_h[mask] = h_lo + (h_hi - h_lo) * frac[mask]
        out_c[mask] = c_lo + (c_hi - c_lo) * frac[mask]
    out_h[light < float(ls[-1]["lightness"])] = wrap(float(ls[-1]["hue"]))
    out_c[light < float(ls[-1]["lightness"])] = float(ls[-1]["chroma"])
    return out_h, out_c


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
    neutral_threshold = float(profile.get("neutral_chroma_threshold", NEUTRAL_CHROMA_DEFAULT))
    # 中性保护：chroma < 阈值×0.55 时几乎不动，阈值附近平滑过渡
    neutral_mix = np.clip((chroma - neutral_threshold * 0.55) / max(neutral_threshold, 1e-6), 0.0, 1.0)
    base_weight *= 1.0 - np.clip(neutral_protection, 0.0, 1.0) * (1.0 - neutral_mix)
    degrees = np.mod(np.degrees(hue), 360.0)
    skin = (degrees >= 25) & (degrees <= 75) & (light >= 0.48) & (chroma >= 0.025) & (chroma <= 0.16)
    gold = (degrees >= 70) & (degrees <= 115) & (light >= 0.52) & (chroma >= 0.07)
    base_weight[skin] *= 1.0 - np.clip(skin_protection, 0.0, 1.0)
    base_weight[gold] *= 1.0 - np.clip(gold_protection, 0.0, 1.0)
    # 低彩度像素（近中性/纸底/阴影）少拉，避免整张图蒙上色相
    colorful_confidence = np.clip((chroma - neutral_threshold) / 0.02, 0.0, 1.0)
    for index, family in enumerate(family_map):
        if family is None:
            continue
        mask = indices == index
        if not np.any(mask):
            continue
        # 保留目标内部的浓淡起伏，只把该色族的彩度与色相温和地拉向标准图。
        source_med = float(source_medians[index])
        target_med = float(family["chroma_median"])
        raw_scale = np.clip(target_med / max(source_med, 0.02), 0.55, 1.35)
        # 中性像素不做彩度缩放（scale 向 1 收拢）
        scale = 1.0 + (raw_scale - 1.0) * colorful_confidence[mask]
        # 目标彩度：三档明度插值，与"源彩度×scale"按权重混合
        level_c = level_chroma(family, light[mask])
        weight = base_weight[mask] * colorful_confidence[mask]
        mix_c = chroma[mask] * scale
        desired_c = np.clip(mix_c * (1.0 - weight) + level_c * weight, 0.0, 0.36)
        # 色相有界微调（最大 ±10°），低彩度像素几乎不动 → 红色不会被拉成土黄
        delta = np.mod(family["hue"] - hue[mask] + np.pi, 2 * np.pi) - np.pi
        pull = np.clip(delta * weight * 0.5, -0.175, 0.175)
        new_hue = hue[mask] + pull
        out_a[mask] = np.cos(new_hue) * desired_c
        out_b[mask] = np.sin(new_hue) * desired_c
    result = np.stack((light, out_a, out_b), axis=-1)
    return oklab_to_rgb(result)
