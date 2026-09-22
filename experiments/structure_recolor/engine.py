from __future__ import annotations

import math
from dataclasses import dataclass
import hashlib

import numpy as np
from PIL import Image
from PIL import ImageFilter

from color_engine.color_space import oklab_to_rgb, rgb_to_oklab
from color_engine.palette import level_hue_chroma


@dataclass
class TileLayers:
    structure_l: np.ndarray
    source_hue: np.ndarray
    source_chroma: np.ndarray
    family_index: np.ndarray
    protection: np.ndarray
    structure_preview: Image.Image


def decompose_tile(image: Image.Image, bins: int = 24) -> TileLayers:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = rgb_to_oklab(rgb)
    light = lab[..., 0].astype(np.float32)
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b).astype(np.float32)
    hue = np.mod(np.arctan2(b, a), 2 * np.pi).astype(np.float32)
    family = (np.floor(hue / (2 * np.pi) * bins).astype(np.int16) % bins)
    degrees = np.mod(np.degrees(hue), 360.0)
    # 中性保护：墙底/纸底/土黄底（低彩度暖色）更强保护，避免饱和度增益把它们调黄。
    # 阈值从 0.035 提到 0.05：chroma<0.02 完全保护，0.02~0.05 平滑过渡，>0.05（颜料）不保护。
    neutral = np.clip((0.05 - chroma) / 0.03, 0.0, 1.0)
    skin = ((degrees >= 25) & (degrees <= 75) & (light >= 0.48) & (chroma <= 0.17)).astype(np.float32)
    gold = ((degrees >= 70) & (degrees <= 115) & (light >= 0.52) & (chroma >= 0.07)).astype(np.float32)
    protection = np.maximum(neutral * 0.95, np.maximum(skin * 0.55, gold * 0.72)).astype(np.float32)
    gray = oklab_to_rgb(np.stack((light, np.zeros_like(light), np.zeros_like(light)), axis=-1))
    return TileLayers(light, hue, chroma, family, protection, Image.fromarray(gray, "RGB"))


def _nearest_family(profile: dict, index: int) -> tuple[dict | None, float]:
    bins = int(profile["bins"])
    populated = [family for family in profile["families"] if family["count"]]
    if not populated:
        return None, 0.0
    nearest = min(populated, key=lambda family: min((family["index"] - index) % bins, (index - family["index"]) % bins))
    dist_bins = min((nearest["index"] - index) % bins, (index - nearest["index"]) % bins)
    return nearest, dist_bins * (360.0 / bins)


def recolor_tile(
    layers: TileLayers,
    corrected_l: np.ndarray,
    profile: dict,
    strength: float = 0.85,
    saturation_gain: float = 1.0,
    min_chroma: float = 0.035,
    preserve_local_hue: bool = True,
    max_hue_shift_deg: float = 45.0,
) -> Image.Image:
    """18 色网格矫正：按"源色相分族 + 明度分档（浅/基准/深）"把原图颜色矫正到
    色卡对应档位颜色——明度来自灰阶归一化，色相与彩度向色卡档位矫正。
    preserve_local_hue=True（默认）：色相只做有界矫正（max_hue_shift_deg），
    且按源色相到族的距离衰减（中间色保留，不硬压缩）；
    False：色相直接拉向族色相（强统一，消色相接缝，但会压缩中间色）。"""
    if corrected_l.shape != layers.structure_l.shape:
        raise ValueError("校正亮度层尺寸与结构底稿不一致")
    out_a = np.zeros_like(layers.source_chroma)
    out_b = np.zeros_like(layers.source_chroma)
    global_strength = np.clip(strength, 0.0, 1.0)
    bins = int(profile["bins"])
    for index in range(bins):
        mask = layers.family_index == index
        if not np.any(mask):
            continue
        family, dist = _nearest_family(profile, index)
        if family is None:
            continue
        src_hue = layers.source_hue[mask]
        src_c = layers.source_chroma[mask]
        src_l = np.asarray(corrected_l, dtype=np.float32)[mask]
        # 可信度：源图彩度越高越信任色相分桶；近中性（纸底/阴影/墨线）不做彩色矫正。
        # 阈值放宽（0.015 起 / 0.04 满）：AI 修复块整体偏灰时也能向色卡靠，
        # 纸底等真正中性像素 confidence≈0，混合权重 w≈0，仍完全不动。
        colorful_confidence = np.clip((layers.source_chroma[mask] - 0.015) / 0.025, 0.0, 1.0)
        # 目标 = 色卡该族该档的 (色相, 彩度)，按明度三档插值（18 色网格）
        level_hue, level_c = level_hue_chroma(family, src_l)
        # 饱和度增益的"资格"：
        #   - 蓝/绿/红等非暖带：用源彩度相对该族基准彩度的比例（蓝族基准天生低也能变浓）
        #   - 黄/橙暖带（40~90°，土黄底/墙底所在）：用严格绝对阈值，墙面不被调黄
        family_median = max(float(family.get("chroma_median", 0.03)), 0.02)
        relative = layers.source_chroma[mask] / family_median
        absolute_gate = np.clip((layers.source_chroma[mask] - 0.015) / 0.010, 0.0, 1.0)
        elig_rel = np.clip((relative - 0.4) / 0.6, 0.0, 1.0) * absolute_gate
        src_degrees = np.mod(np.degrees(layers.source_hue[mask]), 360.0)
        warm = (src_degrees >= 40.0) & (src_degrees <= 90.0)
        elig_abs = np.clip((layers.source_chroma[mask] - 0.050) / 0.020, 0.0, 1.0) * absolute_gate
        gain_eligibility = np.where(warm, elig_abs, elig_rel)
        level_c = level_c * (1.0 + (max(saturation_gain, 0.0) - 1.0) * gain_eligibility)
        # 轻微的亮度调制模拟颜料在暗部更浓，幅度温和（暗部 +3%）
        light_modulation = np.clip(1.03 - (src_l - 0.52) * 0.25, 0.97, 1.06)
        protected_saturation = 1.0 - layers.protection[mask] * 0.35
        # 目标彩度 = 色卡档位满值（confidence 不衰减目标，否则源图偏灰时永远靠不近色卡）。
        # confidence 只在下方混合权重 w 里控制"混合多少"，近中性像素 w→0 仍完全不动。
        desired_c = np.clip(
            level_c * light_modulation * protected_saturation,
            0.0, 0.40,
        )
        # 色相矫正：默认向色卡档位色相做有界矫正，且按"源色相到族"的距离衰减——
        # 族内（距离近）充分矫正，族间中间色（距离远）保留原色相，避免硬压缩。
        if preserve_local_hue:
            delta = np.mod(level_hue - src_hue + np.pi, 2 * np.pi) - np.pi
            dist_rad = math.radians(max(dist, 0.0))
            max_shift = math.radians(max_hue_shift_deg)
            # 距离衰减：族内（<30°）几乎全量矫正；>60° 不矫正
            attenuation = np.clip(1.0 - dist_rad / math.radians(60.0), 0.0, 1.0)
            pull = np.clip(delta * global_strength * colorful_confidence * attenuation,
                           -max_shift, max_shift)
            new_hue = src_hue + pull
        else:
            new_hue = level_hue
        # 彩度：源彩度与色卡档位彩度按可信度混合，保留原图浓度的一部分
        w = global_strength * colorful_confidence
        final_c = src_c * (1.0 - w) + desired_c * w
        out_a[mask] = np.cos(new_hue) * final_c
        out_b[mask] = np.sin(new_hue) * final_c
    result = np.stack((np.asarray(corrected_l, dtype=np.float32), out_a, out_b), axis=-1)
    return Image.fromarray(oklab_to_rgb(result), "RGB")


def _lowpass(light: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.clip(np.asarray(light) * 255.0, 0, 255).astype(np.uint8), "L")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0


def joint_low_frequency_normalize(
    placements: list[tuple[str, int, int, np.ndarray]],
    canvas_size: tuple[int, int],
    overlap: int = 20,
    strength: float = 0.9,
) -> tuple[list[tuple[str, int, int, np.ndarray]], dict]:
    if not placements:
        return [], {"biases": {}}
    scale = 0.125
    canvas_w, canvas_h = canvas_size
    small_w, small_h = max(1, round(canvas_w * scale)), max(1, round(canvas_h * scale))
    value_sum = np.zeros((small_h, small_w), dtype=np.float32)
    weight_sum = np.zeros((small_h, small_w), dtype=np.float32)
    small_tiles = {}
    for tile_id, x, y, light in placements:
        h, w = light.shape
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        small = np.asarray(Image.fromarray(light, "F").resize((sw, sh), Image.Resampling.BILINEAR), dtype=np.float32)
        # 先去掉线条等高频，建立整幅画共同的低频底色场。
        low = _lowpass(small, max(3.0, min(sw, sh) / 32.0))
        sx, sy = round(x * scale), round(y * scale)
        small_tiles[tile_id] = (sx, sy, low)
        value_sum[sy:sy + sh, sx:sx + sw] += low
        weight_sum[sy:sy + sh, sx:sx + sw] += 1.0
    composite = value_sum / np.maximum(weight_sum, 1.0)
    # 缺块空洞（weight_sum==0）用已有像素的中位数填充，避免 0 拉低邻近块的低频场（局部拼合可容忍缺块）
    if np.any(weight_sum == 0):
        filled = composite[weight_sum > 0]
        fill_value = float(np.median(filled)) if filled.size else 0.5
        composite = composite.copy()
        composite[weight_sum == 0] = fill_value
    # 大半径只留下跨图块连续的光照/墙体底色趋势；矩形块状态被平均掉。
    target_field = _lowpass(composite, max(12.0, min(small_w, small_h) / 18.0))

    corrected = []
    report = {"biases": {}, "correction_ranges": {}, "method": "continuous_global_field", "overlap": overlap}
    for tile_id, x, y, light in placements:
        sx, sy, tile_low = small_tiles[tile_id]
        sh, sw = tile_low.shape
        target = target_field[sy:sy + sh, sx:sx + sw]
        correction_small = np.clip((target - tile_low) * strength, -0.10, 0.10)
        correction = np.asarray(
            Image.fromarray(correction_small, "F").resize((light.shape[1], light.shape[0]), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        adjusted = np.clip(light + correction, 0.0, 1.0).astype(np.float32)
        corrected.append((tile_id, x, y, adjusted))
        report["biases"][tile_id] = round(float(np.median(correction)), 6)
        report["correction_ranges"][tile_id] = [round(float(np.quantile(correction, 0.05)), 6),
                                                   round(float(np.quantile(correction, 0.95)), 6)]
    return corrected, report


def _local_line_response(light: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local_base = _lowpass(light, max(5.0, min(light.shape) / 42.0))
    dark_residual = np.maximum(local_base - light, 0.0)
    # 线条拥有较强的局部暗残差，同时在至少一个方向上很窄；污渍通常没有这种尖锐响应。
    small_blur = _lowpass(light, 1.2)
    sharp_dark = np.maximum(small_blur - light, 0.0)
    line_score = dark_residual * np.clip(sharp_dark / 0.018, 0.0, 1.0)
    return local_base, line_score


def _shared_ink_lightness(
    residual: np.ndarray, q10: float, q90: float, ink_lightness: float = 0.48
) -> np.ndarray:
    span = max(q90 - q10, 0.035)
    response = np.clip((residual - q10) / span, 0.0, 1.0)
    # 目标值为窄带中心；保留 0.01 的笔触层次，但不改变墨线支持区域。
    return (ink_lightness + 0.005 - 0.010 * response).astype(np.float32)


def _ink_shape_support(reliable: np.ndarray, residual: np.ndarray, expand_px: int = 2) -> np.ndarray:
    # 将可靠核心向外扩若干像素，接住同一笔画的抗锯齿/灰边；远处纸纹与暗面不参与。
    if expand_px <= 0:
        return reliable & (residual > 1e-6)
    core = Image.fromarray(reliable.astype(np.uint8) * 255, "L")
    near_core = np.asarray(core.filter(ImageFilter.MaxFilter(2 * expand_px + 1)), dtype=np.uint8) > 0
    return near_core & (residual > 1e-6)


def joint_grayscale_normalize(
    placements: list[tuple[str, int, int, np.ndarray]],
    canvas_size: tuple[int, int],
    overlap: int = 20,
    ink_lightness: float = 0.48,
    ink_expand_px: int = 2,
) -> tuple[list[tuple[str, int, int, np.ndarray]], dict]:
    if not 0.25 <= ink_lightness <= 0.65:
        raise ValueError("ink_lightness must be between 0.25 and 0.65")
    if not placements:
        return [], {"method": "continuous_base_plus_shared_ink_curve", "tiles": {}}
    low_corrected, low_report = joint_low_frequency_normalize(
        placements, canvas_size, overlap=overlap, strength=0.92
    )
    detected = []
    all_scores = []
    for tile_id, x, y, light in low_corrected:
        local_base, line_score = _local_line_response(light)
        residual = np.maximum(local_base - light, 0.0)
        all_scores.append(line_score.reshape(-1))
        detected.append((tile_id, x, y, light, local_base, line_score, residual))
    shared_threshold = max(0.050, float(np.quantile(np.concatenate(all_scores), 0.88)))
    detected = [item + (item[5] >= shared_threshold,) for item in detected]
    reliable_values = [item[6][item[7]] for item in detected]
    nonempty_values = [values for values in reliable_values if values.size]
    pooled = np.concatenate(nonempty_values) if nonempty_values else np.empty(0, dtype=np.float32)
    input_q10 = float(np.quantile(pooled, 0.10)) if pooled.size else 0.10
    input_q90 = float(np.quantile(pooled, 0.90)) if pooled.size else 0.34
    corrected = []
    tiles_report = {}
    for tile_id, x, y, light, local_base, line_score, residual, reliable in detected:
        values = residual[reliable]
        target_l = _shared_ink_lightness(residual, input_q10, input_q90, ink_lightness)
        # 可靠核心只用于求浓淡；实际缩放覆盖同一线条的完整暗残差横截面。
        shape_support = _ink_shape_support(reliable, residual, ink_expand_px)
        result = light.copy()
        if values.size:
            target_median = float(np.median(target_l[reliable]))
            low_gain, high_gain = 0.0, 8.0
            for _ in range(24):
                trial_gain = (low_gain + high_gain) * 0.5
                trial_median = float(np.median(local_base[reliable] - residual[reliable] * trial_gain))
                if trial_median > target_median:
                    low_gain = trial_gain
                else:
                    high_gain = trial_gain
            ink_depth_gain = (low_gain + high_gain) * 0.5
            result[shape_support] = np.clip(
                local_base[shape_support] - residual[shape_support] * ink_depth_gain, 0.0, 1.0
            )
        else:
            ink_depth_gain = 1.0
        result_base, _ = _local_line_response(result)
        result_residual = np.maximum(result_base - result, 0.0)
        final_values = result[reliable]
        corrected.append((tile_id, x, y, result))
        tiles_report[tile_id] = {
            "ink_response_median": round(float(np.median(values)), 6) if values.size else None,
            "ink_depth_gain": round(float(ink_depth_gain), 6),
            "ink_l_median": round(float(np.median(final_values)), 6) if final_values.size else None,
            "ink_l_p90": round(float(np.quantile(final_values, 0.90)), 6) if final_values.size else None,
            "ink_residual_median": round(float(np.median(result_residual[reliable])), 6) if final_values.size else None,
            "reliable_ink_pixels": int(np.count_nonzero(reliable)),
            "reliable_ink_fraction": round(float(np.mean(reliable)), 6),
            "ink_support_hash": hashlib.sha256(np.packbits(shape_support).tobytes()).hexdigest(),
        }
    return corrected, {
        "method": "continuous_base_plus_shared_ink_curve",
        "low_frequency": low_report,
        "shared_line_score_threshold": round(shared_threshold, 6),
        "shared_response_q10": round(input_q10, 6),
        "shared_response_q90": round(input_q90, 6),
        "ink_lightness": round(float(ink_lightness), 6),
        "tiles": tiles_report,
    }
