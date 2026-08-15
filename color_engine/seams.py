from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .color_space import rgb_to_oklab


def _gray(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    return arr


def estimate_subpixel_shift(first, second, max_shift: float = 4.0) -> dict:
    a, b = _gray(first), _gray(second)
    if a.shape != b.shape or min(a.shape) < 3:
        return {"dx": 0.0, "dy": 0.0, "confidence": 0.0}
    a = a - a.mean()
    b = b - b.mean()
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2(a * window)
    fb = np.fft.fft2(b * window)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.abs(np.fft.ifft2(cross))
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    def refined(coord: int, length: int, axis: int) -> float:
        before = np.take(corr, (coord - 1) % length, axis=axis).max()
        center = np.take(corr, coord, axis=axis).max()
        after = np.take(corr, (coord + 1) % length, axis=axis).max()
        denom = before - 2 * center + after
        fraction = 0.0 if abs(denom) < 1e-12 else 0.5 * (before - after) / denom
        value = coord + float(np.clip(fraction, -0.5, 0.5))
        return value - length if value > length / 2 else value

    dy = refined(py, corr.shape[0], 0)
    dx = refined(px, corr.shape[1], 1)
    peak = float(corr[py, px])
    sidelobes = corr.copy()
    # PSR 排除主峰及其相邻采样；分数位移的能量会自然分摊到这些点，不能把它当第二个峰。
    for oy in range(-2, 3):
        for ox in range(-2, 3):
            sidelobes[(py + oy) % corr.shape[0], (px + ox) % corr.shape[1]] = np.nan
    side_mean = float(np.nanmean(sidelobes))
    side_std = float(np.nanstd(sidelobes))
    psr = (peak - side_mean) / max(side_std, 1e-9)
    confidence = float(np.clip((psr - 3.0) / 12.0, 0.0, 1.0))
    if math.hypot(dx, dy) > max_shift:
        confidence = 0.0
    return {"dx": round(float(dx), 3), "dy": round(float(dy), 3), "confidence": round(float(confidence), 4)}


def _structure_similarity(first: np.ndarray, second: np.ndarray) -> float:
    a, b = _gray(first), _gray(second)
    gy_a, gx_a = np.gradient(a)
    gy_b, gx_b = np.gradient(b)
    edge_a = np.hypot(gx_a, gy_a).ravel()
    edge_b = np.hypot(gx_b, gy_b).ravel()
    if edge_a.std() < 1e-6 or edge_b.std() < 1e-6:
        return 1.0 if np.mean(np.abs(a - b)) < 2 else 0.0
    return float(np.clip(np.corrcoef(edge_a, edge_b)[0, 1], -1.0, 1.0))


def _similarity_after_shift(first: np.ndarray, second: np.ndarray, dx: float, dy: float) -> float:
    image = Image.fromarray(np.asarray(second, dtype=np.uint8))
    scores = []
    for sign in (-1.0, 1.0):
        aligned = image.transform(
            image.size, Image.Transform.AFFINE, (1, 0, sign * dx, 0, 1, sign * dy),
            resample=Image.Resampling.BILINEAR,
        )
        scores.append(_structure_similarity(first, np.asarray(aligned)))
    return max(scores)


def analyze_overlap(first: Image.Image, second: Image.Image, first_id: str, second_id: str, orientation: str) -> dict:
    if first.size != second.size:
        second = second.resize(first.size, Image.Resampling.BILINEAR)
    a = np.asarray(first.convert("RGB"), dtype=np.uint8)
    b = np.asarray(second.convert("RGB"), dtype=np.uint8)
    lab_a, lab_b = rgb_to_oklab(a), rgb_to_oklab(b)
    delta = np.linalg.norm(lab_a - lab_b, axis=2)
    color_delta = float(np.median(delta))
    lightness_delta = float(abs(np.median(lab_a[..., 0] - lab_b[..., 0])))
    structure_similarity = _structure_similarity(a, b)
    shift = estimate_subpixel_shift(a, b)
    shift_length = math.hypot(shift["dx"], shift["dy"])
    aligned_similarity = _similarity_after_shift(a, b, shift["dx"], shift["dy"])
    alignment_supported = (
        shift_length > 0.35 and shift["confidence"] >= 0.25
        and (
            structure_similarity >= 0.35
            or (aligned_similarity >= 0.35 and aligned_similarity - structure_similarity >= 0.08)
        )
    )
    if structure_similarity < 0.18 and not alignment_supported:
        level, issue = "red", "structure"
    elif alignment_supported:
        level, issue = "orange", "alignment"
    elif color_delta >= 0.035 or lightness_delta >= 0.03:
        level, issue = ("orange" if color_delta >= 0.09 else "yellow"), "color"
    elif structure_similarity < 0.55:
        level, issue = "yellow", "structure"
    else:
        level, issue = "green", "none"
    return {
        "first": first_id,
        "second": second_id,
        "orientation": orientation,
        "overlap_width": first.width if orientation == "vertical" else first.height,
        "color_delta": round(color_delta, 5),
        "lightness_delta": round(lightness_delta, 5),
        "structure_similarity": round(structure_similarity, 4),
        "aligned_structure_similarity": round(aligned_similarity, 4),
        "estimated_shift": shift,
        "level": level,
        "issue_type": issue,
        "safe_for_color_balance": (
            issue in {"none", "color"} and structure_similarity >= 0.45 and shift_length <= 0.35
        ),
    }


def analyze_placement_seams(placements: list[tuple], rows: int, cols: int) -> list[dict]:
    by_pos = {(int(item[4].get("row", 1)), int(item[4].get("col", 1))): item for item in placements}
    reports = []

    def intersection(a, b):
        ax, ay, ai = a[1], a[2], a[3]
        bx, by, bi = b[1], b[2], b[3]
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + ai.width, bx + bi.width), min(ay + ai.height, by + bi.height)
        if x1 <= x0 or y1 <= y0:
            return None
        return (
            ai.crop((x0 - ax, y0 - ay, x1 - ax, y1 - ay)),
            bi.crop((x0 - bx, y0 - by, x1 - bx, y1 - by)),
        )

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            item = by_pos.get((row, col))
            if item is None:
                continue
            for neighbor, orientation in ((by_pos.get((row, col + 1)), "vertical"),
                                          (by_pos.get((row + 1, col)), "horizontal")):
                if neighbor is None:
                    continue
                overlap = intersection(item, neighbor)
                if overlap:
                    reports.append(analyze_overlap(overlap[0], overlap[1], item[0], neighbor[0], orientation))
                else:
                    reports.append({"first": item[0], "second": neighbor[0], "orientation": orientation,
                                    "level": "red", "issue_type": "no_overlap", "safe_for_color_balance": False})
    return reports
