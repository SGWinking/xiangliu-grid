from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image

from .color_space import oklab_to_rgb, rgb_to_oklab


LOW_FREQUENCY_MODES = {"auto", "lightness", "off"}


def resolve_low_frequency_strengths(mode: str, strength: float) -> tuple[float, float]:
    """Return (lightness, color-temperature) strengths for the simple UI modes."""
    mode = mode if mode in LOW_FREQUENCY_MODES else "auto"
    strength = float(np.clip(strength, 0.0, 1.0))
    if mode == "off":
        return 0.0, 0.0
    if mode == "lightness":
        return strength, 0.0
    return strength, strength * 0.35


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (value >= edge1).astype(np.float32)
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def estimate_low_chroma_cast(images: Iterable[Image.Image], max_edge: int = 320) -> np.ndarray:
    """Estimate the warm low-chroma substrate cast without letting pigments dominate."""
    samples = []
    fallbacks = []
    for image in images:
        preview = image.convert("RGB").copy()
        preview.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        lab = rgb_to_oklab(np.asarray(preview, dtype=np.uint8)).reshape(-1, 3)
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        base = (lab[:, 0] >= 0.20) & (lab[:, 0] <= 0.95) & (chroma <= 0.065)
        warm = base & (lab[:, 2] > 0.0)
        if np.any(warm):
            samples.append(lab[warm, 1:])
        if np.any(base):
            fallbacks.append(lab[base, 1:])
    values = samples if samples else fallbacks
    if not values:
        return np.zeros(2, dtype=np.float32)
    return np.median(np.concatenate(values, axis=0), axis=0).astype(np.float32)


def reduce_warm_cast(
    image: Image.Image,
    target_cast: np.ndarray,
    current_cast: np.ndarray,
    amount: float,
) -> Image.Image:
    """Pull only low-chroma warm substrate toward its pre-calibration cast."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return image.copy()
    target = np.asarray(target_cast, dtype=np.float32).reshape(2)
    current = np.asarray(current_cast, dtype=np.float32).reshape(2)
    delta = np.clip(current - target, -0.04, 0.04)
    if delta[1] <= 0.0:
        return image.copy()

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = rgb_to_oklab(rgb)
    light = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.mod(np.degrees(np.arctan2(b, a)), 360.0)

    # Full correction on faded plaster; smoothly protect stronger red/green/blue pigments.
    chroma_weight = 1.0 - _smoothstep(0.028, 0.055, chroma)
    hue_distance = np.abs(np.mod(hue - 75.0 + 180.0, 360.0) - 180.0)
    warm_weight = 1.0 - _smoothstep(35.0, 70.0, hue_distance)
    positive_b = _smoothstep(0.0, 0.010, b)
    light_weight = _smoothstep(0.15, 0.30, light) * (1.0 - _smoothstep(0.92, 0.99, light))
    mask = (chroma_weight * warm_weight * positive_b * light_weight * amount).astype(np.float32)

    result = lab.copy()
    result[..., 1] = a - delta[0] * mask
    result[..., 2] = b - delta[1] * mask
    return Image.fromarray(oklab_to_rgb(result))
