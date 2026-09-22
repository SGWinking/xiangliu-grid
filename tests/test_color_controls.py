import unittest
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from color_engine.color_space import oklab_to_rgb, rgb_to_oklab
from color_engine.corrections import (
    estimate_low_chroma_cast,
    reduce_warm_cast,
    resolve_low_frequency_strengths,
)
from server import low_frequency_correct_placements


def image_from_lab(background, red, green):
    lab = np.empty((24, 24, 3), dtype=np.float32)
    lab[:] = background
    lab[4:10, 3:9] = red
    lab[13:20, 14:21] = green
    return Image.fromarray(oklab_to_rgb(lab))


class LowFrequencyModeTests(unittest.TestCase):
    def test_modes_keep_the_simple_ui_contract(self):
        self.assertEqual(resolve_low_frequency_strengths("off", 0.5), (0.0, 0.0))
        self.assertEqual(resolve_low_frequency_strengths("lightness", 0.5), (0.5, 0.0))
        self.assertEqual(resolve_low_frequency_strengths("auto", 0.5), (0.5, 0.175))

    def test_frontend_exposes_only_the_simple_controls(self):
        html = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="lowFreqMode"', html)
        self.assertIn('id="yellowReduction"', html)
        self.assertIn('low_freq_mode:', html)
        self.assertIn('yellow_reduction:', html)
        self.assertNotIn('id="lowFreqStrength"', html)

    def test_padded_edge_tile_uses_the_actual_placement_extent(self):
        left = Image.new("RGB", (4, 4), (150, 140, 130))
        padded_right = Image.new("RGB", (4, 4), (155, 142, 128))
        placements = [
            ("R01_C01", 0, 0, left, {}),
            ("R01_C02", 4, 0, padded_right, {}),
        ]
        plan = {"width": 6, "height": 4}

        corrected = low_frequency_correct_placements(
            placements, plan, strength=0.5, mode="lightness", small_width=6
        )

        self.assertEqual([item[3].size for item in corrected], [(4, 4), (4, 4)])


class WarmCastReductionTests(unittest.TestCase):
    def test_restores_low_chroma_background_without_desaturating_pigments(self):
        target = image_from_lab(
            background=(0.72, 0.008, 0.018),
            red=(0.58, 0.055, 0.025),
            green=(0.58, -0.045, 0.035),
        )
        too_warm = image_from_lab(
            background=(0.72, 0.012, 0.030),
            red=(0.58, 0.055, 0.025),
            green=(0.58, -0.045, 0.035),
        )

        target_cast = estimate_low_chroma_cast([target])
        current_cast = estimate_low_chroma_cast([too_warm])
        corrected = reduce_warm_cast(too_warm, target_cast, current_cast, amount=1.0)

        before_lab = rgb_to_oklab(np.asarray(too_warm))
        after_lab = rgb_to_oklab(np.asarray(corrected))
        target_lab = rgb_to_oklab(np.asarray(target))

        before_error = np.linalg.norm(before_lab[0, 0, 1:] - target_lab[0, 0, 1:])
        after_error = np.linalg.norm(after_lab[0, 0, 1:] - target_lab[0, 0, 1:])
        self.assertLess(after_error, before_error * 0.35)

        self.assertLess(np.linalg.norm(after_lab[6, 5, 1:] - before_lab[6, 5, 1:]), 0.0025)
        self.assertLess(np.linalg.norm(after_lab[16, 17, 1:] - before_lab[16, 17, 1:]), 0.0025)

    def test_zero_amount_is_a_no_op(self):
        image = image_from_lab(
            background=(0.72, 0.012, 0.030),
            red=(0.58, 0.18, 0.08),
            green=(0.58, -0.12, 0.11),
        )
        cast = estimate_low_chroma_cast([image])
        corrected = reduce_warm_cast(image, cast, cast, amount=0.0)
        np.testing.assert_array_equal(np.asarray(corrected), np.asarray(image))


if __name__ == "__main__":
    unittest.main()
