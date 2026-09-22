import tempfile
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


class StitchHistoryTests(unittest.TestCase):
    def test_history_endpoint_returns_stitch_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / "stitch_history.json"
            history_file.write_text('[{"job_name": "recent stitch"}]', encoding="utf-8")
            with patch.object(server, "STITCH_HISTORY_FILE", history_file):
                httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
                thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                thread.start()
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{httpd.server_port}/api/history", timeout=3
                    ) as response:
                        payload = json.load(response)
                    self.assertEqual(payload["stitch_items"][0]["job_name"], "recent stitch")
                finally:
                    httpd.shutdown()
                    httpd.server_close()

    def test_successful_stitch_settings_are_saved_deduplicated_and_limited(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / "stitch_history.json"
            with patch.object(server, "STITCH_HISTORY_FILE", history_file):
                base = {
                    "manifest_path": r"D:\jobs\wall\tiles_manifest.json",
                    "restored_dir": r"D:\jobs\wall\tiles",
                    "output_dir": r"D:\jobs\wall\stitched",
                    "mode": "full",
                    "trim_padding": True,
                    "palette_reference_paths": r"D:\refs\palette.png",
                    "unknown_internal_value": "must not be persisted",
                }
                for index in range(12):
                    payload = dict(base, out_name=f"result_{index}.png")
                    result = {
                        "out_path": rf"D:\jobs\wall\stitched\result_{index}.png",
                        "canvas_width": 24162,
                        "canvas_height": 7051,
                    }
                    server.remember_stitch_history(payload, result)

                items = server.load_stitch_history()
                self.assertEqual(len(items), 10)
                self.assertEqual(items[0]["settings"]["out_name"], "result_11.png")
                self.assertEqual(items[-1]["settings"]["out_name"], "result_2.png")
                self.assertNotIn("unknown_internal_value", items[0]["settings"])
                self.assertEqual(items[0]["canvas_width"], 24162)
                self.assertEqual(items[0]["canvas_height"], 7051)

                server.remember_stitch_history(dict(base, out_name="result_11.png"), result)
                self.assertEqual(len(server.load_stitch_history()), 10)

    def test_frontend_can_restore_every_saved_stitch_field(self):
        html = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="stitchHistory"', html)
        self.assertIn('data.stitch_items', html)
        self.assertIn('applyStitchHistory', html)
        for element_id in (
            "manifest", "restored", "stitchOutput", "outName", "stitchMode",
            "selectedTiles", "blendMode", "colorMode", "trimPadding", "referencePath",
            "paletteReferences", "paletteProfile", "seamDiagnostics",
        ):
            self.assertIn(f'"{element_id}"', html)


if __name__ == "__main__":
    unittest.main()
