from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path


def write_seam_report(reports: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "seam_diagnostics.json"
    json_path.write_text(json.dumps({"version": 1, "seams": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter(item.get("level", "red") for item in reports)
    colors = {"green": "#3a8f64", "yellow": "#ba8b21", "orange": "#c96327", "red": "#b43b3b"}
    rows = []
    for item in reports:
        shift = item.get("estimated_shift", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['first'])} → {html.escape(item['second'])}</td>"
            f"<td>{html.escape(item.get('orientation', ''))}</td>"
            f"<td style='color:{colors.get(item.get('level'), colors['red'])}'>{html.escape(item.get('level', 'red'))}</td>"
            f"<td>{html.escape(item.get('issue_type', ''))}</td>"
            f"<td>{item.get('color_delta', '-')}</td><td>{item.get('lightness_delta', '-')}</td>"
            f"<td>{shift.get('dx', '-')} / {shift.get('dy', '-')}</td><td>{shift.get('confidence', '-')}</td>"
            "</tr>"
        )
    html_path = output_dir / "seam_diagnostics.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>接缝诊断</title>"
        "<style>body{font:15px system-ui;margin:32px;background:#f3eee5;color:#252525}table{border-collapse:collapse;width:100%;background:white}"
        "th,td{padding:10px;border:1px solid #d7d0c5;text-align:left}th{background:#e7ded0}</style>"
        f"<h1>接缝诊断</h1><p>绿 {counts['green']}　黄 {counts['yellow']}　橙 {counts['orange']}　红 {counts['red']}</p>"
        "<table><thead><tr><th>接缝</th><th>方向</th><th>级别</th><th>问题</th><th>色差</th><th>亮度差</th><th>位移 x/y</th><th>置信度</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>", encoding="utf-8")
    return {"json_path": str(json_path), "html_path": str(html_path), "counts": dict(counts)}
