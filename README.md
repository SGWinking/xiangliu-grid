# Xiangliu Grid / 相柳网格

**Version:** `0.4.1`

<img src="assets/logo.png" alt="Xiangliu Grid logo" width="180">

Xiangliu Grid is a local web tool for splitting large master images into strict, numbered square tiles, then stitching selected or complete tiles back into their original grid positions — with seam-balanced color correction so that dozens of independently AI-restored tiles merge into one seamless image.

相柳网格是一个本地网页工具，用于把已经处理好的大图母版严格切分为带编号的方形图块，并在修复后按原始网格位置进行局部或完整拼合。v0.4 起内置接缝平衡与低频颜色场校正，让几十块独立修复的图块拼回后看不出色块边界。

## Origin / 缘起

This small tool was created for the mural recreation and restoration workflow of Dayun Chanyuan. The project uses very high-resolution scanned mural images, but current AI image models work more reliably on controlled `2048 x 2048` image blocks. Xiangliu Grid helps split a large scan into numbered, overlapping, workable tiles so different people can restore sections in parallel and stitch them back into place later.

这个小工具的制作缘起，是因为我们正在推进大云禅院壁画修复与重现工作。项目中有高清扫描版的大图，但受当前 AI 图像模型能力限制，工作图块更适合控制在 `2048 x 2048`。相柳网格用于把一张巨大的扫描图切分成带编号、带重叠区、便于分工处理的工作界面，后续再按原始位置局部或完整拼合回来。

## Why "Xiangliu" / 名称含义

Xiangliu, or 相柳, is a many-headed mythic being from the **Classic of Mountains and Seas**. The name fits this tool because a single large image is divided into many coordinated parts, each with its own identity, while still belonging to one whole body.

"相柳"出自《山海经》，具有多首、多分支的意象。这个工具把一张大图拆成多个编号明确的图块，每一块都可以单独处理，但最终仍能回到同一个整体之中，因此命名为"相柳网格"。

## Features / 功能

- Strict square tiling: every tile is exactly `2048 x 2048` by default.
- Fixed overlap: adjacent tiles overlap by `20px` by default.
- No resampling: the tool only crops the prepared master image. It does not scale, enlarge, or shrink pixels.
- Top-left alignment: the source image is pinned to the top-left corner; blank padding appears only on the right and bottom edges.
- Visual grid preview: the browser preview shows the full working canvas, including right/bottom blank padding.
- Numbered outputs: tiles use stable IDs such as `R01_C01`, `R04_C12`.
- Manifest files: JSON and CSV manifests record tile IDs, coordinates, status fields, and filenames.
- Locator map: `tile_locator_map.jpg` shows each tile number on the full working canvas.
- Patch Align: choose any candidate tile set, use OpenCV to locate a restored patch image, preview the placement, then output a new versioned `patch_runs/run_###/tiles` folder.
- **Seam Balance color correction**: measure adjacent tile edge color differences, solve per-tile RGB offsets globally, apply small biases, then stitch with wide feather blending.
- **Low-frequency color field correction**: estimate a smooth color drift field from a thumbnail, upsample it, and apply it per-tile at full resolution to remove large-area brightness/color shifts without losing detail.
- **Padding trim**: when stitching, optionally trim the right/bottom padding added during splitting, so output matches the original source dimensions.
- Stitching modes:
  - Local preview: stitch only returned tiles into the smallest local region.
  - Full canvas fill: place returned tiles back into the full canvas with blanks elsewhere.
  - Complete stitch: require all tiles and fail if any are missing.

功能概览：

- 严格方形切块：默认每块 `2048 x 2048`。
- 固定重叠区：默认相邻块重叠 `20px`。
- 不重采样：工具只裁切已准备好的母版，不做缩放、放大或缩小。
- 左上贴齐：母版左边和上边严格贴住画布，空白只出现在最右边和最底边。
- 可视化编号预览：预览区域显示完整工作画布，包括右/下白边。
- 稳定编号：输出图块使用 `R01_C01`、`R04_C12` 等编号。
- 清单记录：生成 JSON 和 CSV manifest，记录编号、坐标、状态和文件名。
- 定位总览：生成 `tile_locator_map.jpg`，在完整画布上显示所有编号。
- **接缝平衡调色**：测量相邻图块边缘的颜色差异，全局求解每块 RGB 偏移量，施加小幅校正后宽羽化拼合。
- **低频颜色场校正**：用缩略图估算平滑的颜色漂移场，放大后逐块在全分辨率上应用，消除大面积明暗/色温不均，不损失细节。
- **补色边裁切**：拼合时可选裁掉切分时添加的右/下补色边，输出尺寸还原为母版原图尺寸。
- 拼合模式：
  - 局部试拼：只拼回指定或已回传图块的最小局部区域。
  - 回填整图：把已有图块放回完整画布，其他区域留空。
  - 完整拼合：要求所有图块存在，缺块时报错。

## Color & Stitch Pipeline / 调色与拼合技术路线

### The Problem / 问题

When a large mural scan is split into 48 tiles and each tile is independently AI-restored, every tile comes back with its own low-frequency color drift: some blocks are slightly brighter, some slightly warmer, some slightly desaturated. Simple edge feathering softens the seam but cannot fix the per-block color mismatch — the result looks like a patchwork of slightly different shades.

大图切成 48 块后，每块独立用 AI 修复，回来时每块都有自己的低频色彩漂移：有的偏亮、有的偏暖、有的偏灰。只做边缘羽化能软化接缝，但解决不了整块的色偏——拼出来还是一块深一块浅。

### Solution: Two-Layer Processing / 解决方案：两层处理

#### Layer 1 — Seam Balance (per-tile RGB bias) / 第一层：接缝平衡（块级 RGB 偏移）

Instead of trying to color-match each tile to a reference image (which amplifies differences), this method builds a **constraint network between neighboring tiles** and solves for a small overall offset per tile.

不再试图让每块图各自调到完美，而是在拼合阶段建立"块与块之间的颜色约束网络"，求出每个图块应当做多少小幅整体偏移。

**Algorithm / 算法步骤：**

1. Read all tiles and find row/column adjacency relationships.
2. For each pair of adjacent tiles, sample a 20px strip at the shared edge (with 80px inset to avoid corners/padding). Compute the **median RGB** of each strip.
3. The difference between adjacent strips becomes a constraint: `bias_left - bias_right ≈ right_edge_mean - left_edge_mean`.
4. All constraints form a **linear system**. Solve with **least squares** + **regularization** (bias toward zero, so tiles only shift as much as necessary).
5. **Clamp** each tile's bias to ±max_bias (default ±18 color levels) to prevent over-correction.
6. Apply the bias to each tile: `tile = tile + bias`.

**Why this works / 为什么有效：**

- It's a **global** solution, not a local patch — the entire grid is balanced at once.
- Each tile gets a **single small RGB offset** applied uniformly, so the same pixel value maps to the same output anywhere → edges are naturally continuous.
- Regularization keeps changes conservative: `regularize = 0.35` means "only adjust as much as the edges actually need."
- The max-bias clamp prevents any tile from being pushed to an unnatural color.

#### Layer 2 — Low-Frequency Color Field Correction / 第二层：低频颜色场校正

Seam balance fixes edge mismatches between adjacent tiles. But if the entire upper-left is slightly dark and the lower-right is slightly bright (a slow drift across many tiles), seam balance alone cannot fix that. This is a **low-frequency** problem.

接缝平衡解决了相邻块边缘的跳变。但如果整张图左上偏暗、右下偏亮（跨多块的缓变），接缝平衡解决不了——这是低频问题。

**Algorithm / 算法步骤：**

1. Build a small thumbnail (600px wide) of the stitched image.
2. Convert to LAB color space.
3. For each channel (L, A, B), apply a **large-radius Gaussian blur** (radius ≈ 1/4 of thumbnail width) to extract the low-frequency drift field.
4. Compute the global mean of the blurred field.
5. The correction field = `blurred_field - global_mean`.
6. **Upsample** the correction field to full resolution (it's smooth, so upsampling loses no information).
7. For each **full-resolution tile**, extract the corresponding region of the correction field and subtract it in LAB space.
8. Convert back to RGB.

**Why this preserves detail / 为什么不损失细节：**

- The correction field is **smooth and low-frequency** — it only contains slow brightness/color drifts, not image content.
- Upsampling a smooth field to full resolution is lossless (there's no high-frequency detail to lose).
- The correction is applied to the **full-resolution original tiles**, not to an upscaled image. All mural details, brushstrokes, and textures are preserved at original resolution.
- The `strength` parameter (0–1) controls how aggressively the drift is removed. 0.5 is a good default; 1.0 fully flattens large-area drift.

### Stitching / 拼合

After color correction, tiles are stitched with **weighted feather blending**:

- Each tile generates a feather mask: center weight is high, edges gradually decrease.
- Adjacent tiles' edges blend via weighted average: `result = sum(tile * mask) / sum(mask)`.
- Default feather is 80px for seam-balance mode (wider than the 20px overlap), producing smooth transitions.

This is more robust than simple paste-overwrite, which creates hard seams.

颜色校正后，图块用加权羽化混合拼合：每块生成羽化权重图（中心高、边缘低），相邻块在重叠区加权平均。接缝平衡模式默认羽化 80px，比 20px 的实际重叠区更宽，过渡更平滑。

### Measured Results / 量化效果

On a 48-tile mural (4 rows × 12 columns, 24162 × 7051 px):

| Metric | Before | After Seam Balance | Reduction |
|---|---|---|---|
| Mean seam difference | 6.47 | 2.54 | -61% |
| P90 seam difference | 15.03 | 5.37 | -64% |
| Max seam difference | 36.00 | 11.67 | -68% |

Image sharpness (Laplacian variance) is preserved within 2% of the uncorrected baseline — no detail loss.

| Metric | Before | After Seam Balance | Reduction |
|---|---|---|---|
| 平均接缝色差 | 6.47 | 2.54 | -61% |
| P90 接缝色差 | 15.03 | 5.37 | -64% |
| 最大接缝色差 | 36.00 | 11.67 | -68% |

图像清晰度（拉普拉斯方差）与未校正基准差异 <2%，无细节损失。

### Advantages / 优势

1. **No reference image needed** — the method works from the tiles' own edge relationships. No need to manually pick a "standard" tile or color card.
2. **Deterministic, not AI** — reproducible results, no random variation, no model dependency.
3. **Conservative by design** — regularization + max-bias clamp means it only changes what needs changing. A tile that already matches its neighbors gets near-zero bias.
4. **Scales to huge images** — the low-frequency correction uses a 600px thumbnail for estimation but applies at full resolution, so memory is bounded regardless of output size.
5. **No detail loss** — correction is a smooth low-frequency field applied per-tile at full resolution. Brushstrokes, textures, and fine details are untouched.

1. **不需要参考图** — 算法直接从图块间的边缘关系推导，无需手动指定"标准块"或色卡。
2. **确定性算法，非 AI** — 结果可复现，无随机性，不依赖模型。
3. **设计上保守** — 正则化 + 最大偏移限制，只改必要的部分。已经和邻居匹配的块几乎不动。
4. **支持超大图** — 低频校正用 600px 缩略图估算，全分辨率应用，内存开销可控。
5. **不损失细节** — 校正量是平滑的低频场，在全分辨率图块上应用，笔触、纹理、细节完全保留。

## Workflow / 推荐流程

1. Prepare and scale the large source image externally with your preferred imaging software.
2. Open Xiangliu Grid locally.
3. Select the prepared master image.
4. Preview the grid and padding.
5. Split tiles.
6. Restore or edit tiles in parallel (each person/AI works on their own tiles independently).
7. Use Patch Align for local restored patch images when needed.
8. Stitch with **Seam Balance** (default) + optional **Low-Frequency Correction** + **Trim Padding** for final output.

推荐流程：

1. 先用外部图像软件处理/缩放大图母版。
2. 本地启动相柳网格。
3. 选择已处理好的母版图。
4. 预览编号、网格和右/下白边。
5. 开始切分。
6. 团队按编号并行修复图块（每人/AI 独立处理各自的块）。
7. 需要时用局部对齐工具放置修复补丁。
8. 拼合时选**接缝平衡**（默认）+ 可选**低频颜色场校正**+ **裁掉补色边**，输出最终整图。

## Cropping Rules / 裁切规则

Default values:

```text
Tile size: 2048 x 2048
Overlap:   20px
Stride:    2048 - 20 = 2028px
```

Adjacent horizontal tiles:

```text
Tile 1: x = 0..2047
Tile 2: x = 2028..4075
Overlap: x = 2028..2047, exactly 20px
```

Vertical tiles follow the same rule.

默认规则：

```text
切块尺寸：2048 x 2048
重叠区域：20px
步长：2048 - 20 = 2028px
```

横向相邻块：

```text
第 1 块：x = 0..2047
第 2 块：x = 2028..4075
重叠区：x = 2028..2047，正好 20px
```

纵向同理。

## Run Locally / 本地运行

Double-click:

```text
xiangliu-grid\run_xiangliu_grid.bat
```

Or run in PowerShell:

```powershell
.\xiangliu-grid\run_xiangliu_grid.ps1
```

Then open:

```text
http://127.0.0.1:8765
```

## Outputs / 输出文件

After splitting, the tool creates:

```text
outputs/<job_name>/tiles/
outputs/<job_name>/tiles_manifest.csv
outputs/<job_name>/tiles_manifest.json
outputs/<job_name>/tile_locator_map.jpg
```

Tile filename example:

```text
R01_C01_x0_y0_v001.png
```

## Notes / 注意事项

- Use the file picker for large images. Drag-and-drop copies the file into the tool folder.
- The tool does not scale pixels. Scaling should happen before import.
- Right and bottom padding is expected when the master size is not an exact multiple of the stride.
- Keep `Rxx_Cxx` in restored filenames so the stitcher can place each tile correctly.
- For best seam-balance results, use **Trim Padding** so color statistics exclude the padding area.

- 大图建议使用"选择原图"，拖拽会复制文件到工具目录。
- 工具不缩放像素，缩放请在导入前完成。
- 当母版尺寸不是步长整数倍时，右侧和底部出现补空是正常现象。
- 修复后的文件名请保留 `Rxx_Cxx` 编号，方便准确回拼。
- 接缝平衡建议配合"裁掉补色边"使用，避免补色区干扰颜色统计。

## Changelog / 更新日志

### v0.4.1
- Fixed low-frequency correction to apply per-tile at full resolution (no blur from upscaling).
- Low-frequency correction now runs before stitching, not after.

### v0.4.0
- Added **Seam Balance** color mode: per-tile RGB bias solved from adjacent edge differences.
- Added **Low-Frequency Color Field Correction**: removes large-area brightness/color drift.
- Added **Trim Padding**: output matches original source dimensions.
- Removed per-row/per-column reference image (superseded by seam balance).
- Three strength presets: light (max_bias=10), standard (18), strong (24).

### v0.3.x
- Four-edge feather mask with numpy (true 0→255 gradient, endpoint-accurate).
- Weighted blending instead of sequential paste.
- Per-tile color selection.
- Trim padding for original-dimension output.

## License and Branding / 开源协议与品牌

The source code is released under the [MIT License](LICENSE).

The names **Xiangliu Grid** and **相柳网格**, as well as the project logo and visual identity, are reserved by the project author and are not granted as branding or trademark rights under the MIT License.

源代码使用 [MIT License](LICENSE) 发布。

**Xiangliu Grid / 相柳网格** 的名称、项目 logo 和视觉识别保留为项目作者的品牌资产，不随 MIT 协议授予商标或品牌使用权。
