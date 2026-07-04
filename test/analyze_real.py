# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "..")
import cv2
import numpy as np
from embroidery_generator import (
    imread_unicode, split_bgr_and_background_mask, extract_color_masks,
    split_into_parts, measure_thickness_px, generate_satin_stitches,
    generate_tatami_stitches, compute_px_per_mm,
    SATIN_PITCH_MM, TATAMI_PITCH_MM, TATAMI_OVERLAP_MM, TATAMI_MAX_STITCH_MM,
    THICKNESS_THRESHOLD_MM, HOOP_SIZE_MM,
)

path = r"G:\マイドライブ\刺繍データ\トートM\posterized (2).png"
raw = imread_unicode(path)
bgr, bgmask = split_bgr_and_background_mask(raw)
px_per_mm = compute_px_per_mm(bgr, HOOP_SIZE_MM)
print(f"px_per_mm={px_per_mm:.3f}")

satin_pitch_px = SATIN_PITCH_MM * px_per_mm
tatami_pitch_px = TATAMI_PITCH_MM * px_per_mm
overlap_px = TATAMI_OVERLAP_MM * px_per_mm
max_stitch_px = TATAMI_MAX_STITCH_MM * px_per_mm
thr_px = THICKNESS_THRESHOLD_MM * px_per_mm

color_masks = extract_color_masks(bgr, bgmask)
total_stitches = 0
max_jump_mm_overall = 0
rows = []
for color, cmask in color_masks:
    for part in split_into_parts(cmask):
        thickness_px = measure_thickness_px(part)
        thickness_mm = thickness_px / px_per_mm
        area_px = cv2.countNonZero(part)
        area_mm2 = area_px / (px_per_mm ** 2)
        if thickness_px < thr_px:
            pts = generate_satin_stitches(part, satin_pitch_px)
            stype = "satin"
        else:
            pts = generate_tatami_stitches(part, tatami_pitch_px, overlap_px, max_stitch_px)
            stype = "tatami"
        if not pts:
            continue
        total_stitches += len(pts)
        arr = np.array(pts, dtype=float)
        if len(arr) > 1:
            d = np.linalg.norm(np.diff(arr, axis=0), axis=1) / px_per_mm
            max_jump = d.max()
            avg_jump = d.mean()
        else:
            max_jump = avg_jump = 0
        max_jump_mm_overall = max(max_jump_mm_overall, max_jump)
        rows.append((color, stype, thickness_mm, area_mm2, len(pts), avg_jump, max_jump))

rows.sort(key=lambda r: -r[3])
print(f"{'color':>16} {'type':6} {'thick(mm)':>9} {'area(mm2)':>9} {'pts':>5} {'avgstep(mm)':>11} {'maxstep(mm)':>11}")
for color, stype, thick, area, npts, avgj, maxj in rows:
    print(f"{str(color):>16} {stype:6} {thick:9.2f} {area:9.2f} {npts:5d} {avgj:11.3f} {maxj:11.3f}")

print()
print("total stitches:", total_stitches)
print("total parts:", len(rows))
n_colors = len(set(r[0] for r in rows))
print("distinct colors:", n_colors)
print("overall max jump(mm):", max_jump_mm_overall)

rows_sorted = sorted(rows, key=lambda r: -r[3])
prev_color = None
color_changes = 0
trims = 0
for r in rows_sorted:
    if prev_color is not None:
        trims += 1
        if r[0] != prev_color:
            color_changes += 1
    prev_color = r[0]
print("trims (part-to-part jumps):", trims)
print("color changes:", color_changes)
