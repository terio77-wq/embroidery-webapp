# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "..")
import cv2
import numpy as np
from embroidery_generator import (
    imread_unicode, split_bgr_and_background_mask, extract_color_masks,
    split_into_parts, measure_thickness_px, generate_tatami_stitches, split_into_runs,
    compute_px_per_mm, TATAMI_PITCH_MM, TATAMI_OVERLAP_MM, MAX_STITCH_LENGTH_MM, HOOP_SIZE_MM,
)

path = r"G:\マイドライブ\刺繍データ\トートM\posterized (2).png"
raw = imread_unicode(path)
bgr, bgmask = split_bgr_and_background_mask(raw)
px_per_mm = compute_px_per_mm(bgr, HOOP_SIZE_MM)

tatami_pitch_px = TATAMI_PITCH_MM * px_per_mm
overlap_px = TATAMI_OVERLAP_MM * px_per_mm
max_stitch_px = MAX_STITCH_LENGTH_MM * px_per_mm
max_gap_px = MAX_STITCH_LENGTH_MM * px_per_mm

color_masks = extract_color_masks(bgr, bgmask)
for color, cmask in color_masks:
    if color != (112, 25, 25):
        continue
    for part in split_into_parts(cmask):
        area_mm2 = cv2.countNonZero(part) / (px_per_mm ** 2)
        if area_mm2 < 100:
            continue  # skip the tiny satin-classified ones, only look at the big tatami outline
        pts = generate_tatami_stitches(part, tatami_pitch_px, overlap_px, max_stitch_px)
        runs = split_into_runs(pts, max_gap_px)
        lens = [len(r) for r in runs]
        print(f"part area={area_mm2:.1f}mm2  total_points={len(pts)}  n_runs={len(runs)}")
        print(f"  runs with <=2 points: {sum(1 for l in lens if l<=2)}  (out of {len(runs)})")
        print(f"  run length distribution: min={min(lens)} max={max(lens)} mean={np.mean(lens):.2f} median={np.median(lens)}")
