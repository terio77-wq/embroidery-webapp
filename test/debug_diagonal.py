# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "..")
import cv2
import numpy as np
import pyembroidery
from embroidery_generator import (
    imread_unicode, split_bgr_and_background_mask, extract_color_masks,
    split_into_parts, measure_thickness_px, generate_satin_stitches, generate_tatami_stitches,
    split_into_runs, compute_px_per_mm, process_image, build_pattern,
    SATIN_PITCH_MM, TATAMI_PITCH_MM, TATAMI_OVERLAP_MM, MAX_STITCH_LENGTH_MM, HOOP_SIZE_MM,
)

path = r"G:\マイドライブ\刺繍データ\トートM\posterized (2).png"
raw = imread_unicode(path)
bgr, bgmask = split_bgr_and_background_mask(raw)
px_per_mm = compute_px_per_mm(bgr, HOOP_SIZE_MM)

blocks = process_image(bgr, bgmask, px_per_mm)
pattern = build_pattern(blocks, px_per_mm, bgr.shape)

STITCH = pyembroidery.STITCH
print("=== in-memory pattern (before write) ===")
prev = None
max_len = 0
worst = None
for x, y, cmd in pattern.stitches:
    if cmd == STITCH:
        if prev is not None:
            d = ((x - prev[0])**2 + (y - prev[1])**2)**0.5 / 10.0
            if d > max_len:
                max_len = d
                worst = (prev, (x, y, cmd))
        prev = (x, y)
    else:
        prev = (x, y)
print("max STITCH-STITCH length (mm):", max_len)
print("worst pair:", worst)

# write both formats and check on read-back
pyembroidery.write(pattern, "debug_output.pes")
pyembroidery.write(pattern, "debug_output.dst")

for fmt in ("debug_output.pes", "debug_output.dst"):
    p2 = pyembroidery.read(fmt)
    prev = None
    max_len = 0
    worst = None
    for x, y, cmd in p2.stitches:
        if cmd == STITCH:
            if prev is not None:
                d = ((x - prev[0])**2 + (y - prev[1])**2)**0.5 / 10.0
                if d > max_len:
                    max_len = d
                    worst = (prev, (x, y, cmd))
            prev = (x, y)
        else:
            prev = (x, y)
    print(f"=== read back {fmt} ===")
    print("max STITCH-STITCH length (mm):", max_len)
    print("worst pair:", worst)
