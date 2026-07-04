# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "..")
import cv2
import numpy as np
from skimage.morphology import skeletonize
from embroidery_generator import (
    imread_unicode, split_bgr_and_background_mask, extract_color_masks,
    split_into_parts, measure_thickness_px, generate_satin_stitches,
    compute_px_per_mm, SATIN_PITCH_MM, THICKNESS_THRESHOLD_MM, HOOP_SIZE_MM,
)

path = r"G:\マイドライブ\刺繍データ\トートM\posterized (2).png"
raw = imread_unicode(path)
bgr, bgmask = split_bgr_and_background_mask(raw)
px_per_mm = compute_px_per_mm(bgr, HOOP_SIZE_MM)
satin_pitch_px = SATIN_PITCH_MM * px_per_mm
thr_px = THICKNESS_THRESHOLD_MM * px_per_mm

color_masks = extract_color_masks(bgr, bgmask)
for color, cmask in color_masks:
    if color != (112, 25, 25):
        continue
    for part in split_into_parts(cmask):
        area_px = cv2.countNonZero(part)
        area_mm2 = area_px / (px_per_mm ** 2)
        if area_mm2 > 10:
            continue
        thickness_px = measure_thickness_px(part)
        skel = skeletonize(part > 0)
        n_skel_px = int(skel.sum())
        pts = generate_satin_stitches(part, satin_pitch_px)
        print(f"area={area_mm2:.2f}mm2 thickness_px={thickness_px:.2f} skel_px={n_skel_px} out_points={len(pts)}")
