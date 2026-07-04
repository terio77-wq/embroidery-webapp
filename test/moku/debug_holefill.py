# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
import embroidery_generator as eg

path = r"C:\Users\terio\Downloads\posterized (2).png"
raw = eg.imread_unicode(path)
bgr, bgmask = eg.split_bgr_and_background_mask(raw)
px_per_mm = eg.compute_px_per_mm(bgr, eg.HOOP_SIZE_MM)
hole_fill_max_area_px = eg.HOLE_FILL_MAX_AREA_MM2 * (px_per_mm ** 2)
print("hole_fill_max_area_px =", hole_fill_max_area_px)

color_masks = eg.extract_color_masks(bgr, bgmask)
# orange = (35, 136, 252) based on earlier block listing
for color, mask in color_masks:
    if color != (35, 136, 252):
        continue
    print("orange raw area:", cv2.countNonZero(mask))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print("n contours:", len(contours))
    for i, h in enumerate(hierarchy[0]):
        area = cv2.contourArea(contours[i])
        print(f"  contour {i}: parent={h[3]} area={area:.1f}")
    filled = eg.fill_enclosed_holes(mask, bgmask, hole_fill_max_area_px)
    print("filled area:", cv2.countNonZero(filled), "delta:", cv2.countNonZero(filled) - cv2.countNonZero(mask))
