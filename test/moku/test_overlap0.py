# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
import cv2
import embroidery_generator as eg

path = r"C:\Users\terio\Downloads\posterized (2).png"
raw = eg.imread_unicode(path)
bgr, bgmask = eg.split_bgr_and_background_mask(raw)
px_per_mm = eg.compute_px_per_mm(bgr, eg.HOOP_SIZE_MM)

# TATAMI_OVERLAP_MMを0にして生成しなおす
eg.TATAMI_OVERLAP_MM = 0.0
blocks = eg.process_image(bgr, bgmask, px_per_mm)
pattern = eg.build_pattern(blocks, px_per_mm, bgr.shape)
import pyembroidery
pyembroidery.write(pattern, "test/moku/muku_overlap0.pes")

for b in blocks:
    if b.color_bgr == (93,50,27) and b.area_px < 300:
        xs=[p[0] for p in b.points_px]; ys=[p[1] for p in b.points_px]
        w=(max(xs)-min(xs))/px_per_mm; h=(max(ys)-min(ys))/px_per_mm
        print(f'area={b.area_px:.0f}px type={b.stitch_type} stitch_bbox={w:.2f}x{h:.2f}mm')
