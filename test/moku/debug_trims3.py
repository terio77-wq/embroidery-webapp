# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
import embroidery_generator as eg
import pyembroidery
from collections import Counter

path = r"C:\Users\terio\Downloads\posterized (2).png"
raw = eg.imread_unicode(path)
bgr, bgmask = eg.split_bgr_and_background_mask(raw)
px_per_mm = eg.compute_px_per_mm(bgr, eg.HOOP_SIZE_MM)
blocks = eg.process_image(bgr, bgmask, px_per_mm)
pattern = eg.build_pattern(blocks, px_per_mm, bgr.shape)

c0 = Counter(s[2] for s in pattern.stitches)
print("raw pattern:", c0)

settings = {"max_jump": 2047, "max_stitch": 2047, "full_jump": True, "round": True}
norm = pattern.get_normalized_pattern(settings)
c1 = Counter(s[2] for s in norm.stitches)
print("normalized:", c1)
