# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
import embroidery_generator as eg
import pyembroidery

path = r"C:\Users\terio\Downloads\posterized (2).png"
raw = eg.imread_unicode(path)
bgr, bgmask = eg.split_bgr_and_background_mask(raw)
px_per_mm = eg.compute_px_per_mm(bgr, eg.HOOP_SIZE_MM)
blocks = eg.process_image(bgr, bgmask, px_per_mm)

orig_trim = pyembroidery.EmbPattern.trim
orig_move = pyembroidery.EmbPattern.move_abs
counts = {"trim": 0, "move": 0}
def patched_trim(self, *a, **kw):
    counts["trim"] += 1
    return orig_trim(self, *a, **kw)
def patched_move(self, *a, **kw):
    counts["move"] += 1
    return orig_move(self, *a, **kw)
pyembroidery.EmbPattern.trim = patched_trim
pyembroidery.EmbPattern.move_abs = patched_move

pattern = eg.build_pattern(blocks, px_per_mm, bgr.shape)
print(counts)
