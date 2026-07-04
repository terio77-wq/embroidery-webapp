# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
import math
from embroidery_generator import (
    imread_unicode, split_bgr_and_background_mask, compute_px_per_mm,
    process_image, split_into_runs, MAX_STITCH_LENGTH_MM, HOOP_SIZE_MM,
)

path = r"C:\Users\terio\Downloads\posterized (2).png"
raw = imread_unicode(path)
bgr, bgmask = split_bgr_and_background_mask(raw)
px_per_mm = compute_px_per_mm(bgr, HOOP_SIZE_MM)
max_gap_px = MAX_STITCH_LENGTH_MM * px_per_mm

blocks = process_image(bgr, bgmask, px_per_mm)
print(f"px_per_mm={px_per_mm:.3f} total blocks={len(blocks)}")

all_intra_gaps_mm = []
worst = []
for b in blocks:
    runs = split_into_runs(b.points_px, max_gap_px)
    if len(runs) <= 1:
        continue
    for r0, r1 in zip(runs, runs[1:]):
        (x0, y0), (x1, y1) = r0[-1], r1[0]
        d_mm = math.hypot(x1 - x0, y1 - y0) / px_per_mm
        all_intra_gaps_mm.append(d_mm)
    worst.append((len(runs), b.stitch_type, b.area_px / (px_per_mm**2)))

all_intra_gaps_mm.sort()
n = len(all_intra_gaps_mm)
print(f"intra-block run-gap count = {n}")
if n:
    print(f"min={all_intra_gaps_mm[0]:.2f}mm max={all_intra_gaps_mm[-1]:.2f}mm")
    print(f"median={all_intra_gaps_mm[n//2]:.2f}mm p90={all_intra_gaps_mm[int(n*0.9)]:.2f}mm p99={all_intra_gaps_mm[int(n*0.99)]:.2f}mm")
    over10 = sum(1 for d in all_intra_gaps_mm if d > 10)
    over20 = sum(1 for d in all_intra_gaps_mm if d > 20)
    print(f"gaps >10mm: {over10}, >20mm: {over20}")

worst.sort(reverse=True)
print("top 10 blocks by run count:")
for runs_n, st, area_mm2 in worst[:10]:
    print(f"  runs={runs_n} type={st} area={area_mm2:.1f}mm2")
