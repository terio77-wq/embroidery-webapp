# -*- coding: utf-8 -*-
import pyembroidery

p = pyembroidery.read("debug_output.pes")
STITCH = pyembroidery.STITCH
JUMP = pyembroidery.JUMP
TRIM = pyembroidery.TRIM
COLOR_CHANGE = pyembroidery.COLOR_CHANGE

seg = []
segments = []
for x, y, cmd in p.stitches:
    if cmd == STITCH:
        seg.append((x, y))
    else:
        if len(seg) >= 2:
            segments.append(list(seg))
        seg = []
if len(seg) >= 2:
    segments.append(seg)

print("num plotted polylines:", len(segments))
for i, s in enumerate(segments):
    import math
    maxd = 0
    for (x0, y0), (x1, y1) in zip(s, s[1:]):
        d = math.hypot(x1 - x0, y1 - y0) / 10.0
        maxd = max(maxd, d)
    span = math.hypot(s[-1][0] - s[0][0], s[-1][1] - s[0][1]) / 10.0
    if span > 10 or maxd > 5:
        print(f"segment {i}: n_points={len(s)} max_step_mm={maxd:.2f} start={s[0]} end={s[-1]} span_mm={span:.2f}")
