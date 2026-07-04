# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyembroidery

p = pyembroidery.read("test/moku/muku_output5.pes")
STITCH = pyembroidery.STITCH

seg = []
segments = []
for x, y, cmd in p.stitches:
    if cmd == STITCH:
        seg.append((x, y))
    else:
        if len(seg) >= 2:
            segments.append(seg)
        seg = []
if len(seg) >= 2:
    segments.append(seg)

fig, ax = plt.subplots(figsize=(8, 8))
for s in segments:
    xs = [pt[0]/10.0 for pt in s]
    ys = [-pt[1]/10.0 for pt in s]
    ax.plot(xs, ys, linewidth=0.3)
ax.set_xlim(10, 45)
ax.set_ylim(-20, 10)
ax.set_aspect("equal")
ax.set_title("tag zoom (hole-fill applied)")
fig.savefig("test/moku/muku_output5_tag_zoom.png", dpi=200)
print("saved")
