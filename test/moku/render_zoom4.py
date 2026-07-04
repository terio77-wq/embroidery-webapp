# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "test")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyembroidery

p = pyembroidery.read("test/moku/muku_output4.pes")
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

def render(xlim, ylim, out, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    for s in segments:
        xs = [pt[0]/10.0 for pt in s]
        ys = [-pt[1]/10.0 for pt in s]
        ax.plot(xs, ys, linewidth=0.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(out, dpi=200)
    print("saved", out)

# tag/MUKU area (approx from full view: x~15-45, y~-15 to 5)
render((10, 45), (-20, 10), "test/moku/muku_output4_tag_zoom.png", "tag zoom")
# fur shading area (approx x~-40 to -5, y~10-35)
render((-45, -5), (5, 40), "test/moku/muku_output4_fur_zoom.png", "fur zoom")
