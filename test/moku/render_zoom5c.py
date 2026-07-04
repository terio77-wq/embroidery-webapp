# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyembroidery

p = pyembroidery.read("test/moku/muku_output5.pes")
STITCH = pyembroidery.STITCH
COLOR_CHANGE = pyembroidery.COLOR_CHANGE
threads = p.threadlist

seg = []
segments = []
cur_idx = 0
for x, y, cmd in p.stitches:
    if cmd == STITCH:
        seg.append((x, y))
    elif cmd == COLOR_CHANGE:
        if len(seg) >= 2:
            segments.append((seg, cur_idx))
        seg = []
        cur_idx += 1
    else:
        if len(seg) >= 2:
            segments.append((seg, cur_idx))
        seg = []
if len(seg) >= 2:
    segments.append((seg, cur_idx))

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#888888")
ax.set_facecolor("#888888")
for s, idx in segments:
    t = threads[idx] if idx < len(threads) else None
    color = (t.get_red()/255.0, t.get_green()/255.0, t.get_blue()/255.0) if t else "black"
    xs = [pt[0]/10.0 for pt in s]
    ys = [-pt[1]/10.0 for pt in s]
    ax.plot(xs, ys, linewidth=0.5, color=color)
ax.set_xlim(10, 45)
ax.set_ylim(-20, 10)
ax.set_aspect("equal")
ax.set_title("tag zoom (grey bg, true colors)")
fig.savefig("test/moku/muku_output5_tag_zoom_greybg.png", dpi=200, facecolor="#888888")
print("saved")
