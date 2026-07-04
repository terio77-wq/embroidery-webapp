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
thread_idx = 0

seg = []
segments = []  # (points, thread_idx)
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
for s, idx in segments:
    if idx >= len(threads):
        color = "black"
    else:
        t = threads[idx]
        color = (t.get_red()/255.0, t.get_green()/255.0, t.get_blue()/255.0)
    xs = [pt[0]/10.0 for pt in s]
    ys = [-pt[1]/10.0 for pt in s]
    ax.plot(xs, ys, linewidth=0.4, color=color)
ax.set_xlim(10, 45)
ax.set_ylim(-20, 10)
ax.set_aspect("equal")
ax.set_title("tag zoom (true thread colors)")
fig.savefig("test/moku/muku_output5_tag_zoom_truecolor.png", dpi=200)
print("saved, n threads:", len(threads))
