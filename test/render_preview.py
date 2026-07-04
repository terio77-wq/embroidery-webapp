# -*- coding: utf-8 -*-
"""生成したPESの針落ち経路を実際に描画してプレビューする(検証用)。"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyembroidery

pes_path = sys.argv[1] if len(sys.argv) > 1 else "real_output.pes"
out_png = sys.argv[2] if len(sys.argv) > 2 else "preview.png"

p = pyembroidery.read(pes_path)
STITCH = pyembroidery.STITCH
JUMP = pyembroidery.JUMP
TRIM = pyembroidery.TRIM
COLOR_CHANGE = pyembroidery.COLOR_CHANGE

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor("white")

thread_idx = 0
threads = p.threadlist
current_color = "#%02x%02x%02x" % (threads[0].get_red(), threads[0].get_green(), threads[0].get_blue()) if threads else "black"

prev = None
seg_x, seg_y = [], []

def flush(color):
    if len(seg_x) >= 2:
        ax.plot(seg_x, seg_y, color=color, linewidth=0.6, solid_capstyle="round")

for x, y, cmd in p.stitches:
    if cmd == STITCH:
        seg_x.append(x / 10.0)
        seg_y.append(-y / 10.0)  # flip y for natural image orientation
    elif cmd in (JUMP, TRIM):
        flush(current_color)
        seg_x, seg_y = [], []
    elif cmd == COLOR_CHANGE:
        flush(current_color)
        seg_x, seg_y = [], []
        thread_idx += 1
        if thread_idx < len(threads):
            t = threads[thread_idx]
            current_color = "#%02x%02x%02x" % (t.get_red(), t.get_green(), t.get_blue())

flush(current_color)

# hoop outline (100mm x 100mm centered at 0,0)
half = 50
ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="gray", linewidth=1, linestyle="--")

ax.set_aspect("equal")
ax.set_xlim(-55, 55)
ax.set_ylim(-55, 55)
ax.set_title(pes_path)
fig.tight_layout()
fig.savefig(out_png, dpi=150)
print("saved", out_png)
