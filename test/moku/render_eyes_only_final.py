# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyembroidery

p = pyembroidery.read("test/moku/muku_final.pes")
STITCH = pyembroidery.STITCH
JUMP = pyembroidery.JUMP
TRIM = pyembroidery.TRIM
COLOR_CHANGE = pyembroidery.COLOR_CHANGE

thread_idx = 0
threads = p.threadlist
current_color = "#%02x%02x%02x" % (threads[0].get_red(), threads[0].get_green(), threads[0].get_blue())
seg_x, seg_y = [], []
fig, ax = plt.subplots(figsize=(8, 8))
def flush(color):
    if len(seg_x) >= 2:
        ax.plot(seg_x, seg_y, color=color, linewidth=1.2, solid_capstyle="round", marker='.', markersize=2)
for x, y, cmd in p.stitches:
    if cmd == STITCH:
        seg_x.append(x / 10.0); seg_y.append(-y / 10.0)
    elif cmd in (JUMP, TRIM):
        flush(current_color); seg_x, seg_y = [], []
    elif cmd == COLOR_CHANGE:
        flush(current_color); seg_x, seg_y = [], []
        thread_idx += 1
        if thread_idx < len(threads):
            t = threads[thread_idx]
            current_color = "#%02x%02x%02x" % (t.get_red(), t.get_green(), t.get_blue())
flush(current_color)
ax.set_aspect("equal")
ax.set_xlim(15, 33)
ax.set_ylim(-8, 0)
ax.grid(True, linewidth=0.3)
ax.set_title("eyes zoom (mm)")
fig.savefig("test/moku/muku_eyes_zoom_final.png", dpi=250)
print("saved")
