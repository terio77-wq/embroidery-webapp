# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyembroidery

p = pyembroidery.read("test/moku/muku_output6.pes")
STITCH = pyembroidery.STITCH
JUMP = pyembroidery.JUMP
TRIM = pyembroidery.TRIM
COLOR_CHANGE = pyembroidery.COLOR_CHANGE

fig, ax = plt.subplots(figsize=(8, 8))
thread_idx = 0
threads = p.threadlist
current_color = "#%02x%02x%02x" % (threads[0].get_red(), threads[0].get_green(), threads[0].get_blue()) if threads else "black"
seg_x, seg_y = [], []
def flush(color):
    if len(seg_x) >= 2:
        ax.plot(seg_x, seg_y, color=color, linewidth=0.8, solid_capstyle="round")
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
ax.set_xlim(-40, 5)
ax.set_ylim(5, 35)
ax.set_title("face zoom")
fig.savefig("test/moku/muku_preview6_face_zoom.png", dpi=200)
print("saved")
