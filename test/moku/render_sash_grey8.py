# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyembroidery

p = pyembroidery.read("test/moku/muku_output8.pes")
STITCH = pyembroidery.STITCH
JUMP = pyembroidery.JUMP
TRIM = pyembroidery.TRIM
COLOR_CHANGE = pyembroidery.COLOR_CHANGE

thread_idx = 0
threads = p.threadlist
current_color = (threads[0].get_red()/255, threads[0].get_green()/255, threads[0].get_blue()/255)
seg_x, seg_y = [], []
fig, ax = plt.subplots(figsize=(9, 9))
fig.patch.set_facecolor("#777777")
ax.set_facecolor("#777777")
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
            current_color = (t.get_red()/255, t.get_green()/255, t.get_blue()/255)
flush(current_color)
ax.set_aspect("equal")
ax.set_xlim(-15, 22)
ax.set_ylim(-25, 16)
ax.set_title("sash zoom (grey bg)")
fig.savefig("test/moku/muku_sash_zoom8_grey.png", dpi=200, facecolor="#777777")
print("saved")
