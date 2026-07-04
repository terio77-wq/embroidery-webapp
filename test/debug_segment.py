# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyembroidery

p = pyembroidery.read("debug_output.pes")
STITCH = pyembroidery.STITCH

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

# segment 445 was the one with span 80.8mm, 2541 points
target = segments[445]
xs = [p[0]/10.0 for p in target]
ys = [-p[1]/10.0 for p in target]

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(xs, ys, linewidth=0.4, color="navy")
ax.set_aspect("equal")
ax.set_title(f"segment 445 alone, n={len(target)}")
fig.savefig("segment445.png", dpi=150)
print("min/max x:", min(xs), max(xs))
print("min/max y:", min(ys), max(ys))
print("first 5 pts:", list(zip(xs[:5], ys[:5])))
print("last 5 pts:", list(zip(xs[-5:], ys[-5:])))
