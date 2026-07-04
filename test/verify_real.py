# -*- coding: utf-8 -*-
import pyembroidery
import numpy as np

p = pyembroidery.read("real_output.pes")
stitches = p.stitches  # list of [x, y, command]

xs = [s[0] for s in stitches]
ys = [s[1] for s in stitches]
print("total stitch records:", len(stitches))
print("x range mm:", min(xs)/10, max(xs)/10)
print("y range mm:", min(ys)/10, max(ys)/10)

# command counts
from collections import Counter
cmds = Counter(s[2] for s in stitches)
name_map = {v: k for k, v in vars(pyembroidery).items() if isinstance(v, int) and k.isupper()}
for cmd, cnt in cmds.most_common():
    print(cmd, name_map.get(cmd & 0xFFFF0000 if False else cmd, cmd), cnt)

# compute stitch lengths only between consecutive STITCH-type points (ignore jumps/trims/color changes)
STITCH = pyembroidery.STITCH
JUMP = pyembroidery.JUMP
TRIM = pyembroidery.TRIM
COLOR_CHANGE = pyembroidery.COLOR_CHANGE
END = pyembroidery.END

max_len = 0
long_count = 0
prev = None
for x, y, cmd in stitches:
    if cmd == STITCH:
        if prev is not None:
            d = ((x - prev[0])**2 + (y - prev[1])**2) ** 0.5 / 10.0  # mm
            if d > max_len:
                max_len = d
            if d > 5.0:
                long_count += 1
        prev = (x, y)
    else:
        prev = (x, y)  # reset chain reference but don't count jump/trim as a "stitch length"

print("max STITCH-to-STITCH length (mm):", max_len)
print("stitches longer than 5mm:", long_count)
print("thread/color count:", len(p.threadlist))
