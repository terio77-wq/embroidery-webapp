import pyembroidery
p = pyembroidery.read("test_output.pes")
xs = [s[0] for s in p.stitches]
ys = [s[1] for s in p.stitches]
print("stitch count:", len(p.stitches))
print("x range (units, 0.1mm):", min(xs), max(xs), "-> mm:", min(xs)/10, max(xs)/10)
print("y range (units, 0.1mm):", min(ys), max(ys), "-> mm:", min(ys)/10, max(ys)/10)
print("thread count:", len(p.threadlist))
for t in p.threadlist:
    print("  thread color:", t.hex_color())
