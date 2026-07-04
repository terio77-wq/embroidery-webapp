# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
import embroidery_generator as eg

path = r"C:\Users\terio\Downloads\posterized (2).png"
raw = eg.imread_unicode(path)
bgr, bgmask = eg.split_bgr_and_background_mask(raw)
px_per_mm = eg.compute_px_per_mm(bgr, eg.HOOP_SIZE_MM)
blocks = eg.process_image(bgr, bgmask, px_per_mm)
print("n blocks:", len(blocks))

max_gap_px = eg.MAX_STITCH_LENGTH_MM * px_per_mm
trim_threshold_px = eg.TRIM_THRESHOLD_MM * px_per_mm

prev_color = None
n_block_transitions = 0
n_trim_calls = 0
n_bridge_calls = 0
n_runs_total = 0
for block in blocks:
    is_first_block = prev_color is None
    color_changed = (not is_first_block) and block.color_bgr != prev_color
    runs = eg.split_into_runs(block.points_px, max_gap_px)
    n_runs_total += len(runs)
    if not is_first_block:
        n_block_transitions += 1
        n_trim_calls += 1  # simplistic: assume every block transition trims (color or gap)
    n_bridge_calls += max(0, len(runs) - 1)
    prev_color = block.color_bgr

print("n_block_transitions:", n_block_transitions)
print("n_runs_total:", n_runs_total)
print("expected bridge calls (runs-1 per block summed):", n_bridge_calls)
print("expected max trims (<=):", n_block_transitions)
