import cv2
import numpy as np

img = np.full((1000, 1000, 3), (255, 255, 255), dtype=np.uint8)  # white background

# big body (tatami candidate, > 5mm thick at 10px/mm -> >50px)
cv2.circle(img, (500, 550), 300, (255, 180, 80), -1)  # light blue-ish body (BGR)

# thin strap (satin candidate, < 5mm thick -> < 50px wide)
cv2.line(img, (250, 300), (750, 300), (30, 30, 160), 20)  # thin dark red strap, ~20px wide

# small isolated dot, same color as strap, separate part
cv2.circle(img, (500, 700), 15, (30, 30, 160), -1)

# thin curved "text-like" stroke (green), to test satin skeleton path with a bend
pts = np.array([[350, 850], [500, 780], [650, 850]], dtype=np.int32)
cv2.polylines(img, [pts], False, (60, 160, 60), 18)

cv2.imwrite("test_input.png", img)
print("wrote test_input.png", img.shape)
