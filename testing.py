"""Try out the background-cleaning pipeline step by step.

Shows every intermediate stage in its own window so each step can be judged
visually. Press any key to close all windows.
"""

import cv2
import numpy as np

image = cv2.imread("./images/test.png")

# step 1: flatten the grass texture (shapes conserve their original form)
flat = cv2.medianBlur(image, 21)

# step 2: distance from the grass color ("how not-grass is each pixel?")
grass_color = np.median(flat.reshape(-1, 3), axis=0)
dist = np.linalg.norm(flat.astype(np.float32) - grass_color, axis=2)
dist_view = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# step 3: threshold using grass statistics (median + 8*MAD)
med = np.median(dist)
mad = np.median(np.abs(dist - med))
mask = (dist > med + 8 * mad).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

# step 4: contours + area filter
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
shapes = [c for c in contours if cv2.contourArea(c) > 8000]

print(f"threshold = {med + 8 * mad:.1f} (grass median {med:.1f}, MAD {mad:.1f})")
print(f"shapes found: {len(shapes)} (expected 5)")
for i, c in enumerate(shapes):
    x, y, w, h = cv2.boundingRect(c)
    print(f"  shape {i + 1}: area {cv2.contourArea(c):.0f} px^2 at ({x}, {y})")

result = image.copy()
cv2.drawContours(result, shapes, -1, (0, 255, 0), 3)

cv2.imshow("1 - original", image)
cv2.imshow("2 - flattened (median blur)", flat)
cv2.imshow("3 - distance from grass color", dist_view)
cv2.imshow("4 - mask", mask)
cv2.imshow("5 - result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
