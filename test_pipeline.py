"""Offline test of pipeline.py (no SAM/server needed): synthetic scene -> layers -> recompose error."""
import numpy as np, cv2
import pipeline as P

h, w = 300, 400
img = np.full((h, w, 3), 170, np.uint8)
cv2.ellipse(img, (210, 215), (80, 18), 0, 0, 360, (110, 110, 110), -1)   # shadow
cv2.rectangle(img, (150, 110), (250, 210), (200, 40, 40), -1)            # object
mask = np.zeros((h, w), np.uint8); mask[110:211, 150:251] = 1
others = np.zeros_like(mask)

alpha, fg = P.refine_alpha(img, mask)
shadow = P.auto_shadow(img, mask, others)
hole = (mask > 0) | (shadow > 0.02)
bg = P.inpaint_background(img, hole)
layer = dict(alpha=alpha, fg=fg, shadow=shadow)
rec = P.recompose(bg, [layer])
d = np.abs(rec - img.astype(np.float32) / 255)
print('mean err /255:', round(float(d.mean()) * 255, 2), ' p99:', round(float(np.percentile(d, 99)) * 255, 1))
ora = P.build_ora(w, h, [('background', P.png(img[..., ::-1]), 'svg:src-over')])
print('ora bytes:', len(ora))
