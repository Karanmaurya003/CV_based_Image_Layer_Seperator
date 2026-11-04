"""Offline test (no SAM, no matting needed): amodal completion, layout compositing, ORA export."""
import numpy as np, cv2
import pipeline as P

h, w = 300, 400
img = np.full((h, w, 3), 170, np.uint8)
cv2.rectangle(img, (100, 120), (300, 200), (200, 40, 40), -1)   # car
cv2.rectangle(img, (180, 90), (220, 230), (30, 30, 200), -1)    # person in front
car = np.zeros((h, w), np.uint8); car[120:201, 100:301] = 1; car[90:231, 180:221] = 0
person = np.zeros((h, w), np.uint8); person[90:231, 180:221] = 1
layer = dict(mask=car, alpha=car.astype(np.float32), fg=img.astype(np.float32) / 255,
             shadow=np.zeros((h, w), np.float32))
res = P.complete_layer(img, layer, person, P.inpaint_background)
assert res is not None
assert res['mask'][160, 200] == 1, 'hidden car area should now be in the mask'
assert res['mask'][100, 200] == 0, 'person pixels outside the car outline must stay out'
assert res['alpha'][160, 200] == 1.0
print('amodal mask pixels: visible', int(car.sum()), '-> completed', int(res['mask'].sum()))

done = dict(res, shadow=layer['shadow'], vis=True, dx=0, dy=0)
bg = np.full((h, w, 3), 170, np.uint8)
a = P.recompose(bg, [done])
b = P.recompose(bg, [dict(done, dx=30, dy=10)], use_layout=True)
c = P.recompose(bg, [dict(done, vis=False)], use_layout=True)
assert not np.allclose(a, b) and np.allclose(c, bg / 255.0)
ora = P.build_ora(w, h, [('bg', P.png(bg), 'svg:src-over'), ('car', P.png(bg), 'svg:src-over', 30, 10, False)])
assert b"x='30'" in __import__('zipfile').ZipFile(__import__('io').BytesIO(ora)).read('stack.xml')
print('ALL OK')
