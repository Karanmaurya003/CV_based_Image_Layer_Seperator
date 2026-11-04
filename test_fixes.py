"""Offline tests for the review fixes (no models, no server). Run: python test_fixes.py"""
import ast, numpy as np, cv2
import pipeline as P

rng = np.random.default_rng(0)

# --- mask cleanup: island removed, hole filled, thin tail trimmed ---
m = np.zeros((400, 400), np.uint8); m[100:300, 100:300] = 1
m[10:14, 10:14] = 1                 # stray island
m[200:204, 200:204] = 0             # pinhole
m[150:152, 300:340] = 1             # 2px tail
c = P.clean_mask(m)
assert c[11, 11] == 0 and c[201, 201] == 1 and c[151, 320] == 0 and c[200, 200] == 1
st = P.mask_stats(c); assert st['pieces'] == 1 and st['holes'] == 0, st
print('clean_mask ok', st)

# --- spray alpha: recovered over road, undeterminable over snow ---
W = np.array(P.SPRAY_COLOR, np.float32) / 255
road = np.full((100, 100, 3), 0.35, np.float32); snow = np.full((100, 100, 3), 0.95, np.float32)
plate = np.concatenate([road, snow], 1); true_a = 0.5
I = (true_a * W + (1 - true_a) * plate)
img = (I * 255).astype(np.uint8); plate8 = (plate * 255).astype(np.uint8)
region = np.ones((100, 200), np.uint8)
A, V = P.spray_alpha(img, plate8, region)
assert abs(A[50, 50] - 0.5) < 0.03 and V[50, 50], A[50, 50]
assert not V[50, 150] and A[50, 150] == 0
print('spray_alpha ok: road a=%.2f, snow valid=%s' % (A[50, 50], V[50, 150]))

# --- shadow: textured road must not create shadow outside the real one ---
road = (150 + rng.normal(0, 6, (300, 400))).clip(0, 255).astype(np.uint8)
img = np.dstack([road] * 3).copy()
cv2.ellipse(img, (210, 215), (80, 16), 0, 0, 360, (95, 95, 95), -1)
cv2.rectangle(img, (160, 120), (260, 210), (200, 40, 40), -1)
obj = np.zeros((300, 400), np.uint8); obj[120:211, 160:261] = 1
sh = P.auto_shadow(img, obj, np.zeros_like(obj))
far = sh.copy(); far[150:260, 100:330] = 0
assert sh[225, 210] > 0.2 and far.sum() < 1.0, (sh[225, 210], far.sum())
print('shadow ok: inside=%.2f, stray=%.2f' % (sh[225, 210], far.sum()))

# --- fill_region: only the target changes, others masked, fallback is flagged ---
P.STATUS.clear()
base = np.full((300, 400, 3), 120, np.uint8); base[:, 200:] = 180
tgt = np.zeros((300, 400), bool); tgt[100:160, 80:140] = True
oth = np.zeros((300, 400), bool); oth[100:160, 260:320] = True
out = P.fill_region(base, tgt, oth, P.lama_fill)
far = np.ones((300, 400), bool); far[80:180, 60:160] = False
assert (out[far] == base[far]).all(), 'pixels outside the hole must be untouched'
assert abs(int(out[130, 110, 0]) - 120) < 10, out[130, 110]
print('fill_region ok; notes:', [n[:60] for n in P.status_notes()] or 'LaMa used')
if not P.lama_available():
    assert P.status_notes() and 'WARNING' in P.status_notes()[0]

# --- real ensure_bg from main.py, run with stubs ---
src = open('main.py').read(); tree = ast.parse(src)
want = {'union_masks', 'ensure_bg', 'make_fill'}
code = '\n'.join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want)
ns = dict(np=np, cv2=cv2, P=P); exec(code, ns)
H, Wd = 300, 400
img = np.full((H, Wd, 3), 150, np.uint8)
cv2.rectangle(img, (100, 150), (200, 220), (200, 40, 40), -1)               # object
spray = np.zeros((H, Wd), np.uint8); spray[130:200, 190:270] = 1
sprayed = img.astype(np.float32) / 255
sprayed[spray > 0] = 0.5 * np.array(P.SPRAY_COLOR) / 255 + 0.5 * sprayed[spray > 0]
img = (sprayed * 255).astype(np.uint8)
obj = np.zeros((H, Wd), np.uint8); obj[150:221, 100:201] = 1
z = np.zeros((H, Wd), np.float32)
layers = [dict(id=1, kind='object', mask=obj, alpha=obj.astype(np.float32), fg=img / 255.0, shadow=z.copy()),
          dict(id=2, kind='spray', mask=spray, alpha=z.copy(), fg=np.zeros((H, Wd, 3), np.float32), shadow=z.copy())]
s = dict(img=img, layers=layers, bg=None, engine='lama', prompt='', note='')
bg = ns['ensure_bg'](s)
rec = P.recompose(bg, layers)
err = np.abs(rec - img.astype(np.float32) / 255)
print('ensure_bg ok: recomposition mean error %.2f/255, note=%r' % (err.mean() * 255, s['note'][:50]))
assert err.mean() * 255 < 3
print('ALL OK')
