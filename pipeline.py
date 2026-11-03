import io, re, zipfile
import cv2, numpy as np
from PIL import Image
from pymatting import estimate_alpha_cf, estimate_foreground_ml

def png(arr):
    ok, buf = cv2.imencode('.png', np.ascontiguousarray(arr))
    return buf.tobytes()

def rgba_png(rgb01, alpha01):
    bgr = (rgb01[..., ::-1] * 255).astype(np.uint8)
    a = (alpha01 * 255).astype(np.uint8)
    return png(np.dstack([bgr, a]))

def refine_alpha(img_rgb, mask, band=15):
    # soft alpha + decontaminated foreground colors, computed on a crop for speed
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    pad = band * 3
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band, band))
    tri = np.full(m.shape, 0.5)
    tri[cv2.erode(m, k) == 1] = 1.0
    tri[cv2.dilate(m, k) == 0] = 0.0
    crop = img_rgb[y0:y1, x0:x1].astype(np.float64) / 255
    alpha = estimate_alpha_cf(crop, tri)
    fg = estimate_foreground_ml(crop, alpha)
    A = np.zeros((h, w), np.float32)
    F = np.zeros((h, w, 3), np.float32)
    A[y0:y1, x0:x1] = np.clip(alpha, 0, 1)
    F[y0:y1, x0:x1] = np.clip(fg, 0, 1)
    return A, F

def _L(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)

def shadow_from_region(img_rgb, region, obj_mask, others):
    # alpha = how much darker the region is than the nearby unshadowed surface
    L = _L(img_rgb)
    ring = cv2.dilate(region.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    ring &= (region == 0) & (obj_mask == 0) & (others == 0)
    if ring.sum() < 50:
        return np.zeros(L.shape, np.float32)
    ref = np.percentile(L[ring], 60)
    alpha = np.clip(1 - L / max(ref, 1), 0, 1) * (region > 0)
    return cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.2)

def auto_shadow(img_rgb, obj_mask, others, drop=0.2, reach=0.6):
    # first guess: dark pixels near the object that touch it
    L = _L(img_rgb)
    ys, xs = np.where(obj_mask > 0)
    size = max(ys.max() - ys.min(), xs.max() - xs.min())
    r = int(max(15, size * reach))
    m = obj_mask.astype(np.uint8)
    near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    near = (near > 0) & (m == 0) & (others == 0)
    if near.sum() < 50:
        return np.zeros(L.shape, np.float32)
    ref = np.percentile(L[near], 70)
    cand = ((L < ref * (1 - drop)) & near).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    touch = cv2.dilate(m, np.ones((15, 15), np.uint8))
    n, cc = cv2.connectedComponents(cand)
    keep = np.zeros_like(cand)
    for i in range(1, n):
        comp = cc == i
        if touch[comp].any():
            keep[comp] = 1
    return shadow_from_region(img_rgb, keep, obj_mask, others)

_lama = None
def inpaint_background(img_rgb, hole):
    global _lama
    hole = cv2.dilate((hole > 0).astype(np.uint8) * 255, np.ones((11, 11), np.uint8))
    try:
        if _lama is None:
            from simple_lama_inpainting import SimpleLama
            _lama = SimpleLama()
        out = np.array(_lama(Image.fromarray(img_rgb), Image.fromarray(hole)))
        return out[:img_rgb.shape[0], :img_rgb.shape[1]]
    except Exception as e:
        print('LaMa unavailable, falling back to OpenCV Telea:', e)
        bgr = np.ascontiguousarray(img_rgb[..., ::-1])
        return cv2.inpaint(bgr, hole, 5, cv2.INPAINT_TELEA)[..., ::-1]

def recompose(bg_rgb, layers):
    # layers bottom -> top; shadow multiplies, object composites normally
    out = bg_rgb.astype(np.float32) / 255
    for l in layers:
        out = out * (1 - l['shadow'][..., None])
        a = l['alpha'][..., None]
        out = l['fg'] * a + out * (1 - a)
    return np.clip(out, 0, 1)

def build_ora(w, h, layers):
    # layers bottom -> top: (name, png_bytes, composite_op); opens in Krita, GIMP, MyPaint
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('mimetype', 'image/openraster')
        rows = []
        for i, (name, data, op) in enumerate(layers):
            safe = re.sub(r'[^A-Za-z0-9_-]', '_', name)
            src = f'data/{i:02d}_{safe}.png'
            z.writestr(src, data)
            rows.append(f'''<layer name='{safe}' src='{src}' composite-op='{op}' opacity='1.0' x='0' y='0' visibility='visible'/>''')
        body = ''.join(reversed(rows))
        xml = f'''<?xml version='1.0' encoding='UTF-8'?><image version='0.0.3' w='{w}' h='{h}'><stack>{body}</stack></image>'''
        z.writestr('stack.xml', xml)
    return buf.getvalue()
