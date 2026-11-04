import io, re, zipfile
import cv2, numpy as np
from PIL import Image

def png(arr):
    ok, buf = cv2.imencode('.png', np.ascontiguousarray(arr))
    return buf.tobytes()

def rgba_png(rgb01, alpha01):
    bgr = (rgb01[..., ::-1] * 255).astype(np.uint8)
    a = (alpha01 * 255).astype(np.uint8)
    return png(np.dstack([bgr, a]))

def refine_alpha(img_rgb, mask, band=15):
    from pymatting import estimate_alpha_cf, estimate_foreground_ml  # lazy: keeps other code testable
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

def _smooth_L(L, valid, sigma=4.0):
    # masked Gaussian low-pass: object pixels do not leak into the shadow estimate
    v = valid.astype(np.float32)
    num = cv2.GaussianBlur(L * v, (0, 0), sigma)
    den = cv2.GaussianBlur(v, (0, 0), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), L)

def shadow_from_region(img_rgb, region, obj_mask, others):
    # alpha = 1 - ratio, ratio = (low-passed brightness) / (nearby unshadowed brightness), clamped.
    # Low-pass removes tyre tracks and asphalt grain; tiny alphas are dropped.
    L = _L(img_rgb)
    ring = cv2.dilate(region.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    ring &= (region == 0) & (obj_mask == 0) & (others == 0)
    if ring.sum() < 50:
        return np.zeros(L.shape, np.float32)
    ref = np.percentile(L[ring], 60)
    valid = (obj_mask == 0) & (others == 0)
    Ls = _smooth_L(L, valid, 4.0)
    ratio = np.clip(Ls / max(ref, 1), 0.2, 1.0)
    reg = cv2.GaussianBlur(((region > 0) & valid).astype(np.float32), (0, 0), 3)
    alpha = (1 - ratio) * np.clip(reg, 0, 1)
    alpha[alpha < 0.03] = 0
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

STATUS = {}          # filled by lama_fill: {'telea_used': True, 'lama_error': '...'}
_lama = None
_lama_import_failed = False
_lama_error = ''

def lama_available():
    import importlib.util
    return importlib.util.find_spec('simple_lama_inpainting') is not None

def lama_fill(img_rgb, hole):
    """Raw fill: LaMa if possible, else OpenCV Telea. The fallback is recorded in STATUS
    so the UI can show a visible warning (Telea makes flat polygon blobs)."""
    global _lama, _lama_import_failed, _lama_error
    h, w = img_rgb.shape[:2]
    mask = (hole > 0).astype(np.uint8) * 255
    if not _lama_import_failed:
        try:
            if _lama is None:
                from simple_lama_inpainting import SimpleLama
                _lama = SimpleLama()
            out = np.array(_lama(Image.fromarray(img_rgb), Image.fromarray(mask)))
            return np.ascontiguousarray(out[:h, :w])
        except Exception as e:
            _lama_import_failed = isinstance(e, ImportError)
            _lama_error = f'{type(e).__name__}: {str(e)[:160]}'
            print('LaMa unavailable, falling back to OpenCV Telea:', e)
    STATUS['telea_used'] = True
    STATUS['lama_error'] = _lama_error or 'unknown reason'
    bgr = np.ascontiguousarray(img_rgb[..., ::-1])
    return np.ascontiguousarray(cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)[..., ::-1])

def status_notes():
    if STATUS.get('telea_used'):
        return ['WARNING: LaMa was NOT used (' + STATUS.get('lama_error', 'unknown reason') +
                '). The OpenCV fallback gives flat blobs. Fix: python -m pip install simple-lama-inpainting, then restart.']
    return []

def fill_region(img_rgb, target, others, fn, grow=7, context=0.75, feather=2.0):
    """Fill ONE region on a full-resolution crop (hole + context), paste back only that region.
    `others` = regions still to be filled; they are masked too so the filler cannot copy them
    into this hole, but they are not pasted. grow = small dilation of the hole (5 to 9 px)."""
    t = (target > 0).astype(np.uint8)
    o = (others > 0).astype(np.uint8)
    if grow > 1:
        k = np.ones((grow, grow), np.uint8)
        t, o = cv2.dilate(t, k), cv2.dilate(o, k)
    ys, xs = np.where(t > 0)
    if len(ys) == 0:
        return img_rgb
    h, w = t.shape
    m = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * context) + 32
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, h)
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, w)
    crop = np.ascontiguousarray(img_rgb[y0:y1, x0:x1])
    tc = t[y0:y1, x0:x1].astype(np.float32)
    hole = ((t | o)[y0:y1, x0:x1] > 0).astype(np.uint8)
    out = fn(crop, hole).astype(np.float32)
    soft = np.maximum(cv2.GaussianBlur(tc, (0, 0), feather), tc)[..., None]
    res = img_rgb.copy()
    res[y0:y1, x0:x1] = np.clip(out * soft + crop.astype(np.float32) * (1 - soft), 0, 255).astype(np.uint8)
    return res

def inpaint_background(img_rgb, hole, grow=7):
    return fill_region(img_rgb, hole > 0, np.zeros(hole.shape, bool), lama_fill, grow=grow)

def clean_mask(mask, min_island=0.0005, min_hole=0.0005, open_px=3):
    """Remove islands, fill small holes, trim thin tails. Always keeps the largest piece."""
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    thr = max(int(H * W * min_island), 20)
    if open_px > 1:
        opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
        if opened.sum() >= 0.9 * m.sum():       # never delete thin real parts (wires, antennas)
            m = opened
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = areas >= thr
        keep[1 + int(np.argmax(areas))] = True
        m = keep[lab].astype(np.uint8)
    hthr = max(int(H * W * min_hole), 20)
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - m, connectivity=4)
    fillit = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w_, h_, a = st[i]
        border = x == 0 or y == 0 or x + w_ == W or y + h_ == H
        fillit[i] = (not border) and a < hthr
    m[fillit[lab]] = 1
    return m

def mask_stats(mask):
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    pieces = cv2.connectedComponents(m, connectivity=8)[0] - 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - m, connectivity=4)
    holes = sum(1 for i in range(1, n)
                if not (st[i][0] == 0 or st[i][1] == 0 or st[i][0] + st[i][2] == W or st[i][1] + st[i][3] == H))
    return dict(area=round(float(100.0 * m.sum() / (H * W)), 2), pieces=int(pieces), holes=int(holes))

SPRAY_COLOR = (240, 240, 245)

def spray_alpha(img_rgb, plate_rgb, region, color=SPRAY_COLOR, min_sep=0.08):
    """Soft layer for spray/dust: I = a*W + (1-a)*B  =>  a = (I-B)/(W-B), per channel, median.
    Where W-B is tiny (white spray on white snow) alpha cannot be found: returns valid=False there."""
    import warnings
    H, Wd = region.shape
    A = np.zeros((H, Wd), np.float32)
    V = np.zeros((H, Wd), bool)
    ys, xs = np.where(region > 0)
    if len(ys) == 0:
        return A, V
    I = img_rgb[ys, xs].astype(np.float32) / 255
    B = plate_rgb[ys, xs].astype(np.float32) / 255
    d = np.array(color, np.float32) / 255 - B
    ok = np.abs(d) > min_sep
    ratio = np.where(ok, (I - B) / np.where(ok, d, 1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.nanmedian(ratio, axis=1)
    valid = (ok.sum(1) >= 2) & ~np.isnan(a)
    a = np.where(valid, np.clip(a, 0, 1), 0)
    a[a < 0.02] = 0
    A[ys, xs] = a
    V[ys, xs] = valid
    return cv2.GaussianBlur(A, (0, 0), 1.0), V

def shift(arr, dx, dy):
    if not dx and not dy:
        return arr
    h, w = arr.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def recompose(bg_rgb, layers, use_layout=False):
    # layers bottom -> top; shadow multiplies, object composites normally.
    # use_layout=True applies each layer's visibility and (dx, dy) move (editor state).
    out = bg_rgb.astype(np.float32) / 255
    for l in layers:
        sh, a, fg = l['shadow'], l['alpha'], l['fg']
        if use_layout:
            if not l.get('vis', True):
                continue
            dx, dy = l.get('dx', 0), l.get('dy', 0)
            sh, a, fg = shift(sh, dx, dy), shift(a, dx, dy), shift(fg, dx, dy)
        out = out * (1 - sh[..., None])
        a = a[..., None]
        out = fg * a + out * (1 - a)
    return np.clip(out, 0, 1)

# ---------------------------------------------------------------------------
# Amodal completion: rebuild the hidden part of an object that another layer covers.
# ---------------------------------------------------------------------------
def _hull(mask):
    pts = cv2.findNonZero((mask > 0).astype(np.uint8))
    out = np.zeros(mask.shape, np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out

def amodal_region(car_mask, occluder_mask):
    # convex hull of the visible object bridges the gap the occluder leaves;
    # only occluder pixels inside the hull are treated as hidden object
    car = (car_mask > 0).astype(np.uint8)
    occ = (occluder_mask > 0).astype(np.uint8)
    hole = (occ & _hull(car)).astype(np.uint8)
    return hole, (car | hole)

def complete_layer(img_rgb, layer, occluder_mask, fill_fn):
    """Returns dict(mask, alpha, fg) for the completed layer, or None if nothing is hidden.
    The fill mask also covers occluder pixels near the hull, so the filler never
    copies the occluder's colors (e.g. the person's clothes) into the object."""
    car = (layer['mask'] > 0).astype(np.uint8)
    occ = (occluder_mask > 0).astype(np.uint8)
    hole, amodal = amodal_region(car, occ)
    if hole.sum() == 0:
        return None
    hole_d = cv2.dilate(hole, np.ones((9, 9), np.uint8))          # also eats the occluder's edge halo
    fill_mask = cv2.dilate(occ, np.ones((9, 9), np.uint8)) & cv2.dilate(_hull(car), np.ones((31, 31), np.uint8))
    fill_mask = (fill_mask | hole_d).astype(np.uint8)
    filled = fill_fn(img_rgb, fill_mask)
    region = (hole_d > 0) & (amodal > 0)
    A = layer['alpha'].copy(); A[region] = 1.0
    F = layer['fg'].copy(); F[region] = filled[region].astype(np.float32) / 255
    return dict(mask=amodal, alpha=A, fg=F)

def build_ora(w, h, layers):
    # layers bottom -> top: (name, png_bytes, composite_op[, x, y, visible]); opens in Krita, GIMP, MyPaint
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('mimetype', 'image/openraster')
        rows = []
        for i, item in enumerate(layers):
            name, data, op = item[:3]
            x, y, vis = (tuple(item[3:]) + (0, 0, True))[:3]
            safe = re.sub(r'[^A-Za-z0-9_-]', '_', name)
            src = f'data/{i:02d}_{safe}.png'
            z.writestr(src, data)
            state = 'visible' if vis else 'hidden'
            rows.append(f"<layer name='{safe}' src='{src}' composite-op='{op}' opacity='1.0' x='{int(x)}' y='{int(y)}' visibility='{state}'/>")
        body = ''.join(reversed(rows))
        xml = f"<?xml version='1.0' encoding='UTF-8'?><image version='0.0.3' w='{w}' h='{h}'><stack>{body}</stack></image>"
        z.writestr('stack.xml', xml)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Generative fill: Stable Diffusion inpainting + LCM-LoRA (about 5 steps instead of 25).
#   pip install diffusers transformers accelerate peft
# First use downloads ~4 GB (+~135 MB LoRA) into your user cache, outside the repo.
# ---------------------------------------------------------------------------
SD_MODEL = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
LCM_LORA = 'latent-consistency/lcm-lora-sdv1-5'
_sd = None

def _load_sd():
    global _sd
    if _sd is None:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_MODEL, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32,
            safety_checker=None, requires_safety_checker=False).to(dev)
        fast = False
        try:
            from diffusers import LCMScheduler
            pipe.load_lora_weights(LCM_LORA)
            pipe.fuse_lora()
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            fast = True
        except Exception as e:
            print('LCM-LoRA unavailable, using the standard sampler (slower):', e)
        _sd = (pipe, fast)
    return _sd

def generative_fill(img_rgb, hole, prompt='', steps=None, seed=0):
    """Fill `hole` with AI content on a square crop around it, then paste back only
    the hole with a feathered edge, so all other pixels stay exactly as in the original."""
    import torch
    pipe, fast = _load_sd()
    steps = steps or (5 if fast else 25)
    guidance = 1.5 if fast else 7.5
    hole = cv2.dilate((hole > 0).astype(np.uint8) * 255, np.ones((11, 11), np.uint8))
    ys, xs = np.where(hole > 0)
    if len(ys) == 0:
        return img_rgb.copy()
    h, w = hole.shape
    side = int(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * 1.6)
    side = min(max(side, 256), min(h, w))
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    crop = img_rgb[y0:y0 + side, x0:x0 + side]
    m = hole[y0:y0 + side, x0:x0 + side]
    S = 512
    interp = cv2.INTER_AREA if side > S else cv2.INTER_CUBIC
    out = pipe(
        prompt=prompt.strip() or 'empty background, seamless natural continuation of the surrounding scene',
        negative_prompt='person, people, text, watermark, logo, blurry, border',
        image=Image.fromarray(cv2.resize(crop, (S, S), interpolation=interp)),
        mask_image=Image.fromarray(cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)),
        height=S, width=S, num_inference_steps=steps, guidance_scale=guidance, strength=1.0,
        generator=torch.Generator('cpu').manual_seed(seed),
    ).images[0]
    out = cv2.resize(np.array(out), (side, side), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mf = (m > 0).astype(np.float32)
    soft = np.maximum(cv2.GaussianBlur(mf, (0, 0), 3), mf)[..., None]
    res = img_rgb.copy()
    reg = res[y0:y0 + side, x0:x0 + side].astype(np.float32)
    res[y0:y0 + side, x0:x0 + side] = np.clip(out * soft + reg * (1 - soft), 0, 255).astype(np.uint8)
    return res
