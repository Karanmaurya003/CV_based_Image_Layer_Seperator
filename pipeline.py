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


# ---------------------------------------------------------------------------
# Generative fill (Stable Diffusion inpainting). Optional: needs
#   pip install diffusers transformers accelerate
# First use downloads ~4 GB from Hugging Face. Fast on CUDA, slow on CPU.
# ---------------------------------------------------------------------------
SD_MODEL = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
_sd = None

def generative_fill(img_rgb, hole, prompt='', steps=30, guidance=7.5, seed=0):
    """Fill `hole` with AI-generated content. Works on a square crop around the
    hole (resized to 512) and pastes back only the hole area with a soft edge,
    so every pixel outside the hole stays exactly as in the original."""
    global _sd
    import torch
    if _sd is None:
        from diffusers import StableDiffusionInpaintPipeline
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if dev == 'cuda' else torch.float32
        _sd = StableDiffusionInpaintPipeline.from_pretrained(
            SD_MODEL, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False).to(dev)
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
    gen = torch.Generator(device=_sd.device).manual_seed(seed)
    out = _sd(
        prompt=prompt.strip() or 'empty background, seamless natural continuation of the surrounding scene',
        negative_prompt='person, people, car, object, text, watermark, logo, blurry, border',
        image=Image.fromarray(cv2.resize(crop, (S, S), interpolation=interp)),
        mask_image=Image.fromarray(cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)),
        height=S, width=S, num_inference_steps=steps, guidance_scale=guidance, generator=gen,
    ).images[0]
    out = cv2.resize(np.array(out), (side, side), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mf = (m > 0).astype(np.float32)
    soft = np.maximum(cv2.GaussianBlur(mf, (0, 0), 3), mf)[..., None]
    res = img_rgb.copy()
    reg = res[y0:y0 + side, x0:x0 + side].astype(np.float32)
    res[y0:y0 + side, x0:x0 + side] = np.clip(out * soft + reg * (1 - soft), 0, 255).astype(np.uint8)
    return res
