import io, os, json, time, uuid, base64, zipfile
import cv2, numpy as np, torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from segment_anything import sam_model_registry, SamPredictor
import pipeline as P

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SIDE = int(os.environ.get('LAYER_MAX_SIDE', '2048'))   # raise on a strong GPU
sam = sam_model_registry['vit_b'](checkpoint='models/sam_vit_b_01ec64.pth').to(DEVICE)
predictor = SamPredictor(sam)
predictor_zoom = SamPredictor(sam)   # separate embedding for zoomed crops of small objects
app = FastAPI()
SESS, active = {}, {'sid': None}

def get(sid, embed=False):
    s = SESS.get(sid)
    if s is None:
        raise HTTPException(404, 'unknown session')
    if embed and active['sid'] != sid:
        predictor.set_image(s['img'])
        active['sid'] = sid
    return s

def union_masks(s, skip=None):
    m = np.zeros(s['img'].shape[:2], np.uint8)
    for l in s['layers']:
        if l['id'] != skip:
            m |= l['mask']
    return m

def ensure_bg(s):
    if s['bg'] is None:
        img, notes = s['img'], []
        P.STATUS.clear()
        # one region per layer (mask + shadow; spray = mask only), nearest to the camera (lowest in frame) first
        regions = []
        for l in s['layers']:
            m = l['mask'] > 0
            if l.get('kind', 'object') != 'spray':
                m = m | (l['shadow'] > 0.03)
            rows = np.where(m.any(axis=1))[0]
            if len(rows):
                regions.append((int(rows.max()), m))
        regions.sort(key=lambda t: -t[0])
        base = make_fill(s.get('engine'), s.get('prompt', ''), notes)
        cur = img.copy()
        for i, (_, m) in enumerate(regions):
            others = np.zeros_like(m)
            for _, o in regions[i + 1:]:
                others |= o
            cur = P.fill_region(cur, m, others, base)
        # spray / dust: alpha from the difference between photo and clean plate
        for l in s['layers']:
            if l.get('kind') == 'spray':
                region = l['mask'] > 0
                A, V = P.spray_alpha(img, cur, region)
                l['alpha'] = A
                fg = np.empty((*A.shape, 3), np.float32)
                fg[:] = np.array(P.SPRAY_COLOR, np.float32) / 255
                l['fg'] = fg
                keep = region & ~V          # undeterminable (white on white): keep the photo in the plate
                cur[keep] = img[keep]
        s['bg'] = cur
        s['note'] = ' '.join(notes + P.status_notes())
    return s['bg']

@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, 'not an image')
    k = MAX_SIDE / max(bgr.shape[:2])
    if k < 1:
        bgr = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sid = uuid.uuid4().hex[:8]
    SESS[sid] = dict(img=img, layers=[], current=None, bg=None, next_id=1, diff=None,
                    engine='lama', prompt='', note='')
    t = time.time()
    get(sid, embed=True)
    return dict(sid=sid, w=img.shape[1], h=img.shape[0], embed_s=round(time.time() - t, 2), lama=P.lama_available())

@app.get('/api/original/{sid}')
def original(sid: str):
    return Response(P.png(get(sid)['img'][..., ::-1]), media_type='image/png')

class Pt(BaseModel):
    x: float
    y: float
    l: int

class SegReq(BaseModel):
    sid: str
    points: list[Pt] = []
    box: list[float] | None = None
    zoom: bool = False

def _zoom_segment(img, pts, lbl, box):
    # crop around the box, upscale to ~1024 px, run SAM on the crop (own embedding), paste back
    H, W = img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(box[2] - box[0], box[3] - box[1]) * 1.5 + 32
    x0 = int(np.clip(cx - side / 2, 0, W - 1)); x1 = int(min(W, x0 + side))
    y0 = int(np.clip(cy - side / 2, 0, H - 1)); y1 = int(min(H, y0 + side))
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    sc = 1024 / max(ch, cw)
    if sc < 1.2:
        return None                      # already large enough, use the normal path
    predictor_zoom.set_image(cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC))
    zp = zl = None
    if pts is not None:
        inside = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        if inside.any():
            zp, zl = (pts[inside] - [x0, y0]) * sc, lbl[inside]
    zbox = (box - [x0, y0, x0, y0]) * sc
    masks, _, _ = predictor_zoom.predict(point_coords=zp, point_labels=zl, box=zbox, multimask_output=False)
    m = cv2.resize(masks[0].astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((H, W), np.uint8)
    out[y0:y1, x0:x1] = m
    return out

@app.post('/api/segment')
def segment(r: SegReq):
    s = get(r.sid, embed=True)
    pts = np.array([[p.x, p.y] for p in r.points]) if r.points else None
    lbl = np.array([p.l for p in r.points]) if r.points else None
    box = np.array(r.box, dtype=float) if r.box else None
    best = None
    if r.zoom and box is not None:
        best = _zoom_segment(s['img'], pts, lbl, box)
    if best is None:
        multi = len(r.points) == 1 and box is None
        masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl, box=box, multimask_output=multi)
        best = masks[int(np.argmax(scores))].astype(np.uint8)
    s['current'] = best
    mask_png = P.png(best * 255)
    return dict(mask='data:image/png;base64,' + base64.b64encode(mask_png).decode())

class CommitReq(BaseModel):
    sid: str
    name: str = 'object'
    kind: str = 'object'     # 'object' or 'spray'
    clean: bool = True

def info(sid, l):
    lid, t = l['id'], int(time.time() * 1000)
    return dict(id=lid, name=l['name'], kind=l.get('kind', 'object'), q=l.get('q'),
                object=f'/api/layer/{sid}/{lid}/object?t={t}',
                shadow=f'/api/layer/{sid}/{lid}/shadow?t={t}')

@app.post('/api/commit')
def commit(r: CommitReq):
    s = get(r.sid)
    if s['current'] is None or s['current'].sum() == 0:
        raise HTTPException(400, 'no mask selected')
    spray = r.kind == 'spray'
    mask = s['current']
    if r.clean:
        mask = P.clean_mask(mask, open_px=1 if spray else 3)
    h, w = mask.shape
    if spray:
        # soft cloud: give the matte room around the clicked shape; alpha is computed when the plate is built
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        alpha = np.zeros((h, w), np.float32)
        fg = np.empty((h, w, 3), np.float32)
        fg[:] = np.array(P.SPRAY_COLOR, np.float32) / 255
        shadow = np.zeros((h, w), np.float32)
    else:
        alpha, fg = P.refine_alpha(s['img'], mask)
        shadow = P.auto_shadow(s['img'], mask, union_masks(s))
    lid = s['next_id']
    s['next_id'] += 1
    layer = dict(id=lid, name=r.name or f'object_{lid}', kind='spray' if spray else 'object',
                 mask=mask, alpha=alpha, fg=fg, shadow=shadow, q=P.mask_stats(mask),
                 vis=True, dx=0, dy=0)
    s['layers'].append(layer)
    s['current'], s['bg'] = None, None
    return info(r.sid, layer)

class LidReq(BaseModel):
    sid: str
    lid: int

@app.post('/api/set_shadow')
def set_shadow(r: LidReq):
    # use the currently selected SAM mask as this layer's shadow (manual override)
    s = get(r.sid)
    layer = next(l for l in s['layers'] if l['id'] == r.lid)
    if s['current'] is None:
        raise HTTPException(400, 'click the shadow first')
    layer['shadow'] = P.shadow_from_region(s['img'], s['current'], layer['mask'], union_masks(s, skip=r.lid))
    s['current'], s['bg'] = None, None
    return info(r.sid, layer)

@app.get('/api/layer/{sid}/{lid}/{kind}')
def layer_png(sid: str, lid: int, kind: str):
    l = next(x for x in get(sid)['layers'] if x['id'] == lid)
    if kind == 'object':
        return Response(P.rgba_png(l['fg'], l['alpha']), media_type='image/png')
    black = np.zeros((*l['shadow'].shape, 3), np.float32)
    return Response(P.rgba_png(black, l['shadow']), media_type='image/png')

@app.delete('/api/layer/{sid}/{lid}')
def delete_layer(sid: str, lid: int):
    s = get(sid)
    s['layers'] = [l for l in s['layers'] if l['id'] != lid]
    s['bg'] = None
    return dict(ok=True)

def make_fill(engine, prompt, notes):
    def fn(img, hole):
        if engine == 'sd':
            try:
                return P.generative_fill(img, hole, prompt)
            except Exception as e:
                print('Stable Diffusion fill failed, using LaMa:', e)
                notes.append('Stable Diffusion failed (' + str(e)[:100] + '); used LaMa.')
        return P.lama_fill(img, hole)
    return fn

class CompleteReq(BaseModel):
    sid: str
    lid: int
    engine: str = 'lama'
    prompt: str = ''

@app.post('/api/complete')
def complete(r: CompleteReq):
    # rebuild the hidden part of layer `lid` behind the layers stacked above it
    s = get(r.sid)
    idx = next((i for i, l in enumerate(s['layers']) if l['id'] == r.lid), None)
    if idx is None:
        raise HTTPException(404, 'unknown layer')
    layer, above = s['layers'][idx], s['layers'][idx + 1:]
    if not above:
        raise HTTPException(400, 'Nothing is in front of this layer. Commit the covering object (e.g. the person) and move it above this layer.')
    occ = np.zeros_like(layer['mask'])
    for l in above:
        occ |= l['mask']
    notes = []
    P.STATUS.clear()
    base = make_fill(r.engine, r.prompt, notes)
    fill = lambda img, m: P.fill_region(img, m, np.zeros(m.shape, bool), base, grow=1)
    res = P.complete_layer(s['img'], layer, occ, fill)
    if res is None:
        raise HTTPException(400, 'No hidden area found: layers in front do not overlap this object.')
    layer.update(mask=res['mask'], alpha=res['alpha'], fg=res['fg'])
    s['bg'] = None
    out = info(r.sid, layer)
    out['note'] = ' '.join(notes + P.status_notes()) or 'Hidden part completed.'
    return out

class LayoutItem(BaseModel):
    id: int
    dx: float = 0
    dy: float = 0
    vis: bool = True

class LayoutReq(BaseModel):
    sid: str
    layers: list[LayoutItem]   # bottom -> top

@app.post('/api/layout')
def layout(r: LayoutReq):
    s = get(r.sid)
    by = {l['id']: l for l in s['layers']}
    new = []
    for it in r.layers:
        l = by.pop(it.id, None)
        if l:
            l['dx'], l['dy'], l['vis'] = int(round(it.dx)), int(round(it.dy)), it.vis
            new.append(l)
    s['layers'] = new + list(by.values())
    return dict(ok=True)

class BgReq(BaseModel):
    sid: str
    engine: str = 'lama'   # 'lama' (fast) or 'sd' (Stable Diffusion generative fill)
    prompt: str = ''

@app.post('/api/background')
def background(r: BgReq):
    s = get(r.sid)
    s['engine'] = 'sd' if r.engine == 'sd' else 'lama'
    s['prompt'] = r.prompt
    s['bg'] = None
    ensure_bg(s)
    return dict(url=f'/api/background/{r.sid}?t={int(time.time() * 1000)}', note=s['note'])

@app.get('/api/background/{sid}')
def background_png(sid: str):
    return Response(P.png(ensure_bg(get(sid))[..., ::-1]), media_type='image/png')

@app.get('/api/check/{sid}')
def check(sid: str):
    s = get(sid)
    rec = P.recompose(ensure_bg(s), s['layers'])
    d = np.abs(rec - s['img'].astype(np.float32) / 255)
    s['diff'] = P.png((np.clip(d * 8, 0, 1) * 255).astype(np.uint8)[..., ::-1])
    return dict(mean=round(float(d.mean()) * 255, 2), p99=round(float(np.percentile(d, 99)) * 255, 1),
                diff=f'/api/diff/{sid}?t={int(time.time() * 1000)}', note=s['note'])

@app.get('/api/diff/{sid}')
def diff_png(sid: str):
    return Response(get(sid)['diff'], media_type='image/png')

@app.get('/api/export/{sid}')
def export(sid: str):
    s = get(sid)
    bg = ensure_bg(s)
    h, w = bg.shape[:2]
    bg_png = P.rgba_png(bg.astype(np.float32) / 255, np.ones((h, w), np.float32))
    layers = [('background', bg_png, 'svg:src-over')]
    meta = []
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('png/00_background.png', bg_png)
        z.writestr('png/original.png', P.png(s['img'][..., ::-1]))
        comp = P.recompose(bg, s['layers'], use_layout=True)
        z.writestr('png/composite.png', P.png((comp * 255 + 0.5).astype(np.uint8)[..., ::-1]))
        for order, l in enumerate(s['layers'], 1):
            black = np.zeros((h, w, 3), np.float32)
            sh = P.rgba_png(black, l['shadow'])
            ob = P.rgba_png(l['fg'], l['alpha'])
            pos = (l['dx'], l['dy'], l['vis'])
            layers += [(f"shadow_{l['name']}", sh, 'svg:multiply', *pos), (l['name'], ob, 'svg:src-over', *pos)]
            z.writestr(f"png/{l['id']:02d}_{l['name']}_shadow.png", sh)
            z.writestr(f"png/{l['id']:02d}_{l['name']}.png", ob)
            z.writestr(f"png/{l['id']:02d}_{l['name']}_mask.png", P.png(l['mask'] * 255))
            ys, xs = np.where(l['mask'] > 0)
            meta.append(dict(id=l['id'], name=l['name'], order=order, dx=l['dx'], dy=l['dy'], visible=l['vis'],
                             bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None))
        z.writestr('layers.json', json.dumps(dict(width=w, height=h, layers=meta), indent=2))
        z.writestr('layers.ora', P.build_ora(w, h, layers))
    return Response(zbuf.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename=layers.zip'})

app.mount('/', StaticFiles(directory='static', html=True), name='static')
