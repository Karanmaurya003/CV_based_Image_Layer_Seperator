import io, time, uuid, base64, zipfile
import cv2, numpy as np, torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from segment_anything import sam_model_registry, SamPredictor
import pipeline as P

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = sam_model_registry['vit_b'](checkpoint='models/sam_vit_b_01ec64.pth').to(DEVICE)
predictor = SamPredictor(sam)
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
        hole = union_masks(s).astype(bool)
        for l in s['layers']:
            hole |= l['shadow'] > 0.02
        s['note'] = ''
        if s.get('engine') == 'sd':
            try:
                s['bg'] = P.generative_fill(s['img'], hole, s.get('prompt', ''))
            except Exception as e:
                print('Stable Diffusion fill failed, using LaMa:', e)
                s['note'] = 'Stable Diffusion failed (' + str(e)[:120] + '). Used LaMa instead.'
                s['bg'] = P.inpaint_background(s['img'], hole)
        else:
            s['bg'] = P.inpaint_background(s['img'], hole)
    return s['bg']

@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, 'not an image')
    k = 2048 / max(bgr.shape[:2])
    if k < 1:
        bgr = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sid = uuid.uuid4().hex[:8]
    SESS[sid] = dict(img=img, layers=[], current=None, bg=None, next_id=1, diff=None,
                    engine='lama', prompt='', note='')
    t = time.time()
    get(sid, embed=True)
    return dict(sid=sid, w=img.shape[1], h=img.shape[0], embed_s=round(time.time() - t, 2))

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

@app.post('/api/segment')
def segment(r: SegReq):
    s = get(r.sid, embed=True)
    pts = np.array([[p.x, p.y] for p in r.points]) if r.points else None
    lbl = np.array([p.l for p in r.points]) if r.points else None
    box = np.array(r.box) if r.box else None
    multi = len(r.points) == 1 and box is None
    masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl, box=box, multimask_output=multi)
    best = masks[int(np.argmax(scores))].astype(np.uint8)
    s['current'] = best
    mask_png = P.png(best * 255)
    return dict(mask='data:image/png;base64,' + base64.b64encode(mask_png).decode())

class CommitReq(BaseModel):
    sid: str
    name: str = 'object'

def info(sid, l):
    lid, t = l['id'], int(time.time() * 1000)
    return dict(id=lid, name=l['name'],
                object=f'/api/layer/{sid}/{lid}/object?t={t}',
                shadow=f'/api/layer/{sid}/{lid}/shadow?t={t}')

@app.post('/api/commit')
def commit(r: CommitReq):
    s = get(r.sid)
    if s['current'] is None or s['current'].sum() == 0:
        raise HTTPException(400, 'no mask selected')
    mask = s['current']
    alpha, fg = P.refine_alpha(s['img'], mask)
    shadow = P.auto_shadow(s['img'], mask, union_masks(s))
    lid = s['next_id']
    s['next_id'] += 1
    layer = dict(id=lid, name=r.name or f'object_{lid}', mask=mask, alpha=alpha, fg=fg, shadow=shadow)
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
                diff=f'/api/diff/{sid}?t={int(time.time() * 1000)}')

@app.get('/api/diff/{sid}')
def diff_png(sid: str):
    return Response(get(sid)['diff'], media_type='image/png')

@app.get('/api/export/{sid}')
def export(sid: str):
    s = get(sid)
    bg = ensure_bg(s)
    h, w = bg.shape[:2]
    layers = [('background', P.rgba_png(bg.astype(np.float32) / 255, np.ones((h, w), np.float32)), 'svg:src-over')]
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('png/00_background.png', layers[0][1])
        z.writestr('png/original.png', P.png(s['img'][..., ::-1]))
        for l in s['layers']:
            black = np.zeros((h, w, 3), np.float32)
            sh = P.rgba_png(black, l['shadow'])
            ob = P.rgba_png(l['fg'], l['alpha'])
            layers += [(f"shadow_{l['name']}", sh, 'svg:multiply'), (l['name'], ob, 'svg:src-over')]
            z.writestr(f"png/{l['id']:02d}_{l['name']}_shadow.png", sh)
            z.writestr(f"png/{l['id']:02d}_{l['name']}.png", ob)
        z.writestr('layers.ora', P.build_ora(w, h, layers))
    return Response(zbuf.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename=layers.zip'})

app.mount('/', StaticFiles(directory='static', html=True), name='static')
