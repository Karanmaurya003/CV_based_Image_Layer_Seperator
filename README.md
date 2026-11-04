# Layer Studio

    python -m venv .venv
    # Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    # download SAM ViT-B into models/:
    #   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    uvicorn main:app --reload

Open http://127.0.0.1:8000

## Zoom
Mouse wheel zooms at the cursor. Pan with middle-mouse drag or Space + drag. Buttons: +, -, Fit, 100%.

## Generative fill (optional)
    pip install diffusers transformers accelerate
Pick "Fill: Stable Diffusion", optionally type a prompt, then Build background.
First run downloads about 4 GB. A GPU is strongly recommended; on CPU expect minutes per fill.
Only the hole is replaced; pixels outside it stay untouched. If SD fails, LaMa is used and a message is shown.

## Editor features
- Layer panel: top of stack first. Up/down arrows reorder, checkbox hides, x deletes.
- Move tool: tick it, select a layer, drag in the "all layers" or "isolate" view.
- Export ZIP: layers.ora (order, offsets and visibility kept), per-layer PNG + shadow + mask,
  composite.png (current edited layout), original.png, background and layers.json metadata.

## Complete a hidden object (amodal completion)
Example: a person stands in front of a car.
1. Commit the car (visible part), then commit the person.
2. Make sure the person is above the car in the layer panel (use the arrows).
3. Select the car layer, choose a fill engine, press "Complete hidden part of selected layer".
The car layer now includes the part behind the person. Limitation: the hidden shape is the convex
hull of the visible part, which suits cars and boxes, not concave shapes.

## Tests (no models or server needed)
    python test_amodal.py
    python test_pipeline.py     # needs pymatting

## Deferred on purpose (see project flow PDF)
SAM 2, Qwen-Image-Layered (needs a large GPU or cloud), Real-ESRGAN, outpainting, undo/redo.

## Review fixes (background, spray, shadows, masks, UI)
- **Visible LaMa warning**: a red box appears if LaMa is missing and the blobby OpenCV fallback is used.
  Fix: `python -m pip install simple-lama-inpainting`, restart the server.
- **Cleaner background**: tight holes (7 px), each layer filled separately on a full-resolution crop,
  lowest object in the frame first, other unfilled objects masked so they are not copied.
- **Spray / dust layer**: set "Layer type: Spray", select the cloud, commit, then Build background.
  Alpha = (photo - plate) / (spray colour - plate). White spray on white snow cannot be separated,
  so the photo is kept in the plate there.
- **Shadows**: low-passed brightness ratio, clamped, tiny values dropped, soft region edge.
  Preview on white / black / checkerboard to check them.
- **Masks**: Shift+drag draws a box (with "Zoom SAM", small objects are cropped and upscaled first).
  "Clean mask on commit" removes islands, fills pinholes and trims thin tails.
  Each layer shows area, pieces and holes.
- **UI**: layer thumbnails, new layers ticked by default, canvas re-fits on window resize
  (zoom/pan reaches every edge of the photo).
- Raise the image size limit on a strong GPU: `set LAYER_MAX_SIDE=3000` (Windows) before starting.

Tests: `python test_fixes.py`, `python test_amodal.py` (no models needed).
