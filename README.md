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
