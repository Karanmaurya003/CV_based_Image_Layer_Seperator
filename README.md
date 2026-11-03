# Layer Studio

    python -m venv .venv
    # Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    # download SAM ViT-B into models/:
    #   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    uvicorn main:app --reload

Open http://127.0.0.1:8000
