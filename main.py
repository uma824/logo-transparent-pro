import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from PIL import Image, ImageOps
from rembg import new_session, remove

APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = APP_DIR.parent / "frontend"

MAX_BYTES = 15 * 1024 * 1024
ALLOWED = {"image/png", "image/jpeg", "image/webp", "image/svg+xml"}

app = FastAPI(title="Logo Transparent Pro", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Keep one model session alive instead of loading it for every request.
# BiRefNet is a strong general-purpose choice for difficult foreground masks.
SESSION = None


def get_session():
    global SESSION
    if SESSION is None:
        SESSION = new_session(os.getenv("REMBG_MODEL", "birefnet-general"))
    return SESSION


def decode_upload(raw: bytes, content_type: str) -> Image.Image:
    if content_type == "image/svg+xml":
        try:
            import cairosvg
            raw = cairosvg.svg2png(bytestring=raw, output_width=3000)
        except Exception as exc:
            raise HTTPException(400, f"Could not rasterize SVG: {exc}")

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGBA")
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}")


def solid_corner_background_mask(img: Image.Image) -> np.ndarray:
    """
    Logo-aware heuristic for a common case:
    a logo placed on a mostly-solid background.

    Only removes background pixels connected to the image border.
    This is intentionally NOT a global color replacement, so white parts
    inside a logo are not automatically deleted.
    """
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]

    if h < 8 or w < 8:
        return np.full((h, w), 255, np.uint8)

    # Sample a small border region.
    border = np.concatenate([
        rgb[:max(2, h // 30), :, :].reshape(-1, 3),
        rgb[-max(2, h // 30):, :, :].reshape(-1, 3),
        rgb[:, :max(2, w // 30), :].reshape(-1, 3),
        rgb[:, -max(2, w // 30):, :].reshape(-1, 3),
    ], axis=0)

    bg = np.median(border, axis=0).astype(np.float32)

    # Reject this heuristic when the border itself is highly varied.
    distances = np.linalg.norm(border.astype(np.float32) - bg, axis=1)
    if np.percentile(distances, 90) > 30:
        return np.full((h, w), 255, np.uint8)

    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    # Conservative tolerance. Connected-component requirement protects
    # similarly colored interior logo pixels.
    candidate = (dist <= 28).astype(np.uint8)

    # Flood fill from all four borders.
    flood = np.zeros((h + 2, w + 2), np.uint8)
    visited = np.zeros((h, w), np.uint8)
    mask = candidate.copy()

    from collections import deque
    q = deque()

    for x in range(w):
        if mask[0, x]: q.append((0, x))
        if mask[h - 1, x]: q.append((h - 1, x))
    for y in range(h):
        if mask[y, 0]: q.append((y, 0))
        if mask[y, w - 1]: q.append((y, w - 1))

    while q:
        y, x = q.popleft()
        if visited[y, x]:
            continue
        visited[y, x] = 1
        if not mask[y, x]:
            continue
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if mask[ny, nx]:
                    q.append((ny, nx))

    return (255 - visited * 255).astype(np.uint8)


def apply_alpha(img: Image.Image, alpha: np.ndarray) -> Image.Image:
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, "RGBA")


def postprocess_alpha(result: Image.Image) -> Image.Image:
    rgba = np.array(result.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]

    # Remove tiny isolated transparent/opaque noise without touching
    # meaningful anti-aliased edges.
    kernel = np.ones((2, 2), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)

    # Preserve the original RGB and only replace alpha.
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, "RGBA")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/remove")
async def remove_background(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED:
        raise HTTPException(415, "Upload PNG, JPG/JPEG, WEBP, or SVG.")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Maximum upload size is 15 MB.")

    img = decode_upload(raw, file.content_type)

    # Very large canvases can cause excessive RAM usage.
    # Keep enough resolution for professional logo output.
    max_side = 5000
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)),
            Image.Resampling.LANCZOS,
        )

    # If the source already has meaningful transparency, preserve it.
    src_alpha = np.asarray(img.getchannel("A"))
    already_transparent = np.mean(src_alpha < 250) > 0.01

    if already_transparent:
        result = img
    else:
        session = get_session()

        # AI segmentation + alpha matting.
        result = remove(
            img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            post_process_mask=True,
        )

        # For clean solid-background logos, the connected border heuristic
        # can outperform AI around perfectly flat edges. Use it only when
        # it actually finds a meaningful border-connected background.
        heuristic_alpha = solid_corner_background_mask(img)
        ai_alpha = np.asarray(result.convert("RGBA"))[:, :, 3]

        # Conservative blend: only force pixels transparent when both
        # approaches strongly agree that they are background.
        strong_bg = (heuristic_alpha < 20) & (ai_alpha < 80)
        final_alpha = ai_alpha.copy()
        final_alpha[strong_bg] = 0

        result = apply_alpha(img, final_alpha)
        result = postprocess_alpha(result)

    out = io.BytesIO()
    result.save(
        out,
        format="PNG",
        optimize=True,
        compress_level=6,
    )

    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": 'attachment; filename="logo-transparent.png"',
            "Cache-Control": "no-store",
        },
    )
