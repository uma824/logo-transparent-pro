# Logo Transparent Pro

A production-oriented web app for uploading a logo and downloading a transparent PNG.

## Important accuracy note

No background-removal system can honestly guarantee 100% automatic success for *every* possible logo. Logos can contain white/light artwork on white backgrounds, gradients, shadows, semi-transparent effects, textured backgrounds, JPEG artifacts, or multiple foreground objects.

This project therefore uses a hybrid pipeline:

1. **Logo-aware edge/background analysis** for simple solid backgrounds.
2. **BiRefNet via rembg** for difficult images.
3. **Alpha matting + post-processing** for cleaner edges.
4. **Transparent PNG output** with original resolution retained where practical.
5. The UI is designed so a future manual mask/refine editor can be added for the cases where absolute pixel-perfect control is required.

The current default AI model is `birefnet-general`; rembg supports BiRefNet and alpha matting. See the official documentation:
- https://github.com/danielgatis/rembg
- https://pypi.org/project/rembg/

## Project structure

- `backend/` FastAPI image-processing API
- `frontend/` responsive browser UI
- `Dockerfile` container deployment
- `docker-compose.yml` local deployment

## Run locally

### Option A — Docker

```bash
docker compose up --build
```

Then open:

http://localhost:8000

### Option B — Python

Python 3.11+ recommended.

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000

The first AI request downloads the selected model and can take longer.

## Production deployment

Recommended simple architecture:

- Frontend: served by the FastAPI container in this project, or deploy frontend separately to Vercel/Cloudflare Pages.
- Backend: Render, Railway, Fly.io, Google Cloud Run, AWS ECS, or a GPU provider for high throughput.
- For heavy traffic, run multiple workers/containers and use a queue.
- For privacy, do not persist uploads unless you explicitly need history.
- Add rate limiting, file-size limits, MIME validation, EXIF stripping, and automatic deletion of temporary files.

## Accuracy / quality

For a logo-specific commercial product, the strongest production approach is:

- Use the AI model as the first pass.
- Detect simple solid-color backgrounds and use connected-region removal when appropriate.
- Offer an optional "Refine" editor where the user can paint foreground/background.
- Keep the original pixels; only modify the alpha channel whenever possible.
- Export PNG RGBA.
- Never convert a transparent result to JPEG.

That combination is much closer to "pixel-perfect" than claiming an AI model alone is 100% correct.
