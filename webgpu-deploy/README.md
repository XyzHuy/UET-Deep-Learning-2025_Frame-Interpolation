# WebGPU Deploy

This folder contains a small deployment shell around the frame interpolation model:

- `frontend/`: Vite + React app that runs `model.onnx` with ONNX Runtime Web.
- `backend/`: FastAPI fallback that reuses the current PyTorch `VideoInterpolator`.
- `export_onnx.py`: fixed-size ONNX export helper.

## Feasibility

The model is feasible to export to ONNX for fixed input sizes. The current checked export is:

- input: `img0`, `img1`, both `1x3x256x256`
- output: `pred`, `1x3x256x256`
- refiner: `refiner_scale=0.5`
- ONNX size: about 22 MB

Two model changes were needed:

- `safe_shift` now uses `pad` + `slice` instead of in-place slice assignment.
- low-resolution refiner resize now uses fixed `scale_factor`, which traces cleanly.

Browser WebGPU is a good first target for image-pair interpolation and short experiments. Full video interpolation in the browser is possible but should be treated as phase 2 because decoding, recursive x16 frame generation, memory pressure, and MP4 encoding are bigger problems than the model call itself. The backend path is the practical first version for video.

References:

- ONNX Runtime WebGPU docs: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- ONNX Runtime Web install/import docs: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
- PyTorch ONNX exporter docs: https://docs.pytorch.org/docs/stable/onnx.html

## Export ONNX

From the repository root:

```bash
python3 webgpu-deploy/export_onnx.py \
  --height 256 \
  --width 256 \
  --refiner_scale 0.5 \
  --output webgpu-deploy/frontend/public/models/model.onnx \
  --verify
```

For larger browser exports, prefer fixed sizes such as `512x512` or `768x432` and benchmark memory before deploying. Dynamic height/width export is available with `--dynamic`, but fixed shapes are safer for ONNX Runtime WebGPU.

## Frontend

Use Node.js 18 or newer.

```bash
cd webgpu-deploy/frontend
npm install
npm run dev
```

Deploy `frontend/` to Cloudflare Pages, Vercel, or Netlify. If the host complains about large static files, move `model.onnx` to Hugging Face Hub, GitHub Release, or R2, then set the model URL in the UI.

Build command:

```bash
npm run build
```

Output directory:

```text
dist
```

## Backend

Local:

```bash
cd webgpu-deploy/backend
python3 -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker from the repository root:

```bash
docker build -f webgpu-deploy/backend/Dockerfile -t vfi-backend .
docker run --rm -p 8000:8000 vfi-backend
```

Useful environment variables:

```text
MODEL_PATH=/app/checkpoint/model.pth
DEVICE=auto
ALLOWED_ORIGINS=*
KEEP_MODEL_WARM=0
USE_CUDA_APPLY_SHIFT=1
```

GPU management endpoints:

```text
GET  /api/gpu/status
POST /api/gpu/release
```

