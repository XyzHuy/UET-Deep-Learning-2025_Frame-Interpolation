# Deploy

This folder contains the deployment shell around the frame interpolation model:

- `frontend/`: Vite + React app for submitting video interpolation jobs.
- `backend/`: FastAPI service that reuses the current PyTorch `VideoInterpolator`.

## Frontend

Use Node.js 18 or newer.

```bash
cd deploy/frontend
npm install
npm run dev
```

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
cd deploy/backend
python3 -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker from the repository root:

```bash
docker build -f deploy/backend/Dockerfile -t vfi-backend .
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
