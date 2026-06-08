import gc
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Video_Inference import VideoInterpolator  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "checkpoint" / "model.pth"))
DEVICE = os.getenv("DEVICE", "auto")
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
DEMO_DIR = ROOT / "video_demo"
INTERPOLATOR_LOCK = threading.Lock()
KEEP_MODEL_WARM = os.getenv("KEEP_MODEL_WARM", "0").strip().lower() in {"1", "true", "yes", "on"}
_INTERPOLATOR = None
JOBS = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="Frame Interpolation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(DEMO_DIR)), name="demo")


def get_interpolator():
    global _INTERPOLATOR
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    if _INTERPOLATOR is None:
        _INTERPOLATOR = VideoInterpolator(
            model_path=str(MODEL_PATH),
            device=DEVICE,
            refiner_scale=0.5,
            skip_refiner=False,
        )

    return _INTERPOLATOR


@app.get("/api/health")
def health():
    model_exists = MODEL_PATH.exists()
    model_loaded = _INTERPOLATOR is not None
    device = str(_INTERPOLATOR.device) if model_loaded else DEVICE

    return {
        "ok": True,
        "model_exists": model_exists,
        "model_path": str(MODEL_PATH),
        "model_loaded": model_loaded,
        "keep_model_warm": KEEP_MODEL_WARM,
        "device": device,
    }


@app.get("/api/gpu/status")
def gpu_status():
    return gpu_status_payload()


@app.post("/api/gpu/release")
def release_gpu():
    with INTERPOLATOR_LOCK:
        return release_interpolator()


@app.get("/api/demo-videos")
def demo_videos():
    videos = []
    if DEMO_DIR.exists():
        for path in sorted(DEMO_DIR.glob("*_6fps.mp4")):
            videos.append({
                "name": path.name,
                "url": f"/demo/{path.name}",
            })
    return {"videos": videos}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    return public_job(job)


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = get_job(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job is not done")

    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="interpolated.mp4",
        background=BackgroundTask(lambda: cleanup_job(job_id)),
    )


@app.post("/api/interpolate/image")
async def interpolate_image(
    frame0: UploadFile = File(...),
    frame1: UploadFile = File(...),
    refiner_scale: float = Form(0.5),
    skip_refiner: bool = Form(False),
):
    temp_dir = Path(tempfile.mkdtemp(prefix="vfi-image-"))
    try:
        path0 = temp_dir / safe_name(frame0.filename, "frame0.png")
        path1 = temp_dir / safe_name(frame1.filename, "frame1.png")
        await save_upload(frame0, path0)
        await save_upload(frame1, path1)

        img0 = cv2.imread(str(path0), cv2.IMREAD_COLOR)
        img1 = cv2.imread(str(path1), cv2.IMREAD_COLOR)
        if img0 is None or img1 is None:
            raise HTTPException(status_code=400, detail="Could not decode one of the input images")
        if img0.shape != img1.shape:
            raise HTTPException(status_code=400, detail="Input images must have the same size")

        with INTERPOLATOR_LOCK:
            try:
                interpolator = get_interpolator()
                interpolator.configure_runtime(
                    refiner_scale=refiner_scale,
                    skip_refiner=skip_refiner,
                )
                pred_rgb = interpolator.interpolate_frame(
                    cv2.cvtColor(img0, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
                )
            finally:
                release_interpolator_if_cold()
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode output image")

        return Response(content=encoded.tobytes(), media_type="image/png")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/interpolate/video")
async def interpolate_video(
    file: UploadFile | None = File(None),
    demo_video: str | None = Form(None),
    fps_multiplier: int = Form(2),
    refiner_scale: float = Form(0.5),
    skip_refiner: bool = Form(False),
    inference_mode: str = Form("auto"),
    crf: int = Form(18),
    ffmpeg_preset: str = Form("veryfast"),
):
    if fps_multiplier not in {2, 4, 8, 16, 32}:
        raise HTTPException(status_code=400, detail="fps_multiplier must be one of 2, 4, 8, 16, 32")
    inference_mode = normalize_inference_mode(inference_mode)

    temp_dir = Path(tempfile.mkdtemp(prefix="vfi-video-"))
    input_path = temp_dir / "input.mp4"
    output_path = temp_dir / "interpolated.mp4"

    try:
        if demo_video:
            input_path = resolve_demo_video(demo_video)
        elif file is not None:
            input_path = temp_dir / safe_name(file.filename, "input.mp4")
            await save_upload(file, input_path)
        else:
            raise HTTPException(status_code=400, detail="Provide either a video file or demo_video")

        with INTERPOLATOR_LOCK:
            try:
                interpolator = get_interpolator()
                interpolator.configure_runtime(
                    refiner_scale=refiner_scale,
                    skip_refiner=skip_refiner,
                    inference_mode=inference_mode,
                )
                interpolator.interpolate_video(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    fps_multiplier=fps_multiplier,
                    use_ffmpeg=True,
                    crf=crf,
                    ffmpeg_preset=normalize_ffmpeg_preset(ffmpeg_preset),
                )
            finally:
                release_interpolator_if_cold()
    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(error)) from error

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="interpolated.mp4",
        background=BackgroundTask(lambda: shutil.rmtree(temp_dir, ignore_errors=True)),
    )


@app.post("/api/interpolate/video/start")
async def start_interpolate_video(
    file: UploadFile | None = File(None),
    demo_video: str | None = Form(None),
    fps_multiplier: int = Form(2),
    refiner_scale: float = Form(0.5),
    skip_refiner: bool = Form(False),
    inference_mode: str = Form("auto"),
    crf: int = Form(18),
    ffmpeg_preset: str = Form("veryfast"),
):
    if fps_multiplier not in {2, 4, 8, 16, 32}:
        raise HTTPException(status_code=400, detail="fps_multiplier must be one of 2, 4, 8, 16, 32")
    inference_mode = normalize_inference_mode(inference_mode)

    job_id = uuid.uuid4().hex
    temp_dir = Path(tempfile.mkdtemp(prefix=f"vfi-job-{job_id}-"))
    input_path = temp_dir / "input.mp4"
    output_path = temp_dir / "interpolated.mp4"

    if demo_video:
        input_path = resolve_demo_video(demo_video)
    elif file is not None:
        input_path = temp_dir / safe_name(file.filename, "input.mp4")
        await save_upload(file, input_path)
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Provide either a video file or demo_video")

    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0.0,
        "completed": 0,
        "total": 1,
        "message": "Queued",
        "error": None,
        "output_path": str(output_path),
        "temp_dir": str(temp_dir),
        "inference_mode": inference_mode,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    thread = threading.Thread(
        target=run_video_job,
        args=(
            job_id,
            str(input_path),
            str(output_path),
            fps_multiplier,
            refiner_scale,
            skip_refiner,
            inference_mode,
            crf,
            normalize_ffmpeg_preset(ffmpeg_preset),
        ),
        daemon=True,
    )
    thread.start()
    return public_job(job)


async def save_upload(upload: UploadFile, path: Path):
    with path.open("wb") as out_file:
        while chunk := await upload.read(1024 * 1024):
            out_file.write(chunk)


def safe_name(filename, fallback):
    name = Path(filename or fallback).name
    return name or fallback


def resolve_demo_video(name):
    path = DEMO_DIR / Path(name).name
    if not path.exists() or path.parent != DEMO_DIR or not path.name.endswith("_6fps.mp4"):
        raise HTTPException(status_code=404, detail="Demo video not found")
    return path


def normalize_ffmpeg_preset(value):
    allowed = {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}
    value = (value or "veryfast").strip().lower()
    if value not in allowed:
        raise HTTPException(status_code=400, detail=f"ffmpeg_preset must be one of {', '.join(sorted(allowed))}")
    return value


def normalize_inference_mode(value):
    allowed = {"auto", "full_frame"}
    value = (value or "auto").strip().lower()
    if value not in allowed:
        raise HTTPException(status_code=400, detail=f"inference_mode must be one of {', '.join(sorted(allowed))}")
    return value


def release_interpolator_if_cold():
    if not KEEP_MODEL_WARM:
        release_interpolator()


def release_interpolator():
    global _INTERPOLATOR

    was_loaded = _INTERPOLATOR is not None
    interpolator = _INTERPOLATOR
    device = getattr(interpolator, "device", None)
    _INTERPOLATOR = None
    del interpolator

    gc.collect()
    if device is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass

    return {
        "released": was_loaded,
        **gpu_status_payload(include_memory=False),
    }


def gpu_status_payload(include_memory=None):
    model_loaded = _INTERPOLATOR is not None
    payload = {
        "model_loaded": model_loaded,
        "keep_model_warm": KEEP_MODEL_WARM,
        "device": str(_INTERPOLATOR.device) if model_loaded else DEVICE,
        "cuda_available": torch.cuda.is_available(),
    }

    if include_memory is None:
        include_memory = model_loaded

    if include_memory and torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info()
        payload.update({
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_used_mb": (total_memory - free_memory) / (1024 ** 2),
            "cuda_free_mb": free_memory / (1024 ** 2),
            "cuda_total_mb": total_memory / (1024 ** 2),
            "torch_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "torch_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        })

    return payload


def run_video_job(
    job_id,
    input_path,
    output_path,
    fps_multiplier,
    refiner_scale,
    skip_refiner,
    inference_mode,
    crf,
    ffmpeg_preset,
):
    update_job(job_id, status="running", message="Loading model")

    def progress_callback(completed, total):
        progress = completed / total if total else 0.0
        update_job(
            job_id,
            completed=completed,
            total=total,
            progress=progress,
            message=f"Interpolating {completed}/{total}",
        )

    try:
        with INTERPOLATOR_LOCK:
            try:
                interpolator = get_interpolator()
                interpolator.configure_runtime(
                    refiner_scale=refiner_scale,
                    skip_refiner=skip_refiner,
                    inference_mode=inference_mode,
                )
                interpolator.interpolate_video(
                    input_path=input_path,
                    output_path=output_path,
                    fps_multiplier=fps_multiplier,
                    use_ffmpeg=True,
                    crf=crf,
                    ffmpeg_preset=ffmpeg_preset,
                    progress_callback=progress_callback,
                )
            finally:
                release_interpolator_if_cold()

        update_job(job_id, status="done", progress=1.0, message="Done")
    except Exception as error:
        release_interpolator_if_cold()
        update_job(job_id, status="error", error=str(error), message="Failed")


def update_job(job_id, **updates):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


def public_job(job):
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "completed": job["completed"],
        "total": job["total"],
        "message": job["message"],
        "error": job["error"],
        "inference_mode": job.get("inference_mode", "auto"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


def cleanup_job(job_id):
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)
    if job is not None:
        shutil.rmtree(job["temp_dir"], ignore_errors=True)
