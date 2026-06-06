import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertCircle,
  Cpu,
  Download,
  Film,
  Image as ImageIcon,
  Loader2,
  Play,
  Upload,
} from "lucide-react";
import {
  checkBackend,
  getGpuStatus,
  getVideoJob,
  getVideoJobResult,
  listDemoVideos,
  profileVideoWithBackend,
  releaseGpuMemory,
  startVideoJob,
} from "./api";
import { createInterpolationSession, runPairInterpolation } from "./inference/loadModel";
import { fileToImageBitmap, imageBitmapToTensorData } from "./inference/preprocess";
import { imageDataToBlob, tensorToImageData } from "./inference/postprocess";
import "./styles.css";

const MULTIPLIERS = [2, 4, 8, 16, 32];

function App() {
  const outputCanvasRef = useRef(null);
  const [mode, setMode] = useState("image");
  const [modelUrl, setModelUrl] = useState("/models/model.onnx");
  const [sessionInfo, setSessionInfo] = useState(null);
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [demoVideos, setDemoVideos] = useState([]);
  const [selectedDemo, setSelectedDemo] = useState("");
  const [targetSize, setTargetSize] = useState({ width: 256, height: 256 });
  const [fpsMultiplier, setFpsMultiplier] = useState(16);
  const [batchSize, setBatchSize] = useState(1);
  const [profileCandidates, setProfileCandidates] = useState("1,2,4,8");
  const [profileRows, setProfileRows] = useState([]);
  const [tileSize, setTileSize] = useState("auto");
  const [ffmpegPreset, setFfmpegPreset] = useState("veryfast");
  const [jobProgress, setJobProgress] = useState(null);
  const [refinerScale, setRefinerScale] = useState(0.5);
  const [skipRefiner, setSkipRefiner] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [busy, setBusy] = useState(false);
  const [outputBlob, setOutputBlob] = useState(null);
  const [backendState, setBackendState] = useState("unchecked");

  const imagePreviewA = useObjectUrl(imageA);
  const imagePreviewB = useObjectUrl(imageB);
  const uploadedVideoPreview = useObjectUrl(videoFile);
  const selectedDemoInfo = demoVideos.find((video) => video.name === selectedDemo);
  const videoPreview = selectedDemoInfo?.url || uploadedVideoPreview;
  const outputUrl = useObjectUrl(outputBlob);
  const batchEstimate = useMemo(
    () => estimateVramForBatch(profileRows, batchSize),
    [profileRows, batchSize],
  );

  useEffect(() => {
    let cancelled = false;
    listDemoVideos()
      .then((videos) => {
        if (!cancelled) setDemoVideos(videos);
      })
      .catch(() => {
        if (!cancelled) setDemoVideos([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function loadModel() {
    setBusy(true);
    setStatus("Loading ONNX session");
    try {
      const info = await createInterpolationSession(modelUrl);
      setSessionInfo(info);
      setStatus(`Model loaded on ${info.provider.toUpperCase()}`);
    } catch (error) {
      setStatus(`Model load failed: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function runImagePair() {
    if (!sessionInfo?.session) {
      setStatus("Load a model first");
      return;
    }
    if (!imageA || !imageB) {
      setStatus("Select two frames");
      return;
    }

    setBusy(true);
    setOutputBlob(null);
    setStatus("Interpolating image pair");
    try {
      const size = { width: targetSize.width, height: targetSize.height };
      const bitmapA = await fileToImageBitmap(imageA, size);
      const bitmapB = await fileToImageBitmap(imageB, size);
      const tensorA = imageBitmapToTensorData(bitmapA);
      const tensorB = imageBitmapToTensorData(bitmapB);
      const pred = await runPairInterpolation(sessionInfo.session, tensorA, tensorB);
      const imageData = tensorToImageData(pred);

      const canvas = outputCanvasRef.current;
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      canvas.getContext("2d").putImageData(imageData, 0, 0);

      const blob = await imageDataToBlob(imageData);
      setOutputBlob(blob);
      setStatus("Done");
    } catch (error) {
      setStatus(`Inference failed: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function probeBackend() {
    setBusy(true);
    setBackendState("checking");
    setStatus("Checking backend");
    try {
      const info = await checkBackend();
      const videos = await listDemoVideos();
      const gpu = await getGpuStatus();
      setDemoVideos(videos);
      setBackendState(formatGpuState(gpu, info.device || "online"));
      setStatus(`Backend online: ${info.device}`);
    } catch (error) {
      setBackendState("offline");
      setStatus(`Backend offline: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function runVideoProfile() {
    if (!videoFile && !selectedDemo) {
      setStatus("Select a video");
      return;
    }

    setBusy(true);
    setProfileRows([]);
    setStatus("Profiling VRAM");
    try {
      const profile = await profileVideoWithBackend({
        file: selectedDemo ? null : videoFile,
        demoVideo: selectedDemo,
        batchSizes: profileCandidates,
        tileSize,
        refinerScale,
        skipRefiner,
      });
      setProfileRows(profile.rows || []);
      const recommended = [...(profile.rows || [])].reverse().find((row) => row.ok && row.vram_used_percent < 85);
      if (recommended) setBatchSize(recommended.batch_size);
      setStatus(profile.cuda ? `Profiled ${profile.width}x${profile.height}` : `Profile unavailable on ${profile.device}`);
    } catch (error) {
      setStatus(`Profile failed: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function releaseBackendGpu() {
    setBusy(true);
    setStatus("Releasing GPU");
    try {
      const info = await releaseGpuMemory();
      setBackendState(formatGpuState(info, "released"));
      setStatus(info.released ? "GPU released" : "No model loaded");
    } catch (error) {
      setStatus(`Release failed: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function runVideoBackend() {
    if (!videoFile && !selectedDemo) {
      setStatus("Select a video");
      return;
    }

    setBusy(true);
    setOutputBlob(null);
    setJobProgress({ progress: 0, completed: 0, total: 1, message: "Queued" });
    setStatus(`Interpolating video x${fpsMultiplier}`);
    try {
      const job = await startVideoJob({
        file: selectedDemo ? null : videoFile,
        demoVideo: selectedDemo,
        fpsMultiplier,
        batchSize,
        tileSize,
        refinerScale,
        skipRefiner,
        ffmpegPreset,
      });
      let current = job;
      setJobProgress(current);

      while (current.status !== "done" && current.status !== "error") {
        await sleep(800);
        current = await getVideoJob(job.id);
        setJobProgress(current);
      }

      if (current.status === "error") {
        throw new Error(current.error || "Video job failed");
      }

      const blob = await getVideoJobResult(job.id);
      setOutputBlob(blob);
      setJobProgress({ ...current, progress: 1, message: "Done" });
      setStatus("Video ready");
    } catch (error) {
      setStatus(`Backend inference failed: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  const statusTone = useMemo(() => {
    if (status.toLowerCase().includes("failed") || status.toLowerCase().includes("offline")) return "bad";
    if (status.toLowerCase().includes("done") || status.toLowerCase().includes("ready")) return "good";
    return "neutral";
  }, [status]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Frame Interpolation Lab</h1>
          <p>WebGPU first, PyTorch backend fallback.</p>
        </div>
        <div className={`status-pill ${statusTone}`}>
          {busy ? <Loader2 className="spin" size={16} /> : <Cpu size={16} />}
          <span>{status}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <div className="segmented">
            <button className={mode === "image" ? "active" : ""} onClick={() => setMode("image")}>
              <ImageIcon size={16} />
              Image
            </button>
            <button className={mode === "video" ? "active" : ""} onClick={() => setMode("video")}>
              <Film size={16} />
              Video
            </button>
          </div>

          {mode === "image" ? (
            <>
              <Field label="ONNX model">
                <input value={modelUrl} onChange={(event) => setModelUrl(event.target.value)} />
              </Field>
              <div className="grid-2">
                <Field label="Width">
                  <input
                    type="number"
                    min="32"
                    step="32"
                    value={targetSize.width}
                    onChange={(event) => setTargetSize({ ...targetSize, width: Number(event.target.value) })}
                  />
                </Field>
                <Field label="Height">
                  <input
                    type="number"
                    min="32"
                    step="32"
                    value={targetSize.height}
                    onChange={(event) => setTargetSize({ ...targetSize, height: Number(event.target.value) })}
                  />
                </Field>
              </div>
              <FileDrop label="Frame A" accept="image/*" onChange={setImageA} />
              <FileDrop label="Frame B" accept="image/*" onChange={setImageB} />
              <button className="primary" onClick={loadModel} disabled={busy}>
                <Upload size={16} />
                Load model
              </button>
              <button className="primary dark" onClick={runImagePair} disabled={busy}>
                <Play size={16} />
                Run pair
              </button>
            </>
          ) : (
            <>
              <Field label="Toy video">
                <select
                  value={selectedDemo}
                  onChange={(event) => {
                    setSelectedDemo(event.target.value);
                    if (event.target.value) setVideoFile(null);
                  }}
                >
                  <option value="">uploaded file</option>
                  {demoVideos.map((video) => (
                    <option key={video.name} value={video.name}>
                      {video.name}
                    </option>
                  ))}
                </select>
              </Field>
              <FileDrop
                label="Video"
                accept="video/*"
                onChange={(file) => {
                  setVideoFile(file);
                  if (file) setSelectedDemo("");
                }}
              />
              <Field label={`Multiplier x${fpsMultiplier}`}>
                <input
                  type="range"
                  min="0"
                  max={MULTIPLIERS.length - 1}
                  value={MULTIPLIERS.indexOf(fpsMultiplier)}
                  onChange={(event) => setFpsMultiplier(MULTIPLIERS[Number(event.target.value)])}
                />
              </Field>
              <div className="grid-2">
                <Field label={`Batch ${batchSize}`}>
                  <input
                    type="number"
                    min="1"
                    max="64"
                    value={batchSize}
                    onChange={(event) => setBatchSize(Math.max(1, Number(event.target.value)))}
                  />
                </Field>
                <Field label="Candidates">
                  <input value={profileCandidates} onChange={(event) => setProfileCandidates(event.target.value)} />
                </Field>
              </div>
              <Field label="Tile">
                <select value={tileSize} onChange={(event) => setTileSize(event.target.value)}>
                  <option value="auto">auto</option>
                  <option value="384">384</option>
                  <option value="512">512</option>
                  <option value="640">640</option>
                </select>
              </Field>
              <Field label="Encoder">
                <select value={ffmpegPreset} onChange={(event) => setFfmpegPreset(event.target.value)}>
                  <option value="ultrafast">ultrafast</option>
                  <option value="superfast">superfast</option>
                  <option value="veryfast">veryfast</option>
                  <option value="faster">faster</option>
                  <option value="fast">fast</option>
                  <option value="medium">medium</option>
                </select>
              </Field>
              {batchEstimate && (
                <div className="estimate-line">
                  <span>Batch {batchSize}</span>
                  <strong>{Math.round(batchEstimate.devicePeakMb)} MB</strong>
                  <span>{batchEstimate.percent.toFixed(1)}%</span>
                </div>
              )}
              <Field label="Refiner">
                <select value={refinerScale} onChange={(event) => setRefinerScale(Number(event.target.value))}>
                  <option value={1}>full</option>
                  <option value={0.5}>half</option>
                  <option value={0.25}>quarter</option>
                </select>
              </Field>
              <label className="checkline">
                <input
                  type="checkbox"
                  checked={skipRefiner}
                  onChange={(event) => setSkipRefiner(event.target.checked)}
                />
                <span>Skip refiner</span>
              </label>
              <button className="primary" onClick={probeBackend} disabled={busy}>
                <Cpu size={16} />
                Check backend
              </button>
              <button className="primary" onClick={releaseBackendGpu} disabled={busy}>
                <Cpu size={16} />
                Release GPU
              </button>
              <button className="primary" onClick={runVideoProfile} disabled={busy}>
                <Cpu size={16} />
                Profile VRAM
              </button>
              <button className="primary dark" onClick={runVideoBackend} disabled={busy}>
                <Play size={16} />
                Run video x{fpsMultiplier} b{batchSize}
              </button>
              {jobProgress && (
                <div className="progress-box">
                  <div className="progress-meta">
                    <span>{jobProgress.message}</span>
                    <strong>{Math.round((jobProgress.progress || 0) * 100)}%</strong>
                  </div>
                  <div className="progress-track">
                    <div style={{ width: `${Math.min(100, Math.max(0, (jobProgress.progress || 0) * 100))}%` }} />
                  </div>
                  <div className="progress-count">
                    {jobProgress.completed || 0}/{jobProgress.total || 1}
                  </div>
                </div>
              )}
              {profileRows.length > 0 && (
                <div className="metric-table">
                  <div className="metric-head">
                    <span>Batch</span>
                    <span>Peak est.</span>
                    <span>VRAM</span>
                  </div>
                  {profileRows.map((row) => (
                    <button
                      key={row.batch_size}
                      className={row.ok ? "metric-row" : "metric-row bad"}
                      onClick={() => row.ok && setBatchSize(row.batch_size)}
                      disabled={!row.ok}
                    >
                      <span>{row.batch_size}</span>
                      <span>{row.ok ? `${Math.round(displayPeakMb(row))} MB` : "OOM"}</span>
                      <span>{row.ok ? `${row.vram_used_percent.toFixed(1)}%` : "-"}</span>
                    </button>
                  ))}
                </div>
              )}
              <div className="hint">
                <AlertCircle size={15} />
                <span>{backendState}</span>
              </div>
            </>
          )}
        </aside>

        <section className="stage">
          {mode === "image" ? (
            <div className="preview-grid">
              <Preview title="Frame A" src={imagePreviewA} />
              <Preview title="Frame B" src={imagePreviewB} />
              <div className="preview output">
                <div className="preview-title">Output</div>
                <canvas ref={outputCanvasRef} />
              </div>
            </div>
          ) : (
            <div className="video-stage">
              <Preview title="Input" src={videoPreview} video />
              <Preview title="Output" src={outputUrl} video />
            </div>
          )}

          {outputBlob && (
            <a className="download" href={outputUrl} download={mode === "image" ? "interpolated.png" : "interpolated.mp4"}>
              <Download size={16} />
              Download result
            </a>
          )}
        </section>
      </section>
    </main>
  );
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function FileDrop({ label, accept, onChange }) {
  return (
    <label className="file-drop">
      <Upload size={17} />
      <span>{label}</span>
      <input type="file" accept={accept} onChange={(event) => onChange(event.target.files?.[0] ?? null)} />
    </label>
  );
}

function Preview({ title, src, video = false }) {
  return (
    <div className="preview">
      <div className="preview-title">{title}</div>
      {src ? (
        video ? (
          <video src={src} controls />
        ) : (
          <img src={src} alt={title} />
        )
      ) : (
        <div className="empty">No file</div>
      )}
    </div>
  );
}

function useObjectUrl(value) {
  const [url, setUrl] = useState("");

  useEffect(() => {
    if (!value) {
      setUrl("");
      return undefined;
    }

    const nextUrl = URL.createObjectURL(value);
    setUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [value]);

  return url;
}

function estimateVramForBatch(rows, batchSize) {
  const measured = rows
    .filter((row) => row.ok && Number.isFinite(displayPeakMb(row)) && Number.isFinite(row.total_vram_mb))
    .sort((a, b) => a.batch_size - b.batch_size);

  if (measured.length === 0) return null;

  const exact = measured.find((row) => row.batch_size === batchSize);
  if (exact) {
    return {
      devicePeakMb: displayPeakMb(exact),
      percent: displayPeakMb(exact) / exact.total_vram_mb * 100,
    };
  }

  const totalVramMb = measured[0].total_vram_mb;
  if (measured.length === 1) {
    return {
      devicePeakMb: displayPeakMb(measured[0]),
      percent: displayPeakMb(measured[0]) / totalVramMb * 100,
    };
  }

  const lower = [...measured].reverse().find((row) => row.batch_size < batchSize);
  const upper = measured.find((row) => row.batch_size > batchSize);
  const left = lower || measured[0];
  const right = upper || measured[measured.length - 1];
  const slope = right.batch_size === left.batch_size
    ? 0
    : (displayPeakMb(right) - displayPeakMb(left)) / (right.batch_size - left.batch_size);
  const devicePeakMb = Math.max(
    0,
    displayPeakMb(left) + slope * (batchSize - left.batch_size),
  );

  return {
    devicePeakMb,
    percent: devicePeakMb / totalVramMb * 100,
  };
}

function displayPeakMb(row) {
  return row.conservative_peak_used_mb ?? row.device_peak_used_mb;
}

function formatGpuState(info, fallback) {
  if (!info.cuda_available) return fallback;
  if (Number.isFinite(info.cuda_used_mb)) {
    return `${info.cuda_used_mb.toFixed(0)} MB used`;
  }
  return info.model_loaded ? "model loaded" : "cuda ready";
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

createRoot(document.getElementById("root")).render(<App />);
