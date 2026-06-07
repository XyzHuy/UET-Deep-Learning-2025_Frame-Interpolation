import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { AlertCircle, Cpu, Download, Film, Loader2, Play, Upload } from "lucide-react";
import {
  checkBackend,
  getGpuStatus,
  getVideoJob,
  getVideoJobResult,
  listDemoVideos,
  releaseGpuMemory,
  startVideoJob,
} from "./api";
import "./styles.css";

const MULTIPLIERS = [2, 4, 8, 16, 32];

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [demoVideos, setDemoVideos] = useState([]);
  const [selectedDemo, setSelectedDemo] = useState("");
  const [fpsMultiplier, setFpsMultiplier] = useState(16);
  const [ffmpegPreset, setFfmpegPreset] = useState("veryfast");
  const [jobProgress, setJobProgress] = useState(null);
  const [refinerScale, setRefinerScale] = useState(0.5);
  const [skipRefiner, setSkipRefiner] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [busy, setBusy] = useState(false);
  const [outputBlob, setOutputBlob] = useState(null);
  const [backendState, setBackendState] = useState("unchecked");

  const uploadedVideoPreview = useObjectUrl(videoFile);
  const selectedDemoInfo = demoVideos.find((video) => video.name === selectedDemo);
  const videoPreview = selectedDemoInfo?.url || uploadedVideoPreview;
  const outputUrl = useObjectUrl(outputBlob);

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
          <h1>Frame Interpolation Deploy</h1>
          <p>PyTorch backend for video interpolation.</p>
        </div>
        <div className={`status-pill ${statusTone}`}>
          {busy ? <Loader2 className="spin" size={16} /> : <Cpu size={16} />}
          <span>{status}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <div className="section-title">
            <Film size={16} />
            <span>Video job</span>
          </div>

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
          <Field label="Refiner">
            <select value={refinerScale} onChange={(event) => setRefinerScale(Number(event.target.value))}>
              <option value={1}>full</option>
              <option value={0.5}>half</option>
              <option value={0.25}>quarter</option>
            </select>
          </Field>
          <label className="checkline">
            <input type="checkbox" checked={skipRefiner} onChange={(event) => setSkipRefiner(event.target.checked)} />
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
          <button className="primary dark" onClick={runVideoBackend} disabled={busy}>
            <Play size={16} />
            Run video x{fpsMultiplier}
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
          <div className="hint">
            <AlertCircle size={15} />
            <span>{backendState}</span>
          </div>
        </aside>

        <section className="stage">
          <div className="video-stage">
            <Preview title="Input" src={videoPreview} />
            <Preview title="Output" src={outputUrl} />
          </div>

          {outputBlob && (
            <a className="download" href={outputUrl} download="interpolated.mp4">
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

function Preview({ title, src }) {
  return (
    <div className="preview">
      <div className="preview-title">{title}</div>
      {src ? <video src={src} controls /> : <div className="empty">No file</div>}
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
