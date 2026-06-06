const DEFAULT_BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export async function checkBackend(baseUrl = DEFAULT_BACKEND_URL) {
  const response = await fetch(`${baseUrl}/api/health`);
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`);
  }
  return response.json();
}

export async function listDemoVideos(baseUrl = DEFAULT_BACKEND_URL) {
  const response = await fetch(`${baseUrl}/api/demo-videos`);
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`);
  }
  const data = await response.json();
  return data.videos.map((video) => ({
    ...video,
    url: `${baseUrl}${video.url}`,
  }));
}

export async function getGpuStatus(baseUrl = DEFAULT_BACKEND_URL) {
  const response = await fetch(`${baseUrl}/api/gpu/status`);
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`);
  }
  return response.json();
}

export async function releaseGpuMemory(baseUrl = DEFAULT_BACKEND_URL) {
  const response = await fetch(`${baseUrl}/api/gpu/release`, {
    method: "POST",
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Backend returned ${response.status}`);
  }
  return response.json();
}

export async function interpolateVideoWithBackend({
  file,
  demoVideo,
  fpsMultiplier,
  refinerScale,
  skipRefiner,
  ffmpegPreset,
  baseUrl = DEFAULT_BACKEND_URL,
}) {
  const formData = new FormData();
  if (file) formData.append("file", file);
  if (demoVideo) formData.append("demo_video", demoVideo);
  formData.append("fps_multiplier", String(fpsMultiplier));
  formData.append("refiner_scale", String(refinerScale));
  formData.append("skip_refiner", String(skipRefiner));
  formData.append("ffmpeg_preset", ffmpegPreset);

  const response = await fetch(`${baseUrl}/api/interpolate/video`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Backend returned ${response.status}`);
  }

  return response.blob();
}

export async function startVideoJob({
  file,
  demoVideo,
  fpsMultiplier,
  refinerScale,
  skipRefiner,
  ffmpegPreset,
  baseUrl = DEFAULT_BACKEND_URL,
}) {
  const formData = new FormData();
  if (file) formData.append("file", file);
  if (demoVideo) formData.append("demo_video", demoVideo);
  formData.append("fps_multiplier", String(fpsMultiplier));
  formData.append("refiner_scale", String(refinerScale));
  formData.append("skip_refiner", String(skipRefiner));
  formData.append("ffmpeg_preset", ffmpegPreset);

  const response = await fetch(`${baseUrl}/api/interpolate/video/start`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Backend returned ${response.status}`);
  }

  return response.json();
}

export async function getVideoJob(jobId, baseUrl = DEFAULT_BACKEND_URL) {
  const response = await fetch(`${baseUrl}/api/jobs/${jobId}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Backend returned ${response.status}`);
  }
  return response.json();
}

export async function getVideoJobResult(jobId, baseUrl = DEFAULT_BACKEND_URL) {
  const response = await fetch(`${baseUrl}/api/jobs/${jobId}/result`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Backend returned ${response.status}`);
  }
  return response.blob();
}
