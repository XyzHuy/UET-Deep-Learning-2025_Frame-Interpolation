import * as ort from "onnxruntime-web/webgpu";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

export async function createInterpolationSession(modelUrl = "/models/model.onnx") {
  const webgpuAvailable = Boolean(navigator.gpu);
  const executionProviders = webgpuAvailable ? ["webgpu", "wasm"] : ["wasm"];

  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders,
    graphOptimizationLevel: "all",
  });

  return {
    session,
    provider: webgpuAvailable ? "webgpu" : "wasm",
    inputNames: session.inputNames,
    outputNames: session.outputNames,
  };
}

export async function runPairInterpolation(session, img0Tensor, img1Tensor) {
  const img0 = new ort.Tensor("float32", img0Tensor.data, img0Tensor.dims);
  const img1 = new ort.Tensor("float32", img1Tensor.data, img1Tensor.dims);
  const feeds = {
    img0,
    img1,
  };

  const results = await session.run(feeds);
  return results.pred ?? results[session.outputNames[0]];
}
