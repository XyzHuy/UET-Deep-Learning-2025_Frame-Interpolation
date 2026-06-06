export function tensorToImageData(tensor) {
  const [, channels, height, width] = tensor.dims;
  if (channels !== 3) {
    throw new Error(`Expected 3 output channels, received ${channels}`);
  }

  const imageData = new ImageData(width, height);
  const plane = width * height;

  for (let i = 0; i < plane; i += 1) {
    const rgba = i * 4;
    imageData.data[rgba] = Math.round(clamp01(tensor.data[i]) * 255);
    imageData.data[rgba + 1] = Math.round(clamp01(tensor.data[plane + i]) * 255);
    imageData.data[rgba + 2] = Math.round(clamp01(tensor.data[plane * 2 + i]) * 255);
    imageData.data[rgba + 3] = 255;
  }

  return imageData;
}

export function imageDataToBlob(imageData) {
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);

  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Could not encode image result"));
    }, "image/png");
  });
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}
