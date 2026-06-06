export async function fileToImageBitmap(file, targetSize) {
  const bitmap = await createImageBitmap(file);

  if (!targetSize) {
    return bitmap;
  }

  const canvas = new OffscreenCanvas(targetSize.width, targetSize.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, targetSize.width, targetSize.height);
  return canvas.transferToImageBitmap();
}

export function imageBitmapToTensorData(bitmap) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  const floatData = new Float32Array(3 * bitmap.width * bitmap.height);
  const plane = bitmap.width * bitmap.height;

  for (let i = 0; i < plane; i += 1) {
    const rgba = i * 4;
    floatData[i] = data[rgba] / 255;
    floatData[plane + i] = data[rgba + 1] / 255;
    floatData[plane * 2 + i] = data[rgba + 2] / 255;
  }

  return {
    data: floatData,
    dims: [1, 3, bitmap.height, bitmap.width],
  };
}
