# ONNX Model

```bash
python3 onnx_model/export_onnx.py \
  --height 256 \
  --width 256 \
  --refiner_scale 0.5 \
  --output onnx_model/model.onnx \
  --verify
```

Default export settings:

- input: `img0`, `img1`, both `1x3x256x256`
- output: `pred`, `1x3x256x256`
- refiner: `refiner_scale=0.5`
