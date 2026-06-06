import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model import MainModel  # noqa: E402


class InterpolationExportWrapper(torch.nn.Module):
    def __init__(self, model, refiner_scale=0.5, skip_refiner=False):
        super().__init__()
        self.model = model
        self.refiner_scale = refiner_scale
        self.skip_refiner = skip_refiner

    def forward(self, img0, img1):
        pred, _ = self.model(
            img0,
            img1,
            refiner_scale=self.refiner_scale,
            skip_refiner=self.skip_refiner,
        )
        return pred


def parse_args():
    parser = argparse.ArgumentParser(description="Export the frame interpolation model to ONNX.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoint" / "model.pth"))
    parser.add_argument("--output", default=str(ROOT / "webgpu-deploy" / "frontend" / "public" / "models" / "model.onnx"))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--refiner_scale", type=float, default=0.5, choices=[1.0, 0.5, 0.25])
    parser.add_argument("--skip_refiner", action="store_true")
    parser.add_argument("--dynamic", action="store_true", help="Try dynamic height/width axes. Fixed shapes are safer for WebGPU.")
    parser.add_argument("--verify", action="store_true", help="Load the exported model with ONNX Runtime CPU once.")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = MainModel(scales=[1, 2, 4, 8, 16, 32]).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    wrapped = InterpolationExportWrapper(
        model,
        refiner_scale=args.refiner_scale,
        skip_refiner=args.skip_refiner,
    ).eval()

    img0 = torch.randn(1, 3, args.height, args.width, device=device)
    img1 = torch.randn(1, 3, args.height, args.width, device=device)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "img0": {2: "height", 3: "width"},
            "img1": {2: "height", 3: "width"},
            "pred": {2: "height", 3: "width"},
        }

    with torch.inference_mode():
        torch.onnx.export(
            wrapped,
            (img0, img1),
            str(output),
            input_names=["img0", "img1"],
            output_names=["pred"],
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    print(f"Exported ONNX model: {output}")
    print(f"Input shape: 1x3x{args.height}x{args.width}")
    print(f"Refiner scale: {args.refiner_scale}")
    print(f"Skip refiner: {args.skip_refiner}")

    if args.verify:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        sample = np.random.rand(1, 3, args.height, args.width).astype("float32")
        pred = session.run(None, {"img0": sample, "img1": sample})[0]
        print(f"Verified with ONNX Runtime CPU: {pred.shape} {pred.dtype}")


if __name__ == "__main__":
    main()
