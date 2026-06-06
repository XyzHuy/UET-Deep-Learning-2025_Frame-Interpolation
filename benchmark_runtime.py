import argparse
import time
from contextlib import nullcontext

import torch

from Model import MainModel
from cuda_apply_shift import patch_model_apply_shift


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device, use_fp16):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16)


def timed_forward(model, img0, img1, device, use_fp16, iters, model_kwargs):
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode(), autocast_context(device, use_fp16):
            starter.record()
            for _ in range(iters):
                model(img0, img1, **model_kwargs)
            ender.record()
        torch.cuda.synchronize(device)
        return starter.elapsed_time(ender) / iters

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            model(img0, img1, **model_kwargs)
    return (time.perf_counter() - start) * 1000.0 / iters


def add_timing_wrapper(owner, attr_name, label, times, counts, device):
    original = getattr(owner, attr_name)

    def wrapped(*args, **kwargs):
        sync_if_cuda(device)
        start = time.perf_counter()
        out = original(*args, **kwargs)
        sync_if_cuda(device)
        times[label] = times.get(label, 0.0) + time.perf_counter() - start
        counts[label] = counts.get(label, 0) + 1
        return out

    setattr(owner, attr_name, wrapped)


def benchmark_breakdown(model, img0, img1, device, use_fp16, iters, model_kwargs):
    times = {}
    counts = {}

    add_timing_wrapper(model.context_extractor_student, "forward", "context_extractor_student", times, counts, device)
    add_timing_wrapper(model.flow_encoder_student, "forward", "flow_encoder_student", times, counts, device)
    add_timing_wrapper(model.shift_flow_student, "forward", "shift_flow_student", times, counts, device)
    add_timing_wrapper(model.refiner, "forward", "refiner", times, counts, device)
    add_timing_wrapper(model, "apply_shift", "apply_shift", times, counts, device)

    with torch.inference_mode(), autocast_context(device, use_fp16):
        for _ in range(iters):
            model(img0, img1, **model_kwargs)

    sync_if_cuda(device)
    rows = []
    for key, value in sorted(times.items(), key=lambda item: item[1], reverse=True):
        rows.append((key, value * 1000.0 / iters, counts[key]))
    return rows


def load_model(args, device, use_fp16):
    model = MainModel(scales=[1, 2, 4, 8, 16, 32]).to(device).eval()
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if use_fp16:
        model.half()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def make_inputs(args, device, use_fp16):
    img0 = torch.rand(args.batch, 3, args.height, args.width, device=device)
    img1 = torch.rand(args.batch, 3, args.height, args.width, device=device)
    if use_fp16:
        img0 = img0.half()
        img1 = img1.half()
    if device.type == "cuda":
        img0 = img0.to(memory_format=torch.channels_last)
        img1 = img1.to(memory_format=torch.channels_last)
    return img0, img1


def warmup(model, img0, img1, device, use_fp16, warmup_iters, model_kwargs):
    with torch.inference_mode(), autocast_context(device, use_fp16):
        for _ in range(warmup_iters):
            model(img0, img1, **model_kwargs)
    sync_if_cuda(device)


def compare_outputs(base_model, cuda_model, img0, img1, device, use_fp16, model_kwargs):
    with torch.inference_mode(), autocast_context(device, use_fp16):
        base_pred = base_model(img0, img1, **model_kwargs)[0]
        cuda_pred = cuda_model(img0, img1, **model_kwargs)[0]
    sync_if_cuda(device)
    diff = (base_pred.float() - cuda_pred.float()).abs()
    return diff.max().item(), diff.mean().item()


def print_comparison(base_ms, cuda_ms, batch):
    print("\nmodel_runtime_comparison:")
    print(f"{'variant':22s} {'ms/forward':>12s} {'pairs/s':>12s} {'speedup':>10s}")
    print(f"{'pytorch_apply_shift':22s} {base_ms:12.3f} {batch * 1000.0 / base_ms:12.2f} {1.0:10.2f}")
    if cuda_ms is None:
        print(f"{'cuda_apply_shift':22s} {'unavailable':>12s} {'-':>12s} {'-':>10s}")
    else:
        print(f"{'cuda_apply_shift':22s} {cuda_ms:12.3f} {batch * 1000.0 / cuda_ms:12.2f} {base_ms / cuda_ms:10.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MainModel inference runtime.")
    parser.add_argument("--model", default="checkpoint/model.pth", help="Checkpoint path")
    parser.add_argument("--height", type=int, default=256, help="Input height")
    parser.add_argument("--width", type=int, default=256, help="Input width")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for synthetic pair inference")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations")
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"], help="Device")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 autocast/model weights")
    parser.add_argument("--refiner_scale", type=float, default=0.5, choices=[1.0, 0.5, 0.25],
                        help="Run refiner at lower resolution and upsample its residual")
    parser.add_argument("--skip_refiner", action="store_true", help="Skip the residual U-Net refiner")
    parser.add_argument("--no_cuda_apply_shift", action="store_true",
                        help="Only benchmark the pure PyTorch apply_shift path")
    parser.add_argument("--no_breakdown", action="store_true", help="Skip module timing breakdown")
    parser.add_argument("--atol", type=float, default=3e-3, help="Correctness max-abs warning threshold")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    use_fp16 = device.type == "cuda" and not args.fp32

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model_kwargs = {
        "refiner_scale": args.refiner_scale,
        "skip_refiner": args.skip_refiner,
    }
    img0, img1 = make_inputs(args, device, use_fp16)

    base_model = load_model(args, device, use_fp16)
    warmup(base_model, img0, img1, device, use_fp16, args.warmup, model_kwargs)
    base_ms = timed_forward(base_model, img0, img1, device, use_fp16, args.iters, model_kwargs)

    cuda_ms = None
    max_abs = None
    mean_abs = None
    cuda_available = device.type == "cuda" and not args.no_cuda_apply_shift
    if cuda_available:
        cuda_model = load_model(args, device, use_fp16)
        try:
            patch_model_apply_shift(cuda_model)
            max_abs, mean_abs = compare_outputs(base_model, cuda_model, img0, img1, device, use_fp16, model_kwargs)
            warmup(cuda_model, img0, img1, device, use_fp16, args.warmup, model_kwargs)
            cuda_ms = timed_forward(cuda_model, img0, img1, device, use_fp16, args.iters, model_kwargs)
        except Exception as error:
            print(f"CUDA apply_shift benchmark unavailable: {error}")
            cuda_ms = None

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"shape: {args.batch}x3x{args.height}x{args.width}")
    print(f"fp16: {use_fp16}")
    print(f"refiner_scale: {args.refiner_scale}")
    print(f"skip_refiner: {args.skip_refiner}")
    if max_abs is not None:
        status = "OK" if max_abs <= args.atol else "WARN"
        print(f"cuda_apply_shift_diff: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} {status}")

    print_comparison(base_ms, cuda_ms, args.batch)

    if not args.no_breakdown:
        print("\nbaseline_breakdown_ms_per_forward:")
        for label, ms_per_forward, calls in benchmark_breakdown(
            base_model,
            img0,
            img1,
            device,
            use_fp16,
            args.iters,
            model_kwargs,
        ):
            print(f"  {label}: {ms_per_forward:.3f} ms ({calls} calls)")


if __name__ == "__main__":
    main()
