import argparse
import time

import torch

from Model import MainModel


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed_forward(model, img0, img1, device, use_fp16, iters):
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            starter.record()
            for _ in range(iters):
                model(img0, img1)
            ender.record()
        torch.cuda.synchronize()
        return starter.elapsed_time(ender) / iters

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            model(img0, img1)
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


def benchmark_breakdown(model, img0, img1, device, use_fp16, iters):
    times = {}
    counts = {}

    add_timing_wrapper(model.context_extractor_student, "forward", "context_extractor_student", times, counts, device)
    add_timing_wrapper(model.flow_encoder_student, "forward", "flow_encoder_student", times, counts, device)
    add_timing_wrapper(model.shift_flow_student, "forward", "shift_flow_student", times, counts, device)
    add_timing_wrapper(model.refiner, "forward", "refiner", times, counts, device)
    add_timing_wrapper(model, "apply_shift", "apply_shift", times, counts, device)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
        for _ in range(iters):
            model(img0, img1)

    sync_if_cuda(device)
    rows = []
    for key, value in sorted(times.items(), key=lambda item: item[1], reverse=True):
        rows.append((key, value * 1000.0 / iters, counts[key]))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark MainModel inference runtime.")
    parser.add_argument("--model", default="checkpoint/model.pth", help="Checkpoint path")
    parser.add_argument("--height", type=int, default=256, help="Input height")
    parser.add_argument("--width", type=int, default=256, help="Input width")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 autocast/model weights")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for total-runtime measurement")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not args.fp32

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = MainModel(scales=[1, 2, 4, 8, 16, 32]).to(device).eval()
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if use_fp16:
        model.half()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    img0 = torch.rand(1, 3, args.height, args.width, device=device)
    img1 = torch.rand(1, 3, args.height, args.width, device=device)
    if use_fp16:
        img0 = img0.half()
        img1 = img1.half()
    if device.type == "cuda":
        img0 = img0.to(memory_format=torch.channels_last)
        img1 = img1.to(memory_format=torch.channels_last)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
        for _ in range(args.warmup):
            model(img0, img1)
    sync_if_cuda(device)

    measured_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    avg_ms = timed_forward(measured_model, img0, img1, device, use_fp16, args.iters)

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"shape: 1x3x{args.height}x{args.width}")
    print(f"fp16: {use_fp16}")
    print(f"compiled: {args.compile}")
    print(f"avg_forward_ms: {avg_ms:.3f}")

    if not args.compile:
        print("\nbreakdown_ms_per_forward:")
        for label, ms_per_forward, calls in benchmark_breakdown(model, img0, img1, device, use_fp16, args.iters):
            print(f"  {label}: {ms_per_forward:.3f} ms ({calls} calls)")


if __name__ == "__main__":
    main()
