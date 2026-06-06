# Frame Interpolation via Multi-Directional Discrete Shift Modeling

This repository contains the implementation for the final project of the course
**Deep Learning (2526I_AIT3001#_2)** at **University of Engineering and Technology (UET)**.

The project is based on a lightweight video frame interpolation method that models motion using **multi-directional discrete shifts** instead of dense optical flow, combined with a **residual U-Net refinement network** to reduce interpolation artifacts.

---

## Paper

**Frame Interpolation via Multi-Directional Discrete Shift Modeling with Residual Learning U-Net Refinement**
Author: *Huy Duc Vu*
This work is submitted as a **final-term assignment** (5-page paper).

---

## Method Overview

- Motion is approximated using **discrete spatial shifts** at multiple directions and scales
- Bidirectional warped frames are fused using a **learned visibility mask**
- A **context-aware U-Net** predicts residual corrections to refine the interpolated frame
- Total model size: **~3.1M parameters**

---

## Video Inference

Run inference on a video with frame rate interpolation:

```bash
python Video_Inference.py \
  --input input_path \
  --output output_path \
  --model checkpoint/model.pth \
  --fps_multiplier 2   # or 4
```

By default, video inference runs the residual U-Net refiner at `--refiner_scale 0.5`
for a faster no-retraining speed/quality tradeoff. Use `--refiner_scale 1.0` for
the original full-resolution refiner.

Fast approximate refiner modes can be tested without retraining:

```bash
# Run the residual U-Net refiner at half resolution, then upsample the residual.
python Video_Inference.py -i input_path -o output_path -m checkpoint/model.pth --refiner_scale 0.5

# Skip the residual U-Net refiner and output the coarse merged frame.
python Video_Inference.py -i input_path -o output_path -m checkpoint/model.pth --skip_refiner
```

FFmpeg output uses a raw video pipe by default, avoiding temporary PNG frame dumps.
Use `--no_ffmpeg` to fall back to OpenCV `VideoWriter`.

## Runtime Benchmark

Measure model-only inference runtime on CUDA. The benchmark reports both the original
PyTorch `apply_shift` path and the fused CUDA `apply_shift` path when CUDA is available:

```bash
python benchmark_runtime.py \
  --device cuda \
  --model checkpoint/model.pth \
  --height 256 \
  --width 256 \
  --iters 100
```

Use `--batch 1` for the default video-inference configuration. Use
`--no_cuda_apply_shift` to measure only the original PyTorch path.

Video Demo (Comparison)

Original Video (6 FPS) - On One Puch Man

https://github.com/user-attachments/assets/4b13bd06-0757-44c1-aabf-ba682f54e1a7



Interpolated Video (96 FPS) - On One Puch Man


https://github.com/user-attachments/assets/8073f3aa-6ff9-48ad-a7bb-818471eab434

Original Video (6 FPS) - On Red Dead Redemption 2



https://github.com/user-attachments/assets/29263556-96f0-4dad-8a6b-92566fd71b4c



Interpolated Video (96 FPS) - On Red Dead Redemption 2



https://github.com/user-attachments/assets/fc6b7e48-8e86-4955-8b17-53f3de7bc419



The interpolated result demonstrates smoother motion and improved temporal continuity compared to the low-FPS input.

Notes

This project is for academic and educational purposes

Performance is competitive with existing methods while maintaining a compact model size

Current implementation prioritizes clarity over runtime optimization

References

AdaCoF, RIFE, U-Net, Vimeo90K, UCF101, VFIformer
