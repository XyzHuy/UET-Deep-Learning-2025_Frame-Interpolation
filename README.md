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
python Video_inference.py \
  --input input_path \
  --output output_path \
  --model checkpoint/model.pth \
  --fps_multiplier 2   # or 4
```

Video Demo (Comparison)

Original Video (6 FPS)
video_demo/one_punch_6fps.mp4

Interpolated Video (96 FPS)
video_demo/one_punch_96fps_interpolated.mp4

The interpolated result demonstrates smoother motion and improved temporal continuity compared to the low-FPS input.

Notes

This project is for academic and educational purposes

Performance is competitive with existing methods while maintaining a compact model size

Current implementation prioritizes clarity over runtime optimization

References

AdaCoF, RIFE, U-Net, Vimeo90K, UCF101, VFIformer
