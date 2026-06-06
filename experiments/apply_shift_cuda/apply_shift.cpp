#include <torch/extension.h>

torch::Tensor apply_shift_cuda_forward(torch::Tensor img, torch::Tensor weights_full, int64_t num_scales);

torch::Tensor apply_shift_forward(torch::Tensor img, torch::Tensor weights_full, int64_t num_scales) {
  TORCH_CHECK(img.is_cuda(), "img must be a CUDA tensor");
  TORCH_CHECK(weights_full.is_cuda(), "weights_full must be a CUDA tensor");
  TORCH_CHECK(img.dim() == 4, "img must have shape [B, C, H, W]");
  TORCH_CHECK(weights_full.dim() == 4, "weights_full must have shape [B, 9 * num_scales, H, W]");
  TORCH_CHECK(img.scalar_type() == weights_full.scalar_type(), "img and weights_full must have the same dtype");
  TORCH_CHECK(num_scales >= 1 && num_scales <= 6, "num_scales must be in [1, 6] for scales [1, 2, 4, 8, 16, 32]");
  TORCH_CHECK(weights_full.size(0) == img.size(0), "batch size mismatch");
  TORCH_CHECK(weights_full.size(1) == num_scales * 9, "weights_full channel count must be 9 * num_scales");
  TORCH_CHECK(weights_full.size(2) == img.size(2), "height mismatch");
  TORCH_CHECK(weights_full.size(3) == img.size(3), "width mismatch");

  return apply_shift_cuda_forward(img, weights_full, num_scales);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &apply_shift_forward, "Fused apply_shift forward (CUDA)");
}
