#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kDirections = 9;

__device__ __forceinline__ void direction_from_index(int direction, int scale, int& dy, int& dx) {
  switch (direction) {
    case 0:
      dy = 0;
      dx = 0;
      break;
    case 1:
      dy = -scale;
      dx = 0;
      break;
    case 2:
      dy = scale;
      dx = 0;
      break;
    case 3:
      dy = 0;
      dx = -scale;
      break;
    case 4:
      dy = 0;
      dx = scale;
      break;
    case 5:
      dy = -scale;
      dx = -scale;
      break;
    case 6:
      dy = -scale;
      dx = scale;
      break;
    case 7:
      dy = scale;
      dx = -scale;
      break;
    default:
      dy = scale;
      dx = scale;
      break;
  }
}

template <typename scalar_t>
__global__ void apply_shift_kernel(
    const scalar_t* __restrict__ img,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ out,
    int64_t total,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t num_scales,
    int64_t img_s0,
    int64_t img_s1,
    int64_t img_s2,
    int64_t img_s3,
    int64_t w_s0,
    int64_t w_s1,
    int64_t w_s2,
    int64_t w_s3,
    int64_t out_s0,
    int64_t out_s1,
    int64_t out_s2,
    int64_t out_s3) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  const int64_t x = linear % width;
  const int64_t y = (linear / width) % height;
  const int64_t c = (linear / (width * height)) % channels;
  const int64_t b = linear / (width * height * channels);

  float acc = 0.0f;

  for (int64_t scale_index = 0; scale_index < num_scales; ++scale_index) {
    const int scale = 1 << scale_index;

    for (int direction = 0; direction < kDirections; ++direction) {
      int dy = 0;
      int dx = 0;
      direction_from_index(direction, scale, dy, dx);

      const int64_t src_y = y - dy;
      const int64_t src_x = x - dx;
      if (src_y < 0 || src_y >= height || src_x < 0 || src_x >= width) {
        continue;
      }

      const int64_t weight_channel = scale_index * kDirections + direction;
      const int64_t img_offset = b * img_s0 + c * img_s1 + src_y * img_s2 + src_x * img_s3;
      const int64_t weight_offset = b * w_s0 + weight_channel * w_s1 + y * w_s2 + x * w_s3;
      acc += static_cast<float>(img[img_offset]) * static_cast<float>(weights[weight_offset]);
    }
  }

  const int64_t out_offset = b * out_s0 + c * out_s1 + y * out_s2 + x * out_s3;
  out[out_offset] = static_cast<scalar_t>(acc / static_cast<float>(num_scales));
}

}  // namespace

torch::Tensor apply_shift_cuda_forward(torch::Tensor img, torch::Tensor weights_full, int64_t num_scales) {
  const c10::cuda::CUDAGuard device_guard(img.device());
  auto out = torch::empty_strided(img.sizes(), img.strides(), img.options());

  const int64_t batch = img.size(0);
  const int64_t channels = img.size(1);
  const int64_t height = img.size(2);
  const int64_t width = img.size(3);
  const int64_t total = batch * channels * height * width;

  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(img.scalar_type(), "apply_shift_cuda_forward", [&] {
    apply_shift_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        img.data_ptr<scalar_t>(),
        weights_full.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        total,
        batch,
        channels,
        height,
        width,
        num_scales,
        img.stride(0),
        img.stride(1),
        img.stride(2),
        img.stride(3),
        weights_full.stride(0),
        weights_full.stride(1),
        weights_full.stride(2),
        weights_full.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3));
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
