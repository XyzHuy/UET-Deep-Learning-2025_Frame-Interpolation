import os
from pathlib import Path

import torch
import torch.nn.functional as F


_EXTENSION = None
_LOAD_ERROR = None


def load_apply_shift_extension(verbose=False):
    global _EXTENSION, _LOAD_ERROR
    from torch.utils.cpp_extension import load

    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "experiments" / "apply_shift_cuda" / "apply_shift.cpp"),
        str(root / "experiments" / "apply_shift_cuda" / "apply_shift_kernel.cu"),
    ]

    try:
        _EXTENSION = load(
            name="vfi_apply_shift_cuda",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=verbose,
        )
    except Exception as error:
        _LOAD_ERROR = error
        raise

    return _EXTENSION


def patch_model_apply_shift(model, verbose=False):
    if not torch.cuda.is_available():
        return False

    ext = load_apply_shift_extension(verbose=verbose)

    def cuda_apply_shift(img, weights_list, weights_full=None):
        if weights_full is None:
            weights_full = F.interpolate(
                torch.cat(weights_list, dim=1),
                scale_factor=4,
                mode="bilinear",
                align_corners=False,
            )

        if weights_full.dtype != img.dtype:
            weights_full = weights_full.to(dtype=img.dtype)
        if img.is_contiguous(memory_format=torch.channels_last):
            weights_full = weights_full.contiguous(memory_format=torch.channels_last)

        return ext.forward(img, weights_full, len(model.scales))

    model.apply_shift = cuda_apply_shift
    return True


def cuda_apply_shift_enabled_by_default():
    value = os.getenv("USE_CUDA_APPLY_SHIFT", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}
