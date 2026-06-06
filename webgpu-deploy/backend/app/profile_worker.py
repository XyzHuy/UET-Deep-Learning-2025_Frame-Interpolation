import contextlib
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Video_Inference import VideoInterpolator  # noqa: E402


def main():
    payload = json.loads(sys.argv[1])
    interpolator = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            interpolator = VideoInterpolator(
                model_path=payload["model_path"],
                device=payload["device"],
                refiner_scale=payload["refiner_scale"],
                skip_refiner=payload["skip_refiner"],
            )
            interpolator.configure_runtime(
                tile_size=payload["tile_size"],
                refiner_scale=payload["refiner_scale"],
                skip_refiner=payload["skip_refiner"],
            )
            profile = interpolator.profile_batch_sizes(
                payload["height"],
                payload["width"],
                payload["batch_sizes"],
            )
            interpolator.clear_memory_cache()

        print(json.dumps(profile))
    finally:
        del interpolator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
