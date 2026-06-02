import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import subprocess
import math
from Model import MainModel

            
class VideoInterpolator:
    def __init__(
        self,
        model_path,
        device='auto',
        tile_size=None,
        tile_overlap=32,
        use_fp16=True,
        use_cpu_bf16=False,
        channels_last=True,
        refiner_scale=1.0,
        skip_refiner=False
    ):
        self.device = self.resolve_device(device)
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.use_cpu_bf16 = use_cpu_bf16 and self.device.type == 'cpu'
        self.channels_last = channels_last and self.device.type in ('cuda', 'cpu')

        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.mkldnn.enabled = True
        
        # Load model
        self.model = MainModel(scales=[1,2,4,8,16,32]).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.eval()

        if self.use_fp16:
            self.model.half()

        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.refiner_scale = refiner_scale
        self.skip_refiner = skip_refiner
        
        print(f"Model loaded on {self.device}")
        print(f"FP16: {'Enabled' if self.use_fp16 else 'Disabled'}")
        print(f"CPU BF16: {'Enabled' if self.use_cpu_bf16 else 'Disabled'}")
        print(f"Channels last: {'Enabled' if self.channels_last else 'Disabled'}")
        print(f"Refiner: {'Skipped' if self.skip_refiner else f'scale x{self.refiner_scale:g}'}")
        print(f"Tile processing: {'Auto' if tile_size is None else f'{tile_size}x{tile_size}'}")

    @staticmethod
    def resolve_device(device):
        device = device.lower()
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cuda':
            if not torch.cuda.is_available():
                print("CUDA requested but not available. Falling back to CPU.")
                return torch.device('cpu')
            return torch.device('cuda')
        if device == 'cpu':
            return torch.device('cpu')
        raise ValueError(f"Unsupported device: {device}. Use auto, cuda, or cpu.")
    
    def auto_tile_size(self, height, width):
        pixels = height * width
        
        if pixels <= 1280 * 720:  
            return None  # Full frame
        elif pixels <= 1920 * 1080:  # 1080p
            return 640 if self.use_fp16 else 512
        else:  
            return 512 if self.use_fp16 else 384
    
    def pad_to_multiple(self, img, multiple=32):
        h, w = img.shape[2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        
        return img, (h, w)
    
    def interpolate_frame(self, frame0, frame1):
        with torch.inference_mode():
            # Convert to tensor
            img0 = torch.from_numpy(frame0).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            
            img0 = img0.to(self.device, non_blocking=self.device.type == 'cuda')
            img1 = img1.to(self.device, non_blocking=self.device.type == 'cuda')
            
            # Convert to FP16 if enabled
            if self.use_fp16:
                img0 = img0.half()
                img1 = img1.half()

            # Pad to multiple of 32
            img0, orig_size = self.pad_to_multiple(img0)
            img1, _ = self.pad_to_multiple(img1)

            if self.channels_last:
                img0 = img0.contiguous(memory_format=torch.channels_last)
                img1 = img1.contiguous(memory_format=torch.channels_last)
            
            # Auto-detect tile size 
            tile_size = self.tile_size
            if tile_size is None:
                tile_size = self.auto_tile_size(orig_size[0], orig_size[1])
            
            # Tile-based processing for high-res
            autocast_enabled = self.use_fp16 or self.use_cpu_bf16
            autocast_dtype = torch.float16 if self.device.type == 'cuda' else torch.bfloat16
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                if tile_size and (img0.shape[2] > tile_size or img0.shape[3] > tile_size):
                    pred = self.tile_inference(img0, img1, tile_size)
                else:
                    pred = self.model(
                        img0,
                        img1,
                        refiner_scale=self.refiner_scale,
                        skip_refiner=self.skip_refiner
                    )[0]
            
            # Convert back to FP32
            if self.use_fp16 or self.use_cpu_bf16:
                pred = pred.float()
            
            # Crop back to original size
            pred = pred[:, :, :orig_size[0], :orig_size[1]]
            
            # Convert back to numpy
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_np = (pred_np * 255).clip(0, 255).astype(np.uint8)
            
            return pred_np
    
    def tile_inference(self, img0, img1, tile_size):
        """
        Tile-based inference với smart batching
        """
        b, c, h, w = img0.shape
        overlap = self.tile_overlap
        stride = tile_size - overlap
        
        # Calculate number of tiles
        n_tiles_h = math.ceil((h - overlap) / stride)
        n_tiles_w = math.ceil((w - overlap) / stride)
        
        # Output canvas
        output = torch.zeros_like(img0)
        weight_map = torch.zeros_like(img0)
        
        # Create gaussian blend weights
        blend = torch.ones(1, 1, tile_size, tile_size, device=self.device, dtype=img0.dtype)
        
        # Smooth fade at edges
        fade = overlap // 2
        for i in range(fade):
            alpha = (i + 1) / fade
            blend[:, :, i, :] *= alpha
            blend[:, :, -i-1, :] *= alpha
            blend[:, :, :, i] *= alpha
            blend[:, :, :, -i-1] *= alpha
        
        # Process tiles
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile coordinates
                y_start = min(i * stride, h - tile_size)
                x_start = min(j * stride, w - tile_size)
                y_end = y_start + tile_size
                x_end = x_start + tile_size
                
                # Extract tile
                tile0 = img0[:, :, y_start:y_end, x_start:x_end]
                tile1 = img1[:, :, y_start:y_end, x_start:x_end]
                
                # Inference
                pred_tile = self.model(
                    tile0,
                    tile1,
                    refiner_scale=self.refiner_scale,
                    skip_refiner=self.skip_refiner
                )[0]
                
                # Accumulate with blending weights
                output[:, :, y_start:y_end, x_start:x_end] += pred_tile * blend
                weight_map[:, :, y_start:y_end, x_start:x_end] += blend
        
        # Normalize by weight map
        output = output / (weight_map + 1e-8)
        
        return output
    
    def interpolate_video(self, input_path, output_path, fps_multiplier=2, 
                          output_fps=None, use_ffmpeg=True, crf=18):
        """
        Interpolate toàn bộ video
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            fps_multiplier: 2 for x2 FPS, 4 for x4 FPS
            output_fps: Custom output FPS (if None, auto calculate)
            use_ffmpeg: Use ffmpeg for encoding (faster & smaller file)
            crf: FFmpeg CRF value (18=high quality, 23=default, 28=lower quality)
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        new_fps = output_fps if output_fps else orig_fps * fps_multiplier
        
        # Auto tile size info
        auto_tile = self.auto_tile_size(height, width)
        tile_info = f"{auto_tile}x{auto_tile}" if auto_tile else "Full Frame"
        
        print(f"\n Input: {width}x{height} @ {orig_fps:.2f} FPS ({total_frames} frames)")
        print(f" Output: {width}x{height} @ {new_fps:.2f} FPS")
        print(f" Multiplier: x{fps_multiplier}")
        print(f" Processing mode: {tile_info}")
        print(f" Estimated output frames: {total_frames * fps_multiplier}")
        
        if use_ffmpeg:
            output_path = str(output_path)
            cmd = [
                'ffmpeg', '-y',
                '-loglevel', 'error',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',
                '-r', str(new_fps),
                '-i', '-',
                '-an',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', str(crf),
                '-pix_fmt', 'yuv420p',
                output_path
            ]

            print(f"\n Encoding with FFmpeg raw pipe (CRF={crf})...")
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            def write_rgb(frame_rgb):
                process.stdin.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).tobytes())

            pbar = None
            try:
                ret, prev_frame = cap.read()
                
                if not ret:
                    raise ValueError("Cannot read first frame")
                
                prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
                
                pbar = tqdm(total=total_frames, desc="Processing", unit="frame", ncols=100)
                
                while True:
                    ret, curr_frame = cap.read()
                    
                    if not ret:
                        write_rgb(prev_frame)
                        break
                    
                    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                    
                    write_rgb(prev_frame)
                    
                    # Generate intermediate frames
                    if fps_multiplier == 2:
                        # x2: Generate 1 intermediate frame
                        inter_frame = self.interpolate_frame(prev_frame, curr_frame)
                        write_rgb(inter_frame)
                        
                    elif fps_multiplier == 4:
                        # x4: Generate 3 intermediate frames using recursion
                        # 1. Tạo frame giữa (0.5)
                        mid_frame = self.interpolate_frame(prev_frame, curr_frame)
                        
                            # 2. Tạo frame 0.25 (Giữa prev và mid)
                        first_quarter = self.interpolate_frame(prev_frame, mid_frame)
                        
                        # 3. Tạo frame 0.75 (Giữa mid và curr)
                        last_quarter = self.interpolate_frame(mid_frame, curr_frame)
                        
                        # Lưu theo thứ tự: 0.25 -> 0.5 -> 0.75
                        write_rgb(first_quarter)
                        write_rgb(mid_frame)
                        write_rgb(last_quarter)
                    
                    prev_frame = curr_frame
                    pbar.update(1)

                process.stdin.close()
                stderr = process.stderr.read().decode('utf-8', errors='replace')
                return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(f"FFmpeg failed with code {return_code}:\n{stderr}")

                # Get output file size
                output_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"Video saved: {output_path}")
                print(f"Output size: {output_size:.2f} MB")
                
            finally:
                if pbar is not None:
                    pbar.close()
                cap.release()
                if process.poll() is None:
                    process.kill()
        
        else:
            # Direct cv2 VideoWriter (simpler but larger file)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, new_fps, (width, height))
            
            ret, prev_frame = cap.read()
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
            
            pbar = tqdm(total=total_frames, desc="Processing", unit="frame", ncols=100)
            
            while True:
                ret, curr_frame = cap.read()
                
                if not ret:
                    out.write(cv2.cvtColor(prev_frame, cv2.COLOR_RGB2BGR))
                    break
                
                curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                
                # Write original frame
                out.write(cv2.cvtColor(prev_frame, cv2.COLOR_RGB2BGR))
                
                # Generate and write intermediate frames
                if fps_multiplier == 2:
                    inter = self.interpolate_frame(prev_frame, curr_frame)
                    out.write(cv2.cvtColor(inter, cv2.COLOR_RGB2BGR))
                    
                elif fps_multiplier == 4:
                    mid_frame = self.interpolate_frame(prev_frame, curr_frame)        # t=0.5
                    first_quarter = self.interpolate_frame(prev_frame, mid_frame)     # t=0.25
                    last_quarter = self.interpolate_frame(mid_frame, curr_frame)      # t=0.75
                    
                    # Ghi file
                    out.write(cv2.cvtColor(first_quarter, cv2.COLOR_RGB2BGR))
                    out.write(cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR))
                    out.write(cv2.cvtColor(last_quarter, cv2.COLOR_RGB2BGR))
                
                prev_frame = curr_frame
                pbar.update(1)
            
            pbar.close()
            out.release()
            cap.release()
            
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n Video saved: {output_path}")
            print(f" Output size: {output_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Video Frame Interpolation')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input video path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output video path')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--fps_multiplier', '-f', type=int, default=2, choices=[2, 4],
                        help='FPS multiplier (2 or 4)')
    parser.add_argument('--output_fps', type=float, default=None,
                        help='Custom output FPS (overrides multiplier)')
    parser.add_argument('--tile_size', type=int, default=None,
                        help='Manual tile size (None=auto detect)') 
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='Tile overlap size')
    parser.add_argument('--device', type=str, default='cuda', choices=['auto', 'cuda', 'cpu'],
                        help='Device mode: auto, cuda, or cpu')
    parser.add_argument('--cpu', action='store_true',
                        help='Shortcut for --device cpu')
    parser.add_argument('--cpu_threads', type=int, default=None,
                        help='Number of CPU threads for PyTorch inference')
    parser.add_argument('--cpu_interop_threads', type=int, default=None,
                        help='Number of PyTorch inter-op CPU threads')
    parser.add_argument('--cpu_bf16', action='store_true',
                        help='Use CPU bfloat16 autocast. Test this on your CPU; it can be faster only on CPUs with good BF16 support')
    parser.add_argument('--no_channels_last', action='store_true',
                        help='Disable channels-last memory format for inference')
    parser.add_argument('--refiner_scale', type=float, default=0.5, choices=[1.0, 0.5, 0.25],
                        help='Run refiner at lower resolution and upsample its residual. 1.0 keeps original behavior')
    parser.add_argument('--skip_refiner', action='store_true',
                        help='Skip residual U-Net refiner and output the coarse merged frame')
    parser.add_argument('--no_fp16', action='store_true',
                        help='Disable FP16 (use FP32)')
    parser.add_argument('--no_ffmpeg', action='store_true',
                        help='Disable ffmpeg encoding (use cv2 instead)')
    parser.add_argument('--crf', type=int, default=18,
                        help='FFmpeg CRF quality (18=high, 23=default, 28=lower)')
    
    args = parser.parse_args()

    if args.cpu:
        args.device = 'cpu'

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    if args.cpu_interop_threads is not None:
        torch.set_num_interop_threads(args.cpu_interop_threads)
    
    # Initialize interpolator
    interpolator = VideoInterpolator(
        model_path=args.model,
        device=args.device,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        use_fp16=not args.no_fp16,
        use_cpu_bf16=args.cpu_bf16,
        channels_last=not args.no_channels_last,
        refiner_scale=args.refiner_scale,
        skip_refiner=args.skip_refiner
    )
    
    # Process video
    interpolator.interpolate_video(
        input_path=args.input,
        output_path=args.output,
        fps_multiplier=args.fps_multiplier,
        output_fps=args.output_fps,
        use_ffmpeg=not args.no_ffmpeg,
        crf=args.crf
    )


if __name__ == '__main__':
    main()
