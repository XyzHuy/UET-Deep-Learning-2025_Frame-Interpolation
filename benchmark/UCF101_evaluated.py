import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import numpy as np
from Model import MainModel
from benchmark.RIFE_evaluated_formula import ssim_matlab


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MainModel(scales=[1,2,4,8,16,32]).to(device)
checkpoint = torch.load(
    'checkpoint/model.pth',
    map_location=device
)
# Load model weights
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

path = 'ucf101_interp_ours/'
dirs = os.listdir(path)

psnr_list = []
ssim_list = []

print(len(dirs))

for d in dirs:
    img0_path = path + d + '/frame_00.png'
    img1_path = path + d + '/frame_02.png'
    gt_path   = path + d + '/frame_01_gt.png'

    I0 = torch.tensor(
        cv2.imread(img0_path).transpose(2, 0, 1)
    ).to(device).float().unsqueeze(0) / 255.

    I2 = torch.tensor(
        cv2.imread(img1_path).transpose(2, 0, 1)
    ).to(device).float().unsqueeze(0) / 255.

    gt = torch.tensor(
        cv2.imread(gt_path).transpose(2, 0, 1)
    ).to(device).float().unsqueeze(0) / 255.

    with torch.no_grad():
        mid, _ = model(I0, I2)     # mid: [1,3,H,W]

    pred_ssim = torch.round(mid * 255) / 255.   # [1,3,H,W]
    ssim = ssim_matlab(gt, pred_ssim).item()

    pred_np = pred_ssim.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    gt_np = gt.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    psnr = -10 * math.log10(((gt_np - pred_np) ** 2).mean())

    psnr_list.append(psnr)
    ssim_list.append(ssim)

    print(f"Avg PSNR: {np.mean(psnr_list):.4f}  SSIM: {np.mean(ssim_list):.4f}")
