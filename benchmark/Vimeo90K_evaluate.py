import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Model import MainModel
from benchmark.RIFE_evaluated_formula import ssim_matlab


model = MainModel(scales=[1,2,4,8,16,32]).to(device)
checkpoint = torch.load(
    'checkpoint/model.pth',
    map_location=device
)
# Load model weights
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()
path = 'vimeo_triplet/sequences/'
f = open('vimeo_triplet/tri_testlist.txt', 'r')
psnr_list = []
ssim_list = []
for i in f:
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    print(path  + name + '/im1.png')
    I0 = cv2.imread(path  + name + '/im1.png')
    I1 = cv2.imread(path  + name + '/im2.png')
    I2 = cv2.imread(path  + name + '/im3.png')
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    with torch.no_grad():
        mid, _ = model(I0, I2)
    
    gt = torch.tensor(I1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255.
    pred = torch.round(mid * 255) / 255.

    ssim = ssim_matlab(gt, pred).item()
    
    # PSNR
    pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    gt_np = I1 / 255.
    
    psnr = -10 * math.log10(((gt_np - pred_np) ** 2).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))