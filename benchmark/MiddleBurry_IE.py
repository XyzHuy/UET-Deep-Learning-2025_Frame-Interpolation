import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import numpy as np
from Model import MainModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MainModel(scales=[1,2,4,8,16,32]).to(device)
checkpoint = torch.load(
    'checkpoint/model.pth',
    map_location=device
)
# Load model weights
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

names = [
    'Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3',
    'Hydrangea', 'MiniCooper', 'RubberWhale',
    'Urban2', 'Urban3', 'Venus', 'Walking'
]

IE_list = []

for name in names:
    i0 = cv2.imread(f'middlebury-frameinterpolation/other-gt-interp/{name}/frame10.png').transpose(2, 0, 1) / 255.
    i1 = cv2.imread(f'middlebury-frameinterpolation/other-gt-interp/{name}/frame11.png').transpose(2, 0, 1) / 255.
    gt = cv2.imread(f'middlebury-frameinterpolation/other-gt-interp/{name}/frame10i11.png')

    h, w = i0.shape[1], i0.shape[2]

    imgs = torch.zeros([1, 6, 480, 640]).to(device)
    imgs[:, :3, :h, :w] = torch.from_numpy(i0).unsqueeze(0).float().to(device)
    imgs[:, 3:, :h, :w] = torch.from_numpy(i1).unsqueeze(0).float().to(device)

    I0 = imgs[:, :3]
    I2 = imgs[:, 3:]

    with torch.no_grad():
        mid, _ = model(I0, I2)    # mid: [1,3,480,640]

    out = mid[0, :, :h, :w].detach().cpu().numpy().transpose(1, 2, 0)
    out = np.round(out * 255)
    
    IE = np.abs(out - gt.astype(np.float32)).mean()
    IE_list.append(IE)

    print(np.mean(IE_list))
