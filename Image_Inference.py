import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from Model import MainModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MainModel(scales=[1,2,4,8,16,32]).to(device)

checkpoint = torch.load(
    'checkpoint/model.pth',
    map_location=device
)

# Load model weights
model.load_state_dict(checkpoint['model_state_dict'], strict=True)

model.eval()

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return T.ToTensor()(img).unsqueeze(0)  # [1,3,H,W]



img0 = load_image("image_demo/0.webp").to(device)
img1 = load_image("image_demo/2.webp").to(device)

with torch.no_grad():
    pred, aux = model(img0, img1)

def show(tensor, title):
    img = tensor.squeeze(0).permute(1,2,0).cpu().clamp(0,1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(12,8))

plt.subplot(2,3,1); show(img0, "Input img0")
plt.subplot(2,3,2); show(img1, "Input img1")
plt.subplot(2,3,3); show(pred, "Predicted frame")

plt.subplot(2,3,4); show(aux["warped_img0"], "Warped img0")
plt.subplot(2,3,5); show(aux["warped_img1"], "Warped img1")

# Visibility mask (1 channel)
plt.subplot(2,3,6)
plt.imshow(aux["visibility"].squeeze().cpu(), cmap="gray")
plt.title("Visibility / Mask")
plt.axis("off")
plt.tight_layout()

plt.savefig("image_demo/output.png",dpi = 300)
plt.show()
