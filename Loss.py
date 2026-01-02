import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gauss_kernel(size=5, channels=3):
    # Gaussian kernel for pyramid construction
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

def conv_gauss(img, kernel):
    # Apply Gaussian convolution
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros_like(x)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1]).to(x.device))

def laplacian_pyramid(img, max_levels=5):
    # Build Laplacian pyramid
    kernel = gauss_kernel(channels=img.shape[1]).to(img.device)
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr

class LaplacianLoss(nn.Module):
    # Laplacian Pyramid Loss
    def __init__(self, max_levels=5, channels=3):
        super().__init__()
        self.max_levels = max_levels
        
    def forward(self, pred, target):
        pyr_pred = laplacian_pyramid(pred, self.max_levels)
        pyr_target = laplacian_pyramid(target, self.max_levels)
        loss = sum(F.l1_loss(a, b) for a, b in zip(pyr_pred, pyr_target))
        return loss



class TernaryLoss(nn.Module):
    # Ternary Census Transform Loss
    def __init__(self, patch_size=7):
        super().__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        # Create convolution weights for patch extraction
        w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        w = np.transpose(w, (3, 2, 0, 1))
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32))

    def transform(self, img):
        # Apply census transform
        patches = F.conv2d(img, self.w, padding=self.patch_size//2, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        # Convert RGB to grayscale
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        # Hamming distance
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        # Create valid mask for borders
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2*padding, w - 2*padding, device=t.device)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, pred, target):
        pred_census = self.transform(self.rgb2gray(pred))
        target_census = self.transform(self.rgb2gray(target))
        return (self.hamming(pred_census, target_census) * self.valid_mask(pred, 1)).mean()


class MeanShift(nn.Conv2d):
    # Normalize input for VGG
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super().__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class VGGPerceptualLoss(nn.Module):
    # VGG19 Perceptual Loss
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = vgg
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True)
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        pred = self.normalize(pred)
        target = self.normalize(target)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        
        loss = 0
        k = 0
        for i in range(indices[-1]):
            pred = self.vgg_layers[i](pred)
            target = self.vgg_layers[i](target)
            if (i+1) in indices:
                loss += weights[k] * (pred - target.detach()).abs().mean() * 0.1
                k += 1
        
        return loss



class CombinedLoss(nn.Module):
    """
    Combined Loss Function :
    - Laplacian Pyramid Loss (main)
    - VGG Perceptual Loss (texture/style)
    - Ternary Loss (for texture matching)
    - Teacher-Student Distillation
    """
    def __init__(self, use_vgg=True, use_ternary=False):
        super().__init__()
        self.lap_loss = LaplacianLoss(max_levels=5)
        self.use_vgg = use_vgg
        self.use_ternary = use_ternary
        
        if use_vgg:
            self.vgg_loss = VGGPerceptualLoss()
        
        if use_ternary:
            self.ternary_loss = TernaryLoss()
    
    def forward(self, pred, gt, merged_teacher=None, flow_student=None, flow_teacher=None):
        """
        Args:
            pred: Final predicted frame (student output after refinement)
            gt: Ground truth frame
            merged_teacher: Teacher merged output (for distillation)
            flow_student: Student flow predictions (list of 3 scales)
            flow_teacher: Teacher flow prediction
        
        Returns:
            total_loss, loss_dict
        """
        # Main loss - Laplacian Pyramid
        loss_l1 = self.lap_loss(pred, gt)
        
        total_loss = loss_l1
        loss_dict = {'loss_l1': loss_l1.item()}
        
        # VGG Perceptual Loss
        if self.use_vgg:
            loss_vgg = self.vgg_loss(pred, gt)
            total_loss += loss_vgg
            loss_dict['loss_vgg'] = loss_vgg.item()
        
        # Ternary Loss
        if self.use_ternary:
            loss_ternary = self.ternary_loss(pred, gt)
            total_loss += 0.1 * loss_ternary
            loss_dict['loss_ternary'] = loss_ternary.item()
        
        # Teacher-Student Distillation
        if merged_teacher is not None:
            loss_tea = self.lap_loss(merged_teacher, gt)
            total_loss += loss_tea
            loss_dict['loss_tea'] = loss_tea.item()
        
        # Flow Distillation 
        if flow_student is not None and flow_teacher is not None:
            loss_distill = 0
            for i, weights_s in enumerate(flow_student):
                # weights_s shape: [B, 9, H, W]
                if isinstance(flow_teacher, list):
                    weights_t = flow_teacher[i] 
                else:
                    weights_t = F.interpolate(flow_teacher, size=weights_s.shape[2:], mode='bilinear', align_corners=False)
                loss_distill += (weights_s - weights_t.detach()).abs().mean()
        

            total_loss += 0.01 * loss_distill 
            loss_dict['loss_distill'] = loss_distill.item()
        
        return total_loss, loss_dict
