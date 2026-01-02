import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

def conv_dw_pw(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

def safe_shift(x, dy, dx):
    if dy == 0 and dx == 0:
        return x
    shifted_x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    if dy > 0:
        shifted_x[:, :, :dy, :] = 0
    elif dy < 0:
        shifted_x[:, :, dy:, :] = 0
    if dx > 0:
        shifted_x[:, :, :, :dx] = 0
    elif dx < 0:
        shifted_x[:, :, :, dx:] = 0
    return shifted_x

def get_directions(D):

    return [
        (0,0), (-D,0), (D,0), (0,-D), (0,D), 
        (-D,-D), (-D,D), (D,-D), (D,D)
    ]


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ContextExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.out_channels = [16, 32, 64, 96] 
        
        # Scale 1/1
        self.conv0_0 = conv(3, 16, stride=1)
        self.conv0_1 = conv(16, 16)
        
        # Scale 1/2
        self.conv1_0 = conv(16, 32, stride=2)
        self.conv1_1 = conv(32, 32)
        
        # Scale 1/4
        self.conv2_0 = conv(32, 64, stride=2)
        self.conv2_1 = conv(64, 64)
        
        # Scale 1/8
        self.conv3_0 = conv(64, 96, stride=2)
        self.conv3_1 = conv(96, 96)

    def forward(self, img):
        c0 = self.conv0_1(self.conv0_0(img))
        c1 = self.conv1_1(self.conv1_0(c0))
        c2 = self.conv2_1(self.conv2_0(c1))
        c3 = self.conv3_1(self.conv3_0(c2))
        return c0, c1, c2, c3

# Bi-directional Multi-scale Shift Flow Head
class BidirectionalShiftFlowHead(nn.Module):
    def __init__(self, in_ch, base_ch=64, scales=[1,2,4,8], num_directions=9):
        super().__init__()
        
        self.scales = scales
        self.num_directions = num_directions # Dynamic directions
        self.num_scales = len(scales)
        
        self.feature_extractor = nn.Sequential(
            conv_dw_pw(in_ch, base_ch),
            conv_dw_pw(base_ch, base_ch),
            conv_dw_pw(base_ch, base_ch)
        )

        out_channels = self.num_directions * self.num_scales
        
        self.predict_weights_forward = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        self.predict_weights_backward = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        
        self.predict_visibility = nn.Sequential(
            conv(base_ch, base_ch // 2),
            nn.Conv2d(base_ch // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f0, f1):
        x = torch.cat([f0, f1], dim=1)
        feat = self.feature_extractor(x)

        def process_weights(weight_tensor):
            weight_list = []
            for i in range(self.num_scales):

                start_ch = i * self.num_directions
                end_ch = (i + 1) * self.num_directions
                
                weights_i = weight_tensor[:, start_ch:end_ch, ...]
                weights_i = torch.softmax(weights_i, dim=1)
                weight_list.append(weights_i)
            return weight_list
            
        weight_list_fwd = process_weights(self.predict_weights_forward(feat))
        weight_list_bwd = process_weights(self.predict_weights_backward(feat))
        
        visibility = self.predict_visibility(feat)
        
        return weight_list_fwd, weight_list_bwd, visibility

# Context Aware Refiner
class ContextAwareRefiner(nn.Module):
    def __init__(self, base_in_ch, out_ch=3):
        super().__init__()
        
        self.proj_c1 = nn.Conv2d(64, 32, 1, bias=False)  
        self.proj_c2 = nn.Conv2d(128, 64, 1, bias=False) 
        self.proj_c3 = nn.Conv2d(192, 96, 1, bias=False) 
        
        self.down1 = nn.Sequential(
            conv(base_in_ch, 64),
            ChannelAttention(64),
            conv(64, 64)
        )
        
        self.down2 = nn.Sequential(
            conv(64 + 32, 96, stride=2), 
            ChannelAttention(96),
            conv(96, 96)
        )
        
        self.down3 = nn.Sequential(
            conv(96 + 64, 160, stride=2),
            ChannelAttention(160),
            conv(160, 128)
        )
        
        self.neck = nn.Sequential(
            conv(128 + 96, 224),
            ChannelAttention(224),
            conv(224, 192),
            conv(192, 128)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_conv1 = nn.Sequential(
            conv(128 + 96, 128),
            ChannelAttention(128),
            conv(128, 96)
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_conv2 = nn.Sequential(
            conv(96 + 64, 96),
            ChannelAttention(96),
            conv(96, 64)
        )
        
        self.out = nn.Sequential(
            conv(64, 32),
            nn.Conv2d(32, out_ch, 3, 1, 1)
        )

    def forward(self, x, context0, context1):
        c0_0, c1_0, c2_0, c3_0 = context0
        c0_1, c1_1, c2_1, c3_1 = context1
        
        # Encoder 1
        d1 = self.down1(x)
        
        # Context 1 processing
        c1_concat = torch.cat([c1_0, c1_1], dim=1)
        c1_concat = F.interpolate(c1_concat, size=d1.shape[2:], mode='bilinear', align_corners=False)
        c1_concat = self.proj_c1(c1_concat)
        
        # Encoder 2
        d2 = self.down2(torch.cat([d1, c1_concat], dim=1))
        
        # Context 2 processing
        c2_concat = torch.cat([c2_0, c2_1], dim=1)
        c2_concat = F.interpolate(c2_concat, size=d2.shape[2:], mode='bilinear', align_corners=False)
        c2_concat = self.proj_c2(c2_concat)
        
        # Encoder 3
        d3 = self.down3(torch.cat([d2, c2_concat], dim=1))
        
        # Context 3 processing
        c3_concat = torch.cat([c3_0, c3_1], dim=1)
        c3_concat = F.interpolate(c3_concat, size=d3.shape[2:], mode='bilinear', align_corners=False)
        c3_concat = self.proj_c3(c3_concat)
        
        # Neck
        neck = self.neck(torch.cat([d3, c3_concat], dim=1))
        
        # Decoder
        u1 = self.up_conv1(torch.cat([self.up1(neck), d2], dim=1))
        u2 = self.up_conv2(torch.cat([self.up2(u1), d1], dim=1))
        
        return self.out(u2)

# Main Model
class MainModel(nn.Module):
    def __init__(self, scales=[1,2,4,8,16,32]):
        super().__init__()
        self.scales = scales
        num_scales = len(scales)
        
        self.num_directions = len(get_directions(1)) 
        
        # STUDENT
        self.context_extractor_student = ContextExtractor()
        self.flow_encoder_student = nn.Sequential(conv(64, 64), conv(64, 64))
        
        self.shift_flow_student = BidirectionalShiftFlowHead(
            in_ch=128, base_ch=96, scales=scales, num_directions=self.num_directions
        )
 
        self.context_extractor_teacher = ContextExtractor()
        self.flow_encoder_teacher = nn.Sequential(conv(64, 64), conv(64, 64))
        
        self.shift_flow_teacher = BidirectionalShiftFlowHead(
            in_ch=128 + 6, base_ch=96, scales=scales, num_directions=self.num_directions
        )
        c_img = 3 * 2              
        c_warped = 3 * 2          
        c_merged = 3              
        c_mask = 1                 
        c_weights = 2 * num_scales * self.num_directions 
        
        self.refiner_in_ch = c_img + c_warped + c_merged + c_weights + c_mask
        
        self.refiner = ContextAwareRefiner(base_in_ch=self.refiner_in_ch, out_ch=3)
        
    def apply_shift(self, img, weights_list):
        shift_cache = {}
        warped_img = 0
        

        for scale, weights in zip(self.scales, weights_list):
            directions = get_directions(scale)
            if weights.shape[1] != len(directions):
                raise RuntimeError(f"Mismatch: Weights ch={weights.shape[1]} vs Directions={len(directions)}")

            weights_full = F.interpolate(weights, scale_factor=4, 
                                         mode='bilinear', align_corners=False)
            
            warped_scale = 0
            for i, (dy, dx) in enumerate(directions):
                shift_key = (dy, dx)
                if shift_key not in shift_cache:
                    shift_cache[shift_key] = safe_shift(img, dy, dx)
                warped_scale += weights_full[:, i:i+1, ...] * shift_cache[shift_key]
            
            warped_img += warped_scale
        
        return warped_img / len(self.scales)
    
    def forward(self, img0, img1, gt=None):
        # STUDENT BRANCH
        context0_s = self.context_extractor_student(img0)
        context1_s = self.context_extractor_student(img1)
        
        c2_0_s, c2_1_s = context0_s[2], context1_s[2] 
        f0_s = self.flow_encoder_student(c2_0_s)
        f1_s = self.flow_encoder_student(c2_1_s)
         
        weight_list_fwd_s, weight_list_bwd_s, visibility_s = self.shift_flow_student(f0_s, f1_s)
        warped_img0_s = self.apply_shift(img0, weight_list_fwd_s)
        warped_img1_s = self.apply_shift(img1, weight_list_bwd_s)
        
        visibility_full_s = F.interpolate(visibility_s, scale_factor=4, mode='bilinear', align_corners=False)
        merged_student = visibility_full_s * warped_img0_s + (1 - visibility_full_s) * warped_img1_s
        
        #  TEACHER BRANCH (Only training when GT exists) 
        
        merged_teacher = None
        weight_list_fwd_t = None 
        weight_list_bwd_t = None
        
        if gt is not None:
            context0_t = self.context_extractor_teacher(img0)
            context1_t = self.context_extractor_teacher(img1)
            
            c2_0_t, c2_1_t = context0_t[2], context1_t[2]
            f0_t = self.flow_encoder_teacher(c2_0_t)
            f1_t = self.flow_encoder_teacher(c2_1_t)
            
            interp_gt = F.interpolate(gt, scale_factor=0.25, mode='bilinear', align_corners=False)
            f0_tea = torch.cat([f0_t, interp_gt], dim=1)
            f1_tea = torch.cat([f1_t, interp_gt], dim=1)

            weight_list_fwd_t, weight_list_bwd_t, vis_t = self.shift_flow_teacher(f0_tea, f1_tea)
            
            w_img0_t = self.apply_shift(img0, weight_list_fwd_t)
            w_img1_t = self.apply_shift(img1, weight_list_bwd_t)
            vis_full_t = F.interpolate(vis_t, scale_factor=4, mode='bilinear', align_corners=False)
            merged_teacher = vis_full_t * w_img0_t + (1 - vis_full_t) * w_img1_t
        
        all_weights_fwd = torch.cat(weight_list_fwd_s, dim=1)
        all_weights_bwd = torch.cat(weight_list_bwd_s, dim=1)
        
        all_weights_fwd_full = F.interpolate(all_weights_fwd, scale_factor=4, mode='bilinear', align_corners=False)
        all_weights_bwd_full = F.interpolate(all_weights_bwd, scale_factor=4, mode='bilinear', align_corners=False)
    
        refine_input = torch.cat([
            img0, img1,                 
            warped_img0_s, warped_img1_s, 
            merged_student,             
            all_weights_fwd_full,      
            all_weights_bwd_full,       
            visibility_full_s           
        ], dim=1)
        
        residual = self.refiner(refine_input, context0_s, context1_s)
        pred_frame = torch.clamp(merged_student + residual, 0, 1)
        
        return pred_frame, {
            'merged_student': merged_student,
            'merged_teacher': merged_teacher,
            'warped_img0': warped_img0_s,
            'warped_img1': warped_img1_s,
            'visibility': visibility_full_s,
            'weights_fwd_student': weight_list_fwd_s,
            'weights_fwd_teacher': weight_list_fwd_t,
        }