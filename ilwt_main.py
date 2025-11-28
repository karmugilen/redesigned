import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision import models
from PIL import Image
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import json
import math

# Differentiable 1D/2D ILWT (LeGall 5/3) utilities
def _reflect_pad_lastdim_1(x: torch.Tensor):
    if x.size(-1) < 2:
        left = x[..., :1]
        right = x[..., -1:]
    else:
        left = x[..., 1:2]
        right = x[..., -2:-1]
    return torch.cat([left, x, right], dim=-1)

def _ilwt53_forward_1d_lastdim(x: torch.Tensor):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    even_ext = _reflect_pad_lastdim_1(even)
    pred = 0.5 * (even_ext[..., :-2] + even_ext[..., 2:])
    d = odd - pred
    d_ext = _reflect_pad_lastdim_1(d)
    upd = 0.25 * (d_ext[..., :-2] + d_ext[..., 2:])
    s = even + upd
    return s, d

def _ilwt53_inverse_1d_lastdim(s: torch.Tensor, d: torch.Tensor):
    d_ext = _reflect_pad_lastdim_1(d)
    upd = 0.25 * (d_ext[..., :-2] + d_ext[..., 2:])
    even = s - upd
    even_ext = _reflect_pad_lastdim_1(even)
    pred = 0.5 * (even_ext[..., :-2] + even_ext[..., 2:])
    odd = d + pred
    last_len = even.size(-1) + odd.size(-1)
    out_shape = list(even.shape)
    out_shape[-1] = last_len
    y = torch.zeros(out_shape, dtype=even.dtype, device=even.device)
    y[..., 0::2] = even
    y[..., 1::2] = odd
    return y

def _apply_forward_1d_along_dim(x: torch.Tensor, dim: int):
    if dim == -1 or dim == x.dim() - 1:
        return _ilwt53_forward_1d_lastdim(x)
    perm = list(range(x.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_perm = x.permute(*perm)
    s, d = _ilwt53_forward_1d_lastdim(x_perm)
    inv_perm = list(range(x_perm.dim()))
    inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    s = s.permute(*inv_perm)
    d = d.permute(*inv_perm)
    return s, d

def _apply_inverse_1d_along_dim(s: torch.Tensor, d: torch.Tensor, dim: int):
    if dim == -1 or dim == s.dim() - 1:
        return _ilwt53_inverse_1d_lastdim(s, d)
    perm = list(range(s.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    s_perm = s.permute(*perm)
    d_perm = d.permute(*perm)
    x_perm = _ilwt53_inverse_1d_lastdim(s_perm, d_perm)
    inv_perm = list(range(x_perm.dim()))
    inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    x = x_perm.permute(*inv_perm)
    return x

class ILWT53_2D(nn.Module):
    def __init__(self, channels: int):
        super(ILWT53_2D, self).__init__()
        self.channels = channels
        self.padding_info = None

    def _maybe_pad(self, x: torch.Tensor):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        self.padding_info = (pad_h, pad_w)
        return x

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = self._maybe_pad(x)
        s_row, d_row = _apply_forward_1d_along_dim(x, dim=3)
        LL, LH = _apply_forward_1d_along_dim(s_row, dim=2)
        HL, HH = _apply_forward_1d_along_dim(d_row, dim=2)
        out = torch.cat([LL, LH, HL, HH], dim=1)
        return out

    def inverse(self, z: torch.Tensor):
        b, c4, h2, w2 = z.shape
        c = c4 // 4
        LL, LH, HL, HH = torch.split(z, c, dim=1)
        s_row = _apply_inverse_1d_along_dim(LL, LH, dim=2)
        d_row = _apply_inverse_1d_along_dim(HL, HH, dim=2)
        x = _apply_inverse_1d_along_dim(s_row, d_row, dim=3)
        if self.padding_info is not None:
            pad_h, pad_w = self.padding_info
            if pad_h:
                x = x[:, :, :-pad_h, :]
            if pad_w:
                x = x[:, :, :, :-pad_w]
        return x

# -----------------------------------------------------------------------
# Swin Transformer Components
# -----------------------------------------------------------------------

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + x
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        # Check if padding needed
        if H % 2 == 1 or W % 2 == 1:
             x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchExpanding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 2 * C)
        x = x.permute(0, 3, 1, 2) # B, 2C, H, W
        x = F.pixel_shuffle(x, 2) # B, C/2, 2H, 2W
        x = x.permute(0, 2, 3, 1) # B, 2H, 2W, C/2
        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)
        ])

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x)
        return x

class SwinUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, window_size=7):
        super().__init__()
        self.ilwt = ILWT53_2D(in_channels)
        self.embed_dim = embed_dim
        
        # Input projection: 24 channels -> embed_dim
        self.patch_embed = nn.Conv2d(24, embed_dim, kernel_size=4, stride=4) 
        
        # Encoder
        self.layer1 = BasicLayer(embed_dim, depth=2, num_heads=3, window_size=window_size)
        self.down1 = PatchMerging(embed_dim)
        
        self.layer2 = BasicLayer(embed_dim*2, depth=2, num_heads=6, window_size=window_size)
        
        # Bottleneck
        self.bottleneck = BasicLayer(embed_dim*2, depth=2, num_heads=6, window_size=window_size)
        
        # Decoder
        self.up1 = PatchExpanding(embed_dim*2)
        self.layer3 = BasicLayer(embed_dim, depth=2, num_heads=3, window_size=window_size)
        
        # Output projection
        self.norm_out = nn.LayerNorm(embed_dim)
        self.conv_out = nn.ConvTranspose2d(embed_dim, 12, kernel_size=4, stride=4)

    def forward(self, cover, secret):
        c_freq = self.ilwt(cover)
        s_freq = self.ilwt(secret)
        x_in = torch.cat([c_freq, s_freq], dim=1) # (B, 24, H/2, W/2)
        
        # Patch Embed
        x = self.patch_embed(x_in) # (B, C, H/8, W/8)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, L, C)
        
        # Encoder
        x1 = self.layer1(x, H, W)
        x_down = self.down1(x1, H, W)
        
        x2 = self.layer2(x_down, H//2, W//2)
        
        # Bottleneck
        x_bottle = self.bottleneck(x2, H//2, W//2)
        
        # Decoder (with skip from x1? Simplified for now)
        x_up = self.up1(x_bottle, H//2, W//2)
        x_up = x_up + x1 # Skip connection
        
        x3 = self.layer3(x_up, H, W)
        x3 = self.norm_out(x3)
        
        # Reshape back
        x_out = x3.transpose(1, 2).view(B, C, H, W)
        out_freq = self.conv_out(x_out)
        
        stego_freq = c_freq + out_freq
        stego = self.ilwt.inverse(stego_freq)
        return stego

class SwinUNetDecoder(nn.Module):
    def __init__(self, out_channels=3, embed_dim=96, window_size=7):
        super().__init__()
        self.ilwt = ILWT53_2D(out_channels)
        self.patch_embed = nn.Conv2d(12, embed_dim, kernel_size=4, stride=4)
        
        self.layer1 = BasicLayer(embed_dim, depth=2, num_heads=3, window_size=window_size)
        self.down1 = PatchMerging(embed_dim)
        self.layer2 = BasicLayer(embed_dim*2, depth=2, num_heads=6, window_size=window_size)
        
        self.up1 = PatchExpanding(embed_dim*2)
        self.layer3 = BasicLayer(embed_dim, depth=2, num_heads=3, window_size=window_size)
        
        self.norm_out = nn.LayerNorm(embed_dim)
        self.conv_out = nn.ConvTranspose2d(embed_dim, 12, kernel_size=4, stride=4)

    def forward(self, stego):
        stego_freq = self.ilwt(stego)
        
        x = self.patch_embed(stego_freq)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        x1 = self.layer1(x, H, W)
        x_down = self.down1(x1, H, W)
        x2 = self.layer2(x_down, H//2, W//2)
        
        x_up = self.up1(x2, H//2, W//2)
        x_up = x_up + x1
        
        x3 = self.layer3(x_up, H, W)
        x3 = self.norm_out(x3)
        
        x_out = x3.transpose(1, 2).view(B, C, H, W)
        out_freq = self.conv_out(x_out)
        
        secret = self.ilwt.inverse(out_freq)
        return secret

# -----------------------------------------------------------------------
# Perceptual Loss (LPIPS)
# -----------------------------------------------------------------------
class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use VGG16 features
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, input, target):
        # Normalize inputs to [-1, 1] if not already? Assuming inputs are [-1, 1]
        # VGG expects normalized [0, 1] with specific mean/std, but for LPIPS 
        # roughly matching range is often sufficient or we can normalize.
        # Here we assume input is [-1, 1], so convert to [0, 1] then normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(input.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(input.device)
        
        in_01 = (input + 1) / 2.0
        tg_01 = (target + 1) / 2.0
        
        in_norm = (in_01 - mean) / std
        tg_norm = (tg_01 - mean) / std
        
        h1 = self.slice1(in_norm)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)

        h1_t = self.slice1(tg_norm)
        h2_t = self.slice2(h1_t)
        h3_t = self.slice3(h2_t)
        
        loss = F.mse_loss(h1, h1_t) + F.mse_loss(h2, h2_t) + F.mse_loss(h3, h3_t)
        return loss

class SwinStegoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = SwinUNetGenerator()
        self.decoder = SwinUNetDecoder()
        
    def forward(self, cover, secret):
        return self.generator(cover, secret)

# Dataset
class ImageSteganographyDataset(Dataset):
    def __init__(self, image_dir, img_size=224, transform=None):
        png_files = glob.glob(os.path.join(image_dir, "*.png"))
        jpg_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        jpeg_files = glob.glob(os.path.join(image_dir, "*.jpeg"))
        self.image_paths = png_files + jpg_files + jpeg_files
        self.img_size = img_size

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (img_size, img_size),
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        host_path = self.image_paths[idx]
        host_img = Image.open(host_path).convert("RGB")
        host_tensor = self.transform(host_img)

        secret_idx = random.choice(
            [i for i in range(len(self.image_paths)) if i != idx]
        )
        secret_path = self.image_paths[secret_idx]
        secret_img = Image.open(secret_path).convert("RGB")
        secret_tensor = self.transform(secret_img)

        combined_input = torch.cat([host_tensor, secret_tensor], dim=0)
        return combined_input, host_tensor, secret_tensor

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    total_size = len(dataset)
    indices = list(range(total_size))
    random.seed(random_seed)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return train_dataset, val_dataset, test_dataset

# Metrics
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    max_pixel = 2.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse + 1e-10))
    return psnr

def train_model(model, train_dataset, val_dataset, num_epochs=150):
    print("\nTraining ILWT Steganography Swin-UNet...")
    
    os.makedirs("research_metrics", exist_ok=True)
    os.makedirs("research_checkpoints", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Perceptual Loss
    lpips = LPIPSLoss().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion_pixel = nn.MSELoss()
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    
    metrics_history = {
        'epoch': [],
        'loss': [],
        'hiding_psnr': [],
        'recovery_psnr': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (input_tensor, host_tensor, secret_tensor) in enumerate(train_dataloader):
            host_tensor = host_tensor.to(device)
            secret_tensor = secret_tensor.to(device)
            
            optimizer.zero_grad()
            
            stego = model.generator(host_tensor, secret_tensor)
            recovered = model.decoder(stego)
            
            # Loss Calculation
            loss_hiding_mse = criterion_pixel(stego, host_tensor)
            loss_recovery_mse = criterion_pixel(recovered, secret_tensor)
            loss_lpips = lpips(stego, host_tensor)
            
            loss = 10.0 * loss_hiding_mse + 10.0 * loss_recovery_mse + 1.0 * loss_lpips
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        val_hiding_psnr = 0.0
        val_recovery_psnr = 0.0
        with torch.no_grad():
            for _, host_tensor, secret_tensor in val_dataloader:
                host_tensor = host_tensor.to(device)
                secret_tensor = secret_tensor.to(device)
                
                stego = model.generator(host_tensor, secret_tensor)
                recovered = model.decoder(stego)
                
                val_hiding_psnr += calculate_psnr(stego, host_tensor).item()
                val_recovery_psnr += calculate_psnr(recovered, secret_tensor).item()
                
        avg_hiding = val_hiding_psnr / len(val_dataloader)
        avg_recovery = val_recovery_psnr / len(val_dataloader)
        avg_loss = epoch_loss / len(train_dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"Hiding PSNR: {avg_hiding:.2f} | Recovery PSNR: {avg_recovery:.2f}")
        
        metrics_history['epoch'].append(epoch+1)
        metrics_history['loss'].append(avg_loss)
        metrics_history['hiding_psnr'].append(avg_hiding)
        metrics_history['recovery_psnr'].append(avg_recovery)
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"research_checkpoints/swin_epoch_{epoch+1}.pth")
            
    return model, metrics_history

def test_model(model, dataset, num_samples=10):
    print(f"\nTesting model on {num_samples} samples...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    os.makedirs("ilwt_test_results", exist_ok=True)
    
    test_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(test_indices):
        _, host_tensor, secret_tensor = dataset[idx]
        host_tensor = host_tensor.unsqueeze(0).to(device)
        secret_tensor = secret_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            stego = model.generator(host_tensor, secret_tensor)
            recovered = model.decoder(stego)
            
            h_psnr = calculate_psnr(stego, host_tensor).item()
            r_psnr = calculate_psnr(recovered, secret_tensor).item()
            
            # Visualization
            def denorm(x): return torch.clamp((x + 1) / 2, 0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
            
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax[0].imshow(denorm(host_tensor))
            ax[0].set_title("Cover")
            ax[1].imshow(denorm(secret_tensor))
            ax[1].set_title("Secret")
            ax[2].imshow(denorm(stego))
            ax[2].set_title(f"Stego\nPSNR: {h_psnr:.2f}")
            ax[3].imshow(denorm(recovered))
            ax[3].set_title(f"Recovered\nPSNR: {r_psnr:.2f}")
            plt.savefig(f"ilwt_test_results/test_{i}.png")
            plt.close()
            
            print(f"Sample {i}: Hiding PSNR: {h_psnr:.2f}, Recovery PSNR: {r_psnr:.2f}")

def main():
    print("ILWT Steganography Swin-UNet")
    image_dir = "my_images"
    full_dataset = ImageSteganographyDataset(image_dir)
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    
    model = SwinStegoModel()
    model, _ = train_model(model, train_dataset, val_dataset, num_epochs=200)
    
    torch.save(model.state_dict(), "ilwt_swin_model.pth")
    test_model(model, test_dataset)

if __name__ == "__main__":
    main()
