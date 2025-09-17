import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from typing import Tuple

# -----------------------------
# 1) Vision Transformer blocks
# -----------------------------

class PatchEmbed(nn.Module):
    """Conv-based patch embedding."""
    def __init__(self, in_ch=4, embed_dim=384, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)   # [B, D, H/ps, W/ps]
        B, D, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D], N = H_p * W_p
        return x, (H_p, W_p)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each [B, N, heads, head_dim]

        q = q.permute(0, 2, 1, 3)  # [B, heads, N, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # [B, heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):  # [B, N, C]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# 2) ViT-UNet segmentation head
# -----------------------------

class ViTUNet(nn.Module):
    """
    Simple ViT encoder + conv decoder that upsamples to full resolution.
    - in_ch: 4 channels (DEM, LC, 1D precip, 5D precip)
    - patch_size: 16 (so feature map is downsampled by 16)
    - embed_dim: transformer width
    - depth: number of transformer blocks
    """
    def __init__(self, in_ch=4, embed_dim=384, depth=8, num_heads=6, patch_size=16, out_ch=1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(in_ch=in_ch, embed_dim=embed_dim, patch_size=patch_size)
        self.pos_embed = None  # (optional) sinusoidal or learned; for simplicity we omit
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder: reshape tokens to [B, D, H', W'] then upsample back to full res
        self.convproj = nn.Conv2d(embed_dim, 256, kernel_size=1)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # x2
        self.up2 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)  # x4
        self.up3 = nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2)  # x8
        self.up4 = nn.ConvTranspose2d(32,  16,  kernel_size=2, stride=2)  # x16
        self.head = nn.Conv2d(16, out_ch, kernel_size=1)

    def forward(self, x):  # x: [B, 4, H, W]
        B, C, H, W = x.shape
        # ensure H,W divisible by patch_size inside tiling function (we do that in inference)
        tokens, (Hp, Wp) = self.patch_embed(x)  # [B, N, D]
        # (optional pos embed) — kept simple here

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)  # [B, N, D]

        feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)  # [B, D, Hp, Wp]
        y = self.convproj(feat)
        y = self.up1(y)
        y = self.up2(y)
        y = self.up3(y)
        y = self.up4(y)  # [B, 16, H, W]
        y = self.head(y)  # [B, 1, H, W]
        return y


# -----------------------------
# 3) Raster I/O & normalization
# -----------------------------

def read_single_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        crs = src.crs
    return arr, profile, transform, crs

def write_single_band(path, arr, profile):
    prof = profile.copy()
    prof.update(count=1, dtype="float32", compress="deflate", predictor=2, tiled=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)

def normalize_band(x, mean=None, std=None, eps=1e-6):
    if mean is None: mean = float(np.nanmean(x))
    if std  is None: std  = float(np.nanstd(x))
    std = std if std > eps else eps
    return (x - mean) / std, (mean, std)

def prepare_input_tensor(dem, lc, d1, d5):
    # DEM z-score
    dem_n, _ = normalize_band(dem)
    # LC -> binary or scaled; here: keep numeric, z-score as well
    lc_n, _  = normalize_band(lc)
    # Precip z-score
    d1_n, _  = normalize_band(d1)
    d5_n, _  = normalize_band(d5)

    x = np.stack([dem_n, lc_n, d1_n, d5_n], axis=0)  # [4, H, W]
    return torch.from_numpy(x).unsqueeze(0)  # [1, 4, H, W]


# -----------------------------
# 4) Sliding-window inference
# -----------------------------

@torch.no_grad()
def sliding_window_predict(
    model: nn.Module,
    tensor_1x4HW: torch.Tensor,
    tile: Tuple[int, int] = (1024, 1024),
    stride: Tuple[int, int] = (896, 896),   # 128px overlap
    device: str = "cuda"
):
    """
    Predict over a large image using overlapping tiles with smooth blending.
    tensor_1x4HW: torch tensor [1, 4, H, W]
    returns: np.ndarray [H, W]
    """
    model.eval()
    x = tensor_1x4HW.to(device)
    _, _, H, W = x.shape
    th, tw = tile
    sh, sw = stride

    # Output and weight accumulators
    out_acc = torch.zeros((1, 1, H, W), device=device)
    w_acc   = torch.zeros((1, 1, H, W), device=device)

    # cosine window for smooth blending at overlaps
    def blend_mask(h, w, pad=32):
        yy = np.linspace(-math.pi, math.pi, h)
        xx = np.linspace(-math.pi, math.pi, w)
        wy = (0.5*(1+np.cos(yy))).astype(np.float32)
        wx = (0.5*(1+np.cos(xx))).astype(np.float32)
        mask = np.outer(wy, wx)
        mask = torch.from_numpy(mask).to(device).unsqueeze(0).unsqueeze(0)
        return mask.clamp_(min=1e-3)

    mask = blend_mask(th, tw)

    for top in range(0, H, sh):
        for left in range(0, W, sw):
            bottom = min(top + th, H)
            right  = min(left + tw, W)

            # pad tile if near border to ensure size multiple of patch_size (16)
            pad_bottom = th - (bottom - top)
            pad_right  = tw - (right - left)

            x_tile = torch.zeros((1, 4, th, tw), device=device)
            x_tile[:, :, :bottom-top, :right-left] = x[:, :, top:bottom, left:right]

            # make sure H,W divisible by patch size (16) is true for the tile size used
            y_tile = model(x_tile)  # [1,1,th,tw]

            # accumulate with blending mask cropped to actual content size
            m = mask[:, :, :bottom-top, :right-left]
            out_acc[:, :, top:bottom, left:right] += y_tile[:, :, :bottom-top, :right-left] * m
            w_acc[:, :, top:bottom, left:right]   += m

    out = (out_acc / (w_acc + 1e-6)).squeeze().detach().cpu().numpy()  # [H, W]
    return out


# -----------------------------
# 5) Run end-to-end
# -----------------------------

def run_vit_inference(
    dem_path: str,
    lc_path: str,
    d1_path: str,
    d5_path: str,
    out_path: str,
    device: str = "cuda",
    tile=(1024, 1024),
    stride=(896, 896),
    patch_size=16
):
    # Read rasters
    dem, profile, transform, crs = read_single_band(dem_path)
    lc,  _, _, _ = read_single_band(lc_path)
    d1,  _, _, _ = read_single_band(d1_path)
    d5,  _, _, _ = read_single_band(d5_path)

    # Sanity checks
    assert dem.shape == lc.shape == d1.shape == d5.shape, "All inputs must have same H,W"
    H, W = dem.shape

    # Prepare input tensor
    x = prepare_input_tensor(dem, lc, d1, d5)  # [1,4,H,W]

    # Build model
    model = ViTUNet(
        in_ch=4, embed_dim=384, depth=8, num_heads=6,
        patch_size=patch_size, out_ch=1
    ).to(device)

    # (Optional) load pretrained weights if you have them:
    # model.load_state_dict(torch.load("vit_unet_flood.pth", map_location=device))

    # Predict with tiling
    pred = sliding_window_predict(
        model, x, tile=tile, stride=stride, device=device
    )  # [H, W] float32

    # Optionally clamp to non-negative depths
    pred = np.maximum(pred, 0.0).astype(np.float32)

    # Save GeoTIFF with original georeferencing
    write_single_band(out_path, pred, profile)
    print(f"Saved flood map: {out_path}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Replace these with your actual file paths
    DEM_PATH = "/path/to/DEM.tif"            # H=11886, W=9266
    LC_PATH  = "/path/to/LC.tif"
    D1_PATH  = "/path/to/precip_1day.tif"
    D5_PATH  = "/path/to/precip_5day.tif"
    OUT_PATH = "/path/to/flood_inundation_vit.tif"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose GPU if needed

    run_vit_inference(
        DEM_PATH, LC_PATH, D1_PATH, D5_PATH, OUT_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tile=(1024, 1024),    # adjust if you have more/less GPU memory
        stride=(896, 896),    # 128px overlap for smooth seams
        patch_size=16         # ViT patch size
    )
