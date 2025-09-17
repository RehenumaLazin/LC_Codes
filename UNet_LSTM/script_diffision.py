# --- Device & AMP (works for NVIDIA CUDA and AMD ROCm) ---
# import torch
# from contextlib import nullcontext

# if torch.cuda.is_available():                     # <-- covers ROCm too
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# # Use AMP only on NVIDIA CUDA; disable on ROCm/CPU/MPS to avoid dtype mismatches
# if device.type == "cuda" and torch.version.hip is None:  # NVIDIA only
#     from torch.cuda.amp import autocast, GradScaler
#     def autocast_ctx(): return autocast()
#     scaler = GradScaler()
# else:
#     def autocast_ctx(): return nullcontext()             # no-op
#     class _DummyScaler:
#         def scale(self, loss): return loss
#         def step(self, opt): opt.step()
#         def update(self): pass
#     scaler = _DummyScaler()




import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import rasterio
from collections import defaultdict
from sklearn.model_selection import KFold

# --- Device ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    
elif torch.version.hip is not None:
    from torch.amp import autocast, GradScaler
    # device = torch.device("hip")
else:
    from torch.cuda.amp import autocast, GradScaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

# --- Device ---
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("hip") if torch.version.hip is not None
    else torch.device("cpu")
)

from contextlib import nullcontext
# Force full float32 training everywhere (disable autocast on ROCm)
def autocast_ctx():
    return nullcontext()

class _DummyScaler:
    def scale(self, loss): return loss
    def step(self, opt): opt.step()
    def update(self): pass
scaler = _DummyScaler()

# ============================================================
# ======================== DATASET ===========================
# ============================================================

class FloodDataset(Dataset):
    def __init__(self, event, dem_dir, lc_dir, flowline_dir, precip_dir, flood_dir,
                 tile_csv, hours, seq_len,
                 DEM_mean, DEM_std, precip_mean, precip_std,
                 flood_mean=None, flood_std=None,
                 output_geotiff_dir=None, return_metadata=False, mode="train"):
        self.event = event
        self.dem_dir = dem_dir
        self.lc_dir = lc_dir
        self.flowline_dir = flowline_dir
        self.precip_dir = precip_dir
        self.flood_dir = flood_dir
        self.geo_dir = output_geotiff_dir
        self.seq_len = seq_len
        self.hours = hours
        self.return_metadata = return_metadata
        self.mode = mode.lower()

        self.DEM_mean = DEM_mean
        self.DEM_std = DEM_std
        self.precip_mean = precip_mean
        self.precip_std = precip_std
        self.flood_mean = flood_mean
        self.flood_std = flood_std

        self.tiles = pd.read_csv(tile_csv)["ras_name"].tolist()
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        for tile in self.tiles:
            for i in range(self.seq_len - 1, len(self.hours)):
                seq_hours = self.hours[i - self.seq_len + 1:i + 1]
                target_hour = self.hours[i]
                samples.append((tile, seq_hours, target_hour))
        return samples

    def _read_raster(self, path):
        with rasterio.open(path) as src:
            array = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
        return array, transform, crs

    def _load_static_inputs(self, tile):
        dem_path = os.path.join(self.dem_dir, f"{tile}")
        lc_path = os.path.join(self.lc_dir, f"LC_{tile}")
        flowline_path = os.path.join(self.flowline_dir, f"flowline_{tile}")

        dem, _, _ = self._read_raster(dem_path)
        lc, _, _ = self._read_raster(lc_path)
        flowline, _, _ = self._read_raster(flowline_path)

        dem_norm = (dem - self.DEM_mean) / self.DEM_std
        lc_binary = np.where(lc == 11, 1, 0).astype(np.float32)
        flowline_binary = np.where(flowline == 1, 1, 0).astype(np.float32)

        static_input = np.stack([dem_norm, lc_binary, flowline_binary], axis=0)
        return torch.from_numpy(static_input)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tile, seq_hours, target_hour = self.samples[idx]

        static_input = self._load_static_inputs(tile)

        precip_seq = []
        for h in seq_hours:
            p_path = os.path.join(self.precip_dir, f"1H_prec_{self.event}_{h.strftime('%Y%m%d%H')}_{tile}")
            p, _, _ = self._read_raster(p_path)
            p_norm = (p - self.precip_mean) / self.precip_std
            precip_seq.append(p_norm)

        precip_seq = np.stack(precip_seq, axis=0).astype(np.float32)
        precip_tensor = torch.from_numpy(precip_seq)
        _, transform, crs = self._read_raster(p_path)

        if self.mode == "train":
            label_path = os.path.join(self.flood_dir, f"{self.event}_{target_hour.strftime('%Y%m%d%H')}_{tile}")
            label, transform, crs = self._read_raster(label_path)
            if self.flood_mean is not None and self.flood_std is not None:
                label = (label - self.flood_mean) / self.flood_std
            label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            return static_input, precip_tensor, label_tensor

        if self.return_metadata:
            output_geotiff_path = os.path.join(self.geo_dir, f"{target_hour.strftime('%Y%m%d%H')}_{tile}")
            return static_input, precip_tensor, output_geotiff_path, transform, crs

        return static_input, precip_tensor


def custom_collate_fn(batch):
    if len(batch[0]) == 3:  # Training
        static_inputs = torch.stack([item[0] for item in batch])
        precip_tensors = torch.stack([item[1] for item in batch])
        label_tensors = torch.stack([item[2] for item in batch])
        return static_inputs, precip_tensors, label_tensors
    elif len(batch[0]) == 5:  # Testing
        static_inputs = torch.stack([item[0] for item in batch])
        precip_tensors = torch.stack([item[1] for item in batch])
        paths = [item[2] for item in batch]
        transforms = [item[3] for item in batch]
        crs_list = [item[4] for item in batch]
        return static_inputs, precip_tensors, paths, transforms, crs_list


# ============================================================
# ======================== MODEL =============================
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class UNetLSTM(nn.Module):
    def __init__(self, in_static_ch=3, in_seq_ch=1, seq_len=5, hidden_dim=64):
        super().__init__()
        self.encoder_static = ConvBlock(in_static_ch, 32)
        self.encoder_seq = ConvBlock(in_seq_ch, 32)
        self.pool = nn.MaxPool2d(2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(32 + hidden_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, static_input, seq_input):
        B, T, H, W = seq_input.shape
        x_static = self.encoder_static(static_input)
        x_seq = seq_input.view(B * T, 1, H, W)
        x_seq = self.encoder_seq(x_seq)
        x_seq = self.pool(x_seq)
        x_seq = x_seq.view(B, T, 32, H // 2, W // 2)
        x_seq = x_seq.mean(dim=[3, 4])
        lstm_out, _ = self.lstm(x_seq)
        lstm_feat = lstm_out[:, -1, :].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined = torch.cat([x_static, lstm_feat], dim=1)
        return self.decoder(combined)


# ============================================================
# ================== DIFFUSION WRAPPER =======================
# ============================================================

class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, device="cuda"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device

    def forward(self, x, cond_static, cond_seq):
        noise = torch.randn_like(x)
        t = torch.randint(0, self.timesteps, (x.size(0),), device=x.device).long()
        x_noisy = x + noise
        pred = self.model(cond_static, cond_seq)
        return F.mse_loss(pred, x)


# ============================================================
# =============== CHECKPOINT HANDLING ========================
# ============================================================

def save_checkpoint(diffusion, optimizer, epoch, fold, checkpoint_path):
    checkpoint = {
        "epoch": epoch,
        "fold": fold,
        "model_state_dict": diffusion.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[Checkpoint] Saved at fold {fold}, epoch {epoch} → {checkpoint_path}")


def load_checkpoint(checkpoint_path, diffusion, optimizer, device="cuda"):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        diffusion.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[Checkpoint] Resumed fold {checkpoint['fold']}, epoch {checkpoint['epoch']}")
        return checkpoint["epoch"] + 1, checkpoint["fold"]
    else:
        print(f"[Checkpoint] None found at {checkpoint_path}, starting fresh")
        return 0, 0


def load_model(model_path, device="cuda"):
    model = UNetLSTM(in_static_ch=3, in_seq_ch=1, seq_len=5)
    diffusion = GaussianDiffusion(model, timesteps=1000, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        diffusion.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[Model] Loaded {model_path} (epoch {checkpoint['epoch']}, fold {checkpoint['fold']})")
    else:
        print(f"[Model] No checkpoint found at {model_path}")

    return diffusion, optimizer


# ============================================================
# =============== TRAINING / TESTING ========================
# ============================================================

def train_diffusion(diffusion, train_loader, val_loader, optimizer, checkpoint_path, scaler, fold_idx, start_epoch, num_epochs=10):
    diffusion.train()
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for static, seq, target in tqdm(train_loader, desc=f"[Fold {fold_idx}] Epoch {epoch}"):
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                loss = diffusion(target.to(device), static.to(device), seq.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * static.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[Train] Fold {fold_idx}, Epoch {epoch}, Loss={epoch_loss:.4f}")
        save_checkpoint(diffusion, optimizer, epoch, fold_idx, checkpoint_path)

    return diffusion


def test_diffusion(diffusion, dataset, device, output_dir):
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    diffusion.eval()
    with torch.no_grad():
        for static_input, precip_tensor, paths, transforms, crs_list in tqdm(dataloader, desc="Testing"):
            static_input = static_input.to(device)
            precip_tensor = precip_tensor.to(device)
            outputs = diffusion.model(static_input, precip_tensor).cpu().numpy()
            if outputs.ndim == 4:
                outputs = outputs.squeeze(1)
            for i in range(len(paths)):
                with rasterio.open(
                    paths[i],
                    "w",
                    driver="GTiff",
                    height=outputs[i].shape[0],
                    width=outputs[i].shape[1],
                    count=1,
                    dtype="float32",
                    crs=crs_list[i],
                    transform=transforms[i],
                ) as dst:
                    dst.write(outputs[i].astype(np.float32), 1)


# ============================================================
# =============== KFOLD CROSS VALIDATION ====================
# ============================================================

def kfold_train(dataset, k_folds, batch_size, model_dir, num_epochs):
    model = UNetLSTM()
    diffusion = GaussianDiffusion(model, timesteps=1000, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    checkpoint_path = os.path.join(model_dir, "model_checkpoint.pth")

    start_epoch, start_fold = load_checkpoint(checkpoint_path, diffusion, optimizer, device=device)

    all_dates = sorted(list({s[2].date() for s in dataset.samples}))
    for fold_idx in range(start_fold, k_folds):
        val_day = all_dates[fold_idx]
        print(f"[Fold {fold_idx}] Validation Day {val_day}")

        train_indices = [i for i, s in enumerate(dataset.samples) if s[2].date() != val_day]
        val_indices = [i for i, s in enumerate(dataset.samples) if s[2].date() == val_day]

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
                                  num_workers=4, collate_fn=custom_collate_fn, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices),
                                num_workers=4, collate_fn=custom_collate_fn, pin_memory=True)

        diffusion = train_diffusion(diffusion, train_loader, val_loader, optimizer,
                                    checkpoint_path, scaler, fold_idx, start_epoch, num_epochs)
        torch.save({
            "model_state_dict": diffusion.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
            "fold": fold_idx,
        }, os.path.join(model_dir, f"diffusion_fold_{fold_idx}.pth"))
        print(f"[Fold {fold_idx}] Saved diffusion model")


# ============================================================
# ======================= MAIN ===============================
# ============================================================

events_file = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_DEM_tiles.csv"
tile_csv = events_file

DEM_mean = np.mean(np.load("/p/vast1/lazin1/UNet_inputs/mean_stds/DEM_Sonoma_mean.npy")).astype(np.float32)
DEM_std = np.mean(np.load("/p/vast1/lazin1/UNet_inputs/mean_stds/DEM_Sonoma_std_dev.npy")).astype(np.float32)
H1_prec_mean = np.mean(np.load("/p/vast1/lazin1/UNet_inputs/mean_stds/H1_prec_mean.npy")).astype(np.float32)
H1_prec_std = np.mean(np.load("/p/vast1/lazin1/UNet_inputs/mean_stds/H1_prec_std_dev.npy")).astype(np.float32)

model_dir = "/p/vast1/lazin1/UNet_trains/Sonoma_diffusion"
os.makedirs(model_dir, exist_ok=True)

if sys.argv[1] == "train":
    event = "event1"
    dem_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma"
    lc_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma"
    flowline_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_flowline/Sonoma"
    precip_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1H_prec/Sonoma/{event}"
    flood_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}"

    seq_len = 5
    start_time = datetime.strptime("2023010901", "%Y%m%d%H")
    end_time = datetime.strptime("2023011123", "%Y%m%d%H")
    all_hours = pd.date_range(start=start_time, end=end_time, freq="h")

    dataset = FloodDataset(event, dem_dir, lc_dir, flowline_dir, precip_dir, flood_dir,
                           tile_csv, all_hours, seq_len,
                           DEM_mean, DEM_std, H1_prec_mean, H1_prec_std,
                           mode="train")

    kfold_train(dataset, k_folds=3, batch_size=16, model_dir=model_dir, num_epochs=20)

elif sys.argv[1] == "test":
    event = "event2"
    fold_id = 0
    model_path = os.path.join(model_dir, f"diffusion_fold_{fold_id}.pth")
    diffusion, _ = load_model(model_path, device=device)

    output_dir = "/p/vast1/lazin1/UNet_Geotiff_output/Sonoma_event2_diffusion_test"
    os.makedirs(output_dir, exist_ok=True)

    dem_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma"
    lc_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma"
    flowline_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_flowline/Sonoma"
    precip_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1H_prec/Sonoma/{event}"
    flood_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}"

    seq_len = 5
    start_time = datetime.strptime("2019022700", "%Y%m%d%H")
    end_time = datetime.strptime("2019030123", "%Y%m%d%H")
    all_hours = pd.date_range(start=start_time, end=end_time, freq="h")

    dataset = FloodDataset
