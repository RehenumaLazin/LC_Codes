import os
import argparse
import glob
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime



import rasterio
from rasterio.enums import Resampling
import glob




# --- Dataset ---
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
import numpy as np
import pandas as pd
import torch
torch.cuda.empty_cache()

from torch.utils.data import Dataset
import rasterio
from torch.utils.data.sampler import SubsetRandomSampler

print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.memory_reserved() / 1e9, "GB reserved")



import multiprocessing
import os
from tqdm import tqdm
import torch
import numpy as np
import rasterio

def worker(args):
    idx, sample, dataset_params = args
    tile, seq_hours, target_hour = sample
    tile_suffix = tile.split("Sonoma")[-1]
    event = "event2"#tile.split("/")[-1].split("_")[0]

    dem_path = os.path.join(dataset_params['dem_dir'], f"Sonoma{tile_suffix}")
    # print(dem_path, os.path.exists(dem_path))
    lc_path = os.path.join(dataset_params['lc_dir'], f"LC_Sonoma{tile_suffix}")
    # print(lc_path, os.path.exists(lc_path))

    def read_raster(path):
        with rasterio.open(path) as src:
            array = src.read(1)
        return array.astype(np.float32)

    try:
        dem = read_raster(dem_path)
        lc = read_raster(lc_path)

        dem_norm = (dem - dataset_params['DEM_mean']) / dataset_params['DEM_std']
        lc_binary = np.where(lc == 11, 1, 0)

        precip_seq = []
        for h in seq_hours:
            p_path = os.path.join(dataset_params['precip_dir'], f"1H_prec_{event}_{h.strftime('%Y%m%d%H')}_Sonoma{tile_suffix}")
            # print(p_path, os.path.exists(p_path))
            p = read_raster(p_path)
            p_norm = (p - dataset_params['precip_mean']) / dataset_params['precip_std']
            precip_seq.append(p_norm)
        precip_seq = np.stack(precip_seq, axis=0)

        inputs = np.concatenate([
            dem_norm[np.newaxis, :, :],
            lc_binary[np.newaxis, :, :],
            precip_seq
        ], axis=0).astype(np.float32)

        label_path = os.path.join(dataset_params['flood_dir'], f"{event}_{target_hour.strftime('%Y%m%d%H')}_Sonoma{tile_suffix}")
        # print(label_path, os.path.exists(label_path))
        label = read_raster(label_path)

        if dataset_params['flood_mean'] is not None and dataset_params['flood_std'] is not None:
            label = (label - dataset_params['flood_mean']) / dataset_params['flood_std']

        return torch.tensor(inputs), torch.tensor(label).unsqueeze(0)
    except Exception as e:
        return None

# This function allows parallel processing of dataset samples
def process_samples_in_parallel(samples, dataset_params, num_workers=4):
    args = [(i, sample, dataset_params) for i, sample in enumerate(samples)]
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(worker, args), total=len(samples)))
    return [r for r in results if r is not None]


import torch
from torch.utils.data import Dataset
import numpy as np
import os
import rasterio
import pandas as pd

class FloodDataset(Dataset):
    def __init__(self, dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, hours, seq_len,
                 DEM_mean, DEM_std, precip_mean, precip_std,
                 flood_mean=None, flood_std=None, return_metadata=False, mode='train'):
        
        self.dem_dir = dem_dir
        self.lc_dir = lc_dir
        self.precip_dir = precip_dir
        self.flood_dir = flood_dir
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

        self.tiles = pd.read_csv(tile_csv)['ras_name'].tolist()
        self.samples = self._generate_samples()

        # # Cache static inputs to reduce disk I/O
        # self.static_inputs = {}
        # for tile in self.tiles:
        #     self.static_inputs[tile] = self._load_static_inputs(tile)

    def _generate_samples(self):
        samples = []
        for tile in self.tiles:
            event = "event2"
            for i in range(self.seq_len - 1, len(self.hours)):
                seq_hours = self.hours[i - self.seq_len + 1:i + 1]
                target_hour = self.hours[i]
                # if all(os.path.exists(os.path.join(self.precip_dir, f"1H_prec_{event}_{h.strftime('%Y%m%d%H')}_{tile}")) for h in seq_hours) and \
                #    os.path.exists(os.path.join(self.flood_dir, f"{event}_{target_hour.strftime('%Y%m%d%H')}_{tile}")):
                #     samples.append((tile, seq_hours, target_hour))
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

        dem, _, _ = self._read_raster(dem_path)
        lc, _, _ = self._read_raster(lc_path)

        dem_norm = (dem - self.DEM_mean) / self.DEM_std
        lc_binary = np.where(lc == 11, 1, 0).astype(np.float32)

        # Shape: (2, H, W)
        static_input = np.stack([dem_norm, lc_binary], axis=0)
        return torch.from_numpy(static_input)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tile, seq_hours, target_hour = self.samples[idx]
        event = "event2"
        static_input = self._load_static_inputs(tile)

        # static_input = self.static_inputs[tile]  # Already (2, H, W)

        # Load precipitation sequence (T, H, W)
        precip_seq = []
        for h in seq_hours:
            p_path = os.path.join(self.precip_dir, f"1H_prec_{event}_{h.strftime('%Y%m%d%H')}_{tile}")
            p, _, _ = self._read_raster(p_path)
            p_norm = (p - self.precip_mean) / self.precip_std
            precip_seq.append(p_norm)
        precip_seq = np.stack(precip_seq, axis=0).astype(np.float32)
        precip_tensor = torch.from_numpy(precip_seq)

        # Load label
        label_path = os.path.join(self.flood_dir, f"{event}_{target_hour.strftime('%Y%m%d%H')}_{tile}")
        label, transform, crs = self._read_raster(label_path)

        if self.flood_mean is not None and self.flood_std is not None and self.mode == 'train':
            label = (label - self.flood_mean) / self.flood_std

        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        if self.return_metadata:
            return static_input, precip_tensor, label_tensor, transform, crs
        return static_input, precip_tensor, label_tensor


class StaticFloodDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
    
    


def custom_collate_fn(batch):
    # print(len(batch[0]))
    """
    Custom collate function for training and testing.

    For training:
        Returns only inputs and labels.

    For testing:
        Returns inputs, transforms, and CRS metadata.
    """
    if len(batch[0]) == 2:  # Training mode
        inputs = torch.stack([item[0] for item in batch])  # Stack inputs
        labels = torch.stack([item[1] for item in batch])  # Stack labels
        return inputs, labels
    elif len(batch[0]) == 4:  # Testing mode
        inputs = torch.stack([item[0] for item in batch])  # Stack inputs
        transforms = [item[1] for item in batch]  # Extract transforms
        crs_list = [item[2] for item in batch]  # Extract CRS
        return inputs, transforms, crs_list

def MSE(pred, target): #(label - np.mean(FD_mean)) / np.mean(FD_std)
    mse = F.mse_loss(pred, target, reduction='mean').item()

    
    return mse



    
    

# event_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/event2"
events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_DEM_tiles.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_hourly_event2_reduced.csv"
tile_csv = events_file

model_dir = f"/p/vast1/lazin1/UNet_trains/Sonoma_event2_output_prec_dem_LC_HOURLY_UNET_LSTM"
os.makedirs(model_dir, exist_ok=True)



# tiles_df = pd.read_csv(events_file)
# tile_names = tiles_df['ras_name'].tolist()


dem_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma"
lc_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma"
# flood_paths.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}/{tile_name}")

precip_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/1H_prec/Sonoma/event2"
flood_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/event2"

# --- Time setup ---
sequence_length = 10
start_time = datetime.strptime("2019022701", "%Y%m%d%H")
end_time = datetime.strptime("2019030123", "%Y%m%d%H")
all_hours = pd.date_range(start=start_time, end=end_time, freq='h')

DEM_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/DEM_Sonoma_mean.npy"
DEM_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/DEM_Sonoma_std_dev.npy"

DEM_mean = np.mean(np.load(DEM_mean_file))
DEM_std = np.mean(np.load(DEM_std_file))

flow_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/Streamflow_Sonoma_mean.npy"
flow_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/Streamflow_Sonoma_std_dev.npy"

flow_mean = np.mean(np.load(flow_mean_file))
flow_std = np.mean(np.load(flow_std_file))

H1_prec_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/H1_prec_mean.npy"
H1_prec_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/H1_prec_std_dev.npy"

H1_prec_mean = np.mean(np.load(H1_prec_mean_file))
H1_prec_std = np.mean(np.load(H1_prec_std_file))

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # --- Dataset creation ---
    # dataset = FloodDataset(dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, all_hours, sequence_length,
    #                 DEM_mean, DEM_std, H1_prec_mean, H1_prec_std,
    #                 flood_mean=None, flood_std=None, return_metadata=False, mode='train')
    # main()


    
if sys.argv[1] == "dataset":
       # Prepare full dataset once
    dataset = FloodDataset(dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, all_hours, sequence_length,
                           DEM_mean, DEM_std, H1_prec_mean, H1_prec_std,
                           flood_mean=None, flood_std=None, return_metadata=False, mode='train')
    
    
    samples = dataset.samples
    
    
    dataset_params = {
    'dem_dir': dataset.dem_dir,
    'lc_dir': dataset.lc_dir,
    'precip_dir': dataset.precip_dir,
    'flood_dir': dataset.flood_dir,
    'DEM_mean': dataset.DEM_mean,
    'DEM_std': dataset.DEM_std,
    'precip_mean': dataset.precip_mean,
    'precip_std': dataset.precip_std,
    'flood_mean': dataset.flood_mean,
    'flood_std': dataset.flood_std
    }
    
    results = process_samples_in_parallel(samples, dataset_params, num_workers=1)
    
    # all_data = [dataset[i] for i in tqdm(range(len(dataset)))]
    torch.save(results, f"{model_dir}/flood_dataset_all.pt")
    print(f"Full dataset saved to {model_dir}/flood_dataset_all.pt")
    
    print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
    print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
    
    
    dataset = StaticFloodDataset(f"{model_dir}/flood_dataset_all.pt")
    print(f"Full dataset loaded from {model_dir}/flood_dataset_all.pt")




# python script.py train --start_epoch 0 --num_epochs 10
# python script.py test --checkpoint model_epoch_10.pt

