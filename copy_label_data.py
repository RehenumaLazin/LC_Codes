events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined.csv"  #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'
import pandas as pd

import shutil
tiles_df = pd.read_csv(events_file)
tile_names = tiles_df['ras_name'].tolist()  # Assuming the column is named 'tile_name'

import shutil, errno

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise
for tile_name in tile_names:
    tile_str = tile_name.split("crop")[0][:-1] 
    # os.makedirs(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_{tile_str}",exist_ok=True))
    copyanything(f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{tile_str}", f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_{tile_str}")    
    # label_geotiff_path.append(f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{tile_str}") 