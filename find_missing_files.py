import os
import pandas as pd

csv_f = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined.csv"
combined_df = pd.read_csv(csv_f) 
filenames = combined_df['ras_name'].tolist() 
for file in filenames: #events = [raster_path.split("/")[-1][:-4] 
    str = file.split("crop")[0][:-1]
    # print(file,str)
    if not os.path.exists(f"/p/lustre2/lazin1/30D_prec/{str}/30day_prec_{file}"):
        print(f"/p/lustre2/lazin1/30D_prec/{str}/30day_prec_{file}")
