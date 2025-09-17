import os
import glob
import rasterio
from rasterio.merge import merge

def merge_min_rasterio(files, output_path, batch_size=50, temp_dir="temp_merge"):
    os.makedirs(temp_dir, exist_ok=True)
    intermediate_files = []

    # Step 1: Merge in batches
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        temp_out = os.path.join(temp_dir, f"temp_batch_{i//batch_size}.tif")
        print(f"Merging batch {i//batch_size + 1} with {len(batch)} tiles...")

        with rasterio.open(batch[0]) as ref:
            profile = ref.profile

        # Open files in this batch
        with rasterio.Env():
            datasets = [rasterio.open(fp) for fp in batch]
            mosaic, transform = merge(datasets, method='min')
            for ds in datasets:
                ds.close()

        # Update profile and write output
        profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "count": mosaic.shape[0]
        })

        with rasterio.open(temp_out, "w", **profile) as dst:
            dst.write(mosaic)
        intermediate_files.append(temp_out)

    # Step 2: Merge intermediate outputs
    print("Merging final intermediate batches...")
    with rasterio.Env():
        datasets = [rasterio.open(fp) for fp in intermediate_files]
        final_mosaic, final_transform = merge(datasets, method='min')
        for ds in datasets:
            ds.close()

    with rasterio.open(intermediate_files[0]) as ref:
        final_profile = ref.profile
    final_profile.update({
        "height": final_mosaic.shape[1],
        "width": final_mosaic.shape[2],
        "transform": final_transform,
        "count": final_mosaic.shape[0]
    })

    with rasterio.open(output_path, "w", **final_profile) as dst:
        dst.write(final_mosaic)

    # Optional cleanup
    # for f in intermediate_files:
    #     os.remove(f)

# -----------------------------
# MAIN SCRIPT
# -----------------------------

event = f"Sonoma_event2_output_not_normalized_retrain" #"Sonoma_event2_org_output_not_normalized"
output_geo_dir = "/p/vast1/lazin1/UNet_Geotiff_output/" #/p/vast1/lazin1/UNet_Geotiff_output/Sonoma_event2_output_not_normalized_retrain #/p/vast1/lazin1/UNet_Geotiff_output/Sonoma_event2_org_output_not_normalized
event_dir = f"{output_geo_dir}/{event}"
event_str = "event2_2019-02-28_Sonoma" #/p/vast1/lazin1/UNet_Geotiff_output/Sonoma_event2_org_output_not_normalized
prefix = f"{event_str}_crop_"
input_dir = event_dir
method = 'min'
output_file = f"{event_dir}/merged_{event_str}_{method}.tif"

files = sorted(glob.glob(f"{input_dir}/{prefix}*.tif"))
print(f"Found {len(files)} files.")
print(f"Merging into: {output_file}")

merge_min_rasterio(files, output_file, batch_size=50)
