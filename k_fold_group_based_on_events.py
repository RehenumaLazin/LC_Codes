from sklearn.model_selection import KFold
import json
import pandas as pd

# Shortened example list of filenames
# file_names = [
#     'flood_WM_S1A_IW_GRDH_1SDV_20180919T231415_20180919T231440_023774_0297CC_374E', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20180826T231414_20180826T231439_023424_028C94_1E26', 
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20180907T231415_20180907T231440_023599_02922C_3340', 
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20181001T231415_20181001T231440_023949_029D80_240D',
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20181013T231416_20181013T231441_024124_02A33B_A10C',
#     'flood_WM_S1A_IW_GRDH_1SDV_20180919T231350_20180919T231415_023774_0297CC_B86C', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20180814T231349_20180814T231414_023249_0286F7_8F75', 
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20180826T231349_20180826T231414_023424_028C94_F0B9', 
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20180907T231350_20180907T231415_023599_02922C_5F84',
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20181013T231351_20181013T231416_024124_02A33B_4653', 
#     'flood_WM_S1A_IW_GRDH_1SDV_20170829T002645_20170829T002710_018131_01E74D_3220', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20170724T002643_20170724T002708_017606_01D755_602F',
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20170805T002644_20170805T002710_017781_01DCB4_8FDF',
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20170910T002645_20170910T002710_018306_01ECAF_2E7C', 
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20171004T002646_20171004T002711_018656_01F76A_2CC8', 
#     'flood_WM_S1A_IW_GRDH_1SDV_20170829T002735_20170829T002800_018131_01E74D_B8C4', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20170712T002732_20170712T002757_017431_01D208_0517', 
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20170910T002735_20170910T002800_018306_01ECAF_7C03', 
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20171004T002736_20171004T002801_018656_01F76A_9291',
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20171016T002736_20171016T002801_018831_01FCC0_40B3', 
#     'flood_WM_S1A_IW_GRDH_1SDV_20170829T002620_20170829T002645_018131_01E74D_D734',
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20170724T002618_20170724T002643_017606_01D755_0499', 
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20170805T002619_20170805T002644_017781_01DCB4_1716', 
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20171004T002621_20171004T002646_018656_01F76A_595C',
#     'flood_WM_S1A_IW_GRDH_1SDV_20170831T001029_20170831T001054_018160_01E82D_0776', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20170807T001028_20170807T001053_017810_01DD99_F836', 
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20170819T001029_20170819T001054_017985_01E2E8_F302',
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20170912T001029_20170912T001054_018335_01EDA6_7CF2', 
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20170924T001030_20170924T001055_018510_01F303_7AF2',
#     'flood_WM_S1A_IW_GRDH_1SDV_20170831T001004_20170831T001029_018160_01E82D_9366', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20170807T001003_20170807T001028_017810_01DD99_078C', 
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20170819T001004_20170819T001029_017985_01E2E8_5C42', 
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20170912T001004_20170912T001029_018335_01EDA6_2AA9', 
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20170924T001005_20170924T001030_018510_01F303_F9CE', 
#     'flood_WM_S1A_IW_GRDH_1SDV_20190617T000326_20190617T000351_027712_0320C9_3AC6', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20181113T000325_20181113T000350_024562_02B230_0AA0', 
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20181125T000325_20181125T000350_024737_02B8A0_CF52', 
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20181207T000324_20181207T000349_024912_02BE7D_6A7C', 
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20181231T000324_20181231T000349_025262_02CB2C_DF4A',
#     'flood_WM_S1A_IW_GRDH_1SDV_20190617T000236_20190617T000301_027712_0320C9_5E5F', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20181219T000234_20181219T000259_025087_02C4D5_CDFD', 
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20190205T000232_20190205T000257_025787_02DE35_72D4', 
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20190217T000232_20190217T000257_025962_02E467_799D', 
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20190325T000232_20190325T000257_026487_02F77B_0433', 
#     'flood_WM_S1A_IW_GRDH_1SDV_20190617T000301_20190617T000326_027712_0320C9_9D85', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20181207T000259_20181207T000324_024912_02BE7D_E358',
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20181219T000259_20181219T000324_025087_02C4D5_16A4',
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20190205T000257_20190205T000322_025787_02DE35_8A91', 
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20190217T000257_20190217T000322_025962_02E467_DF07', 
#     'flood_WM_S1A_IW_GRDH_1SDV_20190617T000441_20190617T000506_027712_0320C9_42BF', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20181207T000439_20181207T000504_024912_02BE7D_3B80',
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20181219T000439_20181219T000504_025087_02C4D5_A60D', 
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20190112T000438_20190112T000503_025437_02D177_E563', 
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20190124T000438_20190124T000503_025612_02D7DA_723B', 
#     'flood_WM_S1A_IW_GRDH_1SDV_20190617T000416_20190617T000441_027712_0320C9_B9D7', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20190112T000413_20190112T000438_025437_02D177_6C6F', 
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20190205T000412_20190205T000437_025787_02DE35_7FFF', 
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20190217T000412_20190217T000437_025962_02E467_43D4', 
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20190301T000412_20190301T000437_026137_02EAA6_2A9D', 
#     'flood_WM_S1A_IW_GRDH_1SDV_20190617T000351_20190617T000416_027712_0320C9_C310', 
#     'non_flood_WM1_S1A_IW_GRDH_1SDV_20181219T000349_20181219T000414_025087_02C4D5_A4B3',
#     'non_flood_WM2_S1A_IW_GRDH_1SDV_20181231T000349_20181231T000414_025262_02CB2C_C2A6', 
#     'non_flood_WM3_S1A_IW_GRDH_1SDV_20190112T000348_20190112T000413_025437_02D177_13B6',
#     'non_flood_WM4_S1A_IW_GRDH_1SDV_20190205T000347_20190205T000412_025787_02DE35_20FB']


events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined_reduced_events_files.csv'
combined_df = pd.read_csv(events_file, header=None) 
file_names=[]
for raster_path in combined_df[0]: #events = [raster_path.split("/")[-1][:-4] 
    event = raster_path.split("/")[-1][:-4]

    file_names.append(event)  # output shapefile path





# Separate "flood" and "non_flood" samples
flood_samples = [name for name in file_names if name.startswith("flood")]
non_flood_samples = [name for name in file_names if name.startswith("non_flood")]

# Define number of splits
num_splits = 8

# Initialize K-Fold
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

# Create splits ensuring 2 "flood" samples in each validation fold
folds = {}
for fold, (flood_train_idx, flood_val_idx) in enumerate(kf.split(flood_samples), 1):
    flood_train = [flood_samples[i] for i in flood_train_idx]
    flood_val = [flood_samples[i] for i in flood_val_idx]

    # Add non-flood samples for balanced training and validation
    non_flood_kf = KFold(n_splits=num_splits, shuffle=True, random_state=42 + fold)
    non_flood_train_idx, non_flood_val_idx = next(non_flood_kf.split(non_flood_samples))
    non_flood_train = [non_flood_samples[i] for i in non_flood_train_idx]
    non_flood_val = [non_flood_samples[i] for i in non_flood_val_idx]

    # Combine flood and non-flood splits
    folds[fold] = {
        "train": flood_train + non_flood_train,
        "validation": flood_val + non_flood_val,
        "train_indices": flood_train_idx.tolist() + non_flood_train_idx.tolist(),
        "val_indices": flood_val_idx.tolist() + non_flood_val_idx.tolist(),

    }

# Print results
for fold, data in folds.items():
    print(f"Fold {fold}:")
    print(f"  Train: {len(data['train'])} samples")
    print(f"  Validation: {len(data['validation'])} samples")
    print(f"  Flood in Validation: {[name for name in data['validation'] if name.startswith('flood')]}")
    print()
    
# Save folds to JSON files
for fold, data in folds.items():
    with open(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/train_val_strs/fold_{fold}_train_val_indices.json", "w") as f:
        json.dump(
            {
                "train": data["train"],
                "validation": data["validation"],
                # "train_indices": data["train_indices"],
                # "val_indices": data["val_indices"]
                

            },
            f,
            indent=4,
        )
    print(f"Fold {fold} saved.")