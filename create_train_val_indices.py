import pandas as pd
import json

def create_train_val_indices(csv_file, json_file, output_file):
    """
    Create training and validation indices for each fold based on keywords
    and save them as a JSON file.

    Args:
        csv_file (str): Path to the CSV file containing the list of tile names.
        json_file (str): Path to the JSON file containing keywords for folds.
        output_file (str): Path to save the JSON file with training/validation indices.

    Returns:
        None
    """
    # Load tile names from CSV
    tiles_df = pd.read_csv(csv_file)
    tile_names = tiles_df['ras_name'].tolist()  # Assuming the column is named 'tile_name'

    # Load fold-specific keywords from JSON
    with open(json_file, 'r') as f:
        fold_data = json.load(f)

    # Dictionary to store indices for each fold
    # fold_indices = {}

    for fold, keywords in fold_data.items():
        # Get training and validation keywords
        train_keywords = fold_data['train']
        val_keywords = fold_data['validation']

        # Find indices for training and validation
        train_indices = [
            idx for idx, tile_name in enumerate(tile_names)
            if any(keyword in tile_name for keyword in train_keywords)
        ]
        val_indices = [
            idx for idx, tile_name in enumerate(tile_names)
            if any(keyword in tile_name for keyword in val_keywords)
        ]

        # # Save indices for this fold
        # fold_indices[fold] = {
        #     "train_indices": train_indices,
        #     "val_indices": val_indices
        # }

        # Save fold indices to a JSON file
        with open(output_file, 'w') as f:
            json.dump(
                {
                    "train_indices": train_indices,
                    "val_indices": val_indices
                    }, 
                    f, 
                    indent=4)
        print(f"Training and validation indices saved to {output_file}")


# Example usage
# csv_file_path = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined.csv"  # Path to your CSV file
csv_file_path = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined_reduced_events.csv"
for fold in range(1,9):
    

    json_file_path = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/train_val_strs/fold_{fold}_train_val_indices.json"  # Path to your JSON file
    output_file_path = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/train_val_indices/train_val_indices_fold_{fold}.json"  # Path to save output JSON

    # Generate and save train/val indices
    create_train_val_indices(csv_file_path, json_file_path, output_file_path)
