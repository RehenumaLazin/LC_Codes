import pandas as pd
import glob
import os

# Define the directory containing CSV files
csv_dir = "path_to_your_csv_files"  # Change this to your CSV folder
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))  # Find all CSV files

# Define the date range
start_date = pd.Timestamp("2019-02-25")
end_date = pd.Timestamp("2019-03-05")

# Define column names manually since files have no headers
columns = ["Gauge ID", "Year", "Month", "Day", "Flow Data"]

# Initialize a list to store DataFrames
dfs = []

# Process each file
for file in csv_files:
    try:
        # Read CSV without headers and assign column names
        df = pd.read_csv(file, header=None, names=columns)
        print(df)

        # Convert to datetime format
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

        # Filter rows within the required date range
        df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

        # Append to list if data is available
        if not df_filtered.empty:
            dfs.append(df_filtered)
        # print(dfs)
        else:
            print(file)
        

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Merge all valid data
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged data to a new CSV file
    output_file = "merged_streamflow_2019.csv"
    merged_df.to_csv(output_file, index=False)

    print(f"Merged file saved as '{output_file}'.")
else:
    print("No matching data found for the given date range.")
