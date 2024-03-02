import os
import pandas as pd
from estimator.pcd_estimator import PCDEstimator
from loader.csv_loader import CsvLoader
from loader.las_loader import LasLoader
from removal.outlier_removal import OutlierRemoval

# Load the reference dataframe once as it remains constant
df_reference = CsvLoader('./__reference__/reference.csv').load_to_dataframe()

# Directory paths
raw_data_dir = './__rawdata__'
target_data_dir = './__testdata__'

# Prepare a list to store results
results = []

# Iterate through all files in the __testdata__ directory
for filename in os.listdir(raw_data_dir):
    if filename.endswith('.las'):
        raw_file_path = os.path.join(raw_data_dir, filename)
        target_file_path = os.path.join(target_data_dir, filename)

        # Ensure the corresponding file exists in the __rawdata__ folder
        if os.path.exists(raw_file_path):
            # Load datasets
            df_raw = LasLoader(raw_file_path).load_to_dataframe()
            df_target = LasLoader(target_file_path).load_to_dataframe()

            # Process and estimate
            estimator = PCDEstimator(df_reference, df_raw, OutlierRemoval(df_raw).main(), df_target)
            mean_diff_raw, mean_diff_mine, mean_diff_target, better_target = estimator.main()

            # Append the results for each file to the results list
            results.append([filename, mean_diff_raw, mean_diff_mine, mean_diff_target, better_target])

# Convert results list to a DataFrame
results_df = pd.DataFrame(results, columns=['filename', 'mean_diff_raw', 'mean_diff_mine', 'mean_diff_target', 'better_target'])

# Write the DataFrame to a CSV file
results_df.to_csv('results.csv', index=False)
