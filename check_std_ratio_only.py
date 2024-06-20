import os
import pandas as pd
from estimator.pcd_estimator import PCDEstimator
from loader.csv_loader import CsvLoader
from loader.las_loader import LasLoader
from removal.outlier_removal import OutlierRemoval

# Directory paths
raw_data_dir = '__rawdata__1'

# Prepare a list to store results
results = []

# Iterate through all files in the __testdata__ directory
for filename in os.listdir(raw_data_dir):
    if filename.endswith('.las'):
        raw_file_path = os.path.join(raw_data_dir, filename)

        # Ensure the corresponding file exists in the __rawdata__ folder
        if os.path.exists(raw_file_path):
            # Load datasets
            df_raw = LasLoader(raw_file_path).load_to_dataframe()
            OutlierRemoval(df_raw).main(True)
