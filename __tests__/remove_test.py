import os

import laspy
import numpy as np

from loader.las_loader import LasLoader
from removal.outlier_removal import OutlierRemoval


def save_dataframe_to_las(dataframe, output_filepath):
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(dataframe[['E', 'N', 'h']].values, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])

    # 2. Create a LasWriter and a point record, then write it
    with laspy.open(output_filepath, mode="w", header=header) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(dataframe.shape[0], header=header)
        point_record.x = dataframe['E']
        point_record.y = dataframe['N']
        point_record.z = dataframe['h']

        writer.write_points(point_record)


raw_data_dir = '../__rawdata__2'
filtered_dir = os.path.join(raw_data_dir, 'filtered')

for filename in os.listdir(raw_data_dir):
    if filename.endswith('.las'):
        raw_file_path = os.path.join(raw_data_dir, filename)
        if os.path.exists(raw_file_path):
            df_raw = LasLoader(raw_file_path).load_to_dataframe()
            outlier_removed = OutlierRemoval(df_raw).main(True)

            filtered_file_path = os.path.join(filtered_dir, filename)
            save_dataframe_to_las(outlier_removed, filtered_file_path)
