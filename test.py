import os
import numpy as np
import open3d as o3d
from loader.las_loader import LasLoader

# Directory containing the raw .las files
raw_data_dir = './__rawdata__'
all_point_clouds = []

# Iterate through all files in the directory
for filename in os.listdir(raw_data_dir):
    if filename.endswith('.las'):
        raw_file_path = os.path.join(raw_data_dir, filename)
        if os.path.exists(raw_file_path):
            # Load each .las file into a DataFrame
            df_raw = LasLoader(raw_file_path).load_to_dataframe()

            # Convert the DataFrame into an Open3D point cloud format
            # Assuming 'x', 'y', 'z' are columns in your dataframe
            # Adjust column names if they differ
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(df_raw[['E', 'N', 'h']].values)
            all_point_clouds.append(pcd)

# Merge all point clouds into a single one
merged_pcd = all_point_clouds[0]
for pcd in all_point_clouds[1:]:
    merged_pcd += pcd

# Visualize the merged point cloud
o3d.visualization.draw_geometries([merged_pcd])