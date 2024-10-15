from scipy.interpolate import griddata

from src.loader.las_loader import LasLoader
import numpy as np

reference_loader = LasLoader("reference_merged_output.las")
raw_d_loader = LasLoader("raw_D_merged_output.las")
cleaned_loader = LasLoader("cleaned_merged_output.las")

# Load data into DataFrames
df_reference = reference_loader.load_to_dataframe()
df_raw_d = raw_d_loader.load_to_dataframe()
df_cleaned = cleaned_loader.load_to_dataframe()

# Get the min and max coordinates
E_min = min(df_reference['E'].min(), df_raw_d['E'].min(), df_cleaned['E'].min())
E_max = max(df_reference['E'].max(), df_raw_d['E'].max(), df_cleaned['E'].max())
N_min = min(df_reference['N'].min(), df_raw_d['N'].min(), df_cleaned['N'].min())
N_max = max(df_reference['N'].max(), df_raw_d['N'].max(), df_cleaned['N'].max())

# Create grid
num_grid_points = 100  # Adjust as needed
E_grid = np.linspace(E_min, E_max, num_grid_points)
N_grid = np.linspace(N_min, N_max, num_grid_points)
E_mesh, N_mesh = np.meshgrid(E_grid, N_grid)
grid_points = np.column_stack((E_mesh.ravel(), N_mesh.ravel()))

# Interpolate for reference data
points_ref = np.column_stack((df_reference['E'], df_reference['N']))
values_ref = df_reference['h']
h_ref_grid = griddata(points_ref, values_ref, grid_points, method='linear')

# Interpolate for raw D data
points_raw_d = np.column_stack((df_raw_d['E'], df_raw_d['N']))
values_raw_d = df_raw_d['h']
h_raw_d_grid = griddata(points_raw_d, values_raw_d, grid_points, method='linear')

# Interpolate for cleaned data
points_cleaned = np.column_stack((df_cleaned['E'], df_cleaned['N']))
values_cleaned = df_cleaned['h']
h_cleaned_grid = griddata(points_cleaned, values_cleaned, grid_points, method='linear')

# Compute differences
diff_raw_d = h_ref_grid - h_raw_d_grid
diff_cleaned = h_ref_grid - h_cleaned_grid

# Handle NaNs
valid_mask_raw_d = ~np.isnan(diff_raw_d)
diff_raw_d_valid = diff_raw_d[valid_mask_raw_d]
valid_mask_cleaned = ~np.isnan(diff_cleaned)
diff_cleaned_valid = diff_cleaned[valid_mask_cleaned]

# Compute statistics
mean_diff_raw_d = np.mean(diff_raw_d_valid)
rmse_diff_raw_d = np.sqrt(np.mean(diff_raw_d_valid ** 2))

mean_diff_cleaned = np.mean(diff_cleaned_valid)
rmse_diff_cleaned = np.sqrt(np.mean(diff_cleaned_valid ** 2))

print("Statistics for Reference vs Raw D:")
print(f"Mean difference: {mean_diff_raw_d}")
print(f"RMSE: {rmse_diff_raw_d}")

print("\nStatistics for Reference vs Cleaned:")
print(f"Mean difference: {mean_diff_cleaned}")
print(f"RMSE: {rmse_diff_cleaned}")