import logging

from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import pandas as pd

from src.loader.las_loader import LasLoader
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Paths to the LAS files
    reference_filepath = "../__reference__/reference_merged_output.las"
    raw_D_filepath = "../__rawdata__/raw_D_merged_output.las"
    cleaned_filepath = "../__cleaneddata__/cleaned_merged_output.las"

    # Initialize LasLoader instances for each file
    reference_loader = LasLoader(reference_filepath)
    raw_D_loader = LasLoader(raw_D_filepath)
    cleaned_loader = LasLoader(cleaned_filepath)

    # Load data into DataFrames
    reference_df = reference_loader.load_to_dataframe()
    raw_D_df = raw_D_loader.load_to_dataframe()
    cleaned_df = cleaned_loader.load_to_dataframe()

    # Preprocess data
    def preprocess(df):
        # Remove NaNs
        df = df.dropna(subset=['E', 'N', 'h'])
        # Remove duplicate points based on E and N
        df = df.drop_duplicates(subset=['E', 'N'])
        return df

    reference_df = preprocess(reference_df)
    raw_D_df = preprocess(raw_D_df)
    cleaned_df = preprocess(cleaned_df)

    # Define grid parameters based on the extent of the cleaned data
    grid_size = 10  # Adjust grid size as needed (in the same units as E and N)
    xmin, xmax = cleaned_df['E'].min(), cleaned_df['E'].max()
    ymin, ymax = cleaned_df['N'].min(), cleaned_df['N'].max()
    x_edges = np.arange(xmin, xmax + grid_size, grid_size)
    y_edges = np.arange(ymin, ymax + grid_size, grid_size)
    logger.info(f"Generated grid with cell size {grid_size} units.")

    # Function to compute mean heights within each grid cell
    def compute_grid_means(df, x_edges, y_edges):
        # Assign grid cell indices to each point
        df['x_bin'] = np.digitize(df['E'], x_edges) - 1
        df['y_bin'] = np.digitize(df['N'], y_edges) - 1
        # Group by grid cells and compute mean height
        grid_mean = df.groupby(['x_bin', 'y_bin'], as_index=False)['h'].mean()
        return grid_mean

    # Compute mean heights for each dataset
    reference_grid = compute_grid_means(reference_df, x_edges, y_edges)
    raw_D_grid = compute_grid_means(raw_D_df, x_edges, y_edges)
    cleaned_grid = compute_grid_means(cleaned_df, x_edges, y_edges)

    # Merge datasets on grid cells to compare heights
    df_ref_raw_D = pd.merge(reference_grid, raw_D_grid, on=['x_bin', 'y_bin'], suffixes=('_ref', '_raw_D'))
    df_ref_cleaned = pd.merge(reference_grid, cleaned_grid, on=['x_bin', 'y_bin'], suffixes=('_ref', '_cleaned'))

    # Compute height differences
    df_ref_raw_D['diff'] = df_ref_raw_D['h_ref'] - df_ref_raw_D['h_raw_D']
    df_ref_cleaned['diff'] = df_ref_cleaned['h_ref'] - df_ref_cleaned['h_cleaned']

    # Calculate grid cell centers for plotting
    df_ref_raw_D['x_center'] = x_edges[df_ref_raw_D['x_bin']] + grid_size / 2
    df_ref_raw_D['y_center'] = y_edges[df_ref_raw_D['y_bin']] + grid_size / 2

    df_ref_cleaned['x_center'] = x_edges[df_ref_cleaned['x_bin']] + grid_size / 2
    df_ref_cleaned['y_center'] = y_edges[df_ref_cleaned['y_bin']] + grid_size / 2

    logger.info("Computed differences and grid cell centers.")

    # Visualization
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(df_ref_raw_D['x_center'], df_ref_raw_D['y_center'],
                      c=df_ref_raw_D['diff'], s=10, cmap='viridis', marker='s')
    plt.colorbar(sc1, label='Height Difference (m)')
    plt.title('Difference between Reference and Raw_D Data')
    plt.xlabel('Easting (E)')
    plt.ylabel('Northing (N)')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(df_ref_cleaned['x_center'], df_ref_cleaned['y_center'],
                      c=df_ref_cleaned['diff'], s=10, cmap='viridis', marker='s')
    plt.colorbar(sc2, label='Height Difference (m)')
    plt.title('Difference between Reference and Cleaned Data')
    plt.xlabel('Easting (E)')
    plt.ylabel('Northing (N)')
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    logger.info("Visualization complete.")


if __name__ == "__main__":
    main()