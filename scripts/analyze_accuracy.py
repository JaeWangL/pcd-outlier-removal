import logging

from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

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

    # Now create a random grid based on the extent of "cleaned_merged_output.las"
    # We can get min and max of E and N (which are x and y)
    xmin, xmax = cleaned_df['E'].min(), cleaned_df['E'].max()
    ymin, ymax = cleaned_df['N'].min(), cleaned_df['N'].max()

    # Let's generate N random points within this extent
    N_points = 10000  # Adjust as needed for resolution
    x_random = np.random.uniform(xmin, xmax, N_points)
    y_random = np.random.uniform(ymin, ymax, N_points)

    logger.info(f"Generated {N_points} random grid points within the extent of the cleaned data.")

    # Now we need to interpolate z values at these (x_random, y_random) points from each dataset
    # First, prepare the interpolation for reference data
    reference_points = np.vstack((reference_df['E'], reference_df['N'])).T
    reference_z = reference_df['h']

    # Similarly for raw_D and cleaned data
    raw_D_points = np.vstack((raw_D_df['E'], raw_D_df['N'])).T
    raw_D_z = raw_D_df['h']

    cleaned_points = np.vstack((cleaned_df['E'], cleaned_df['N'])).T
    cleaned_z = cleaned_df['h']

    logger.info("Prepared data points for interpolation.")

    # Use scipy's LinearNDInterpolator for TIN interpolation
    # Create interpolators
    reference_interp = LinearNDInterpolator(reference_points, reference_z)
    raw_D_interp = LinearNDInterpolator(raw_D_points, raw_D_z)
    cleaned_interp = LinearNDInterpolator(cleaned_points, cleaned_z)

    logger.info("Created interpolators for each dataset.")

    # Interpolate z values at the grid points
    grid_points = np.vstack((x_random, y_random)).T
    z_reference = reference_interp(grid_points)
    z_raw_D = raw_D_interp(grid_points)
    z_cleaned = cleaned_interp(grid_points)

    logger.info("Interpolated z values at grid points.")

    # Now calculate differences
    valid_mask_ref_raw_D = np.isfinite(z_reference) & np.isfinite(z_raw_D)
    diff_raw_D = np.full_like(z_reference, np.nan)
    diff_raw_D[valid_mask_ref_raw_D] = z_reference[valid_mask_ref_raw_D] - z_raw_D[valid_mask_ref_raw_D]

    valid_mask_ref_cleaned = np.isfinite(z_reference) & np.isfinite(z_cleaned)
    diff_cleaned = np.full_like(z_reference, np.nan)
    diff_cleaned[valid_mask_ref_cleaned] = z_reference[valid_mask_ref_cleaned] - z_cleaned[valid_mask_ref_cleaned]

    logger.info("Calculated differences between reference and other datasets.")

    # Now, let's visualize the differences
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(x_random[valid_mask_ref_raw_D], y_random[valid_mask_ref_raw_D],
                      c=diff_raw_D[valid_mask_ref_raw_D], s=1, cmap='viridis', marker='.')
    plt.colorbar(sc1, label='Height Difference (m)')
    plt.title('Difference between Reference and Raw_D Data')
    plt.xlabel('Easting (E)')
    plt.ylabel('Northing (N)')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(x_random[valid_mask_ref_cleaned], y_random[valid_mask_ref_cleaned],
                      c=diff_cleaned[valid_mask_ref_cleaned], s=1, cmap='viridis', marker='.')
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