import logging

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

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

    # Filter cleaned data to include only points where h is between -5 and 5
    z_min, z_max = -5, 5
    cleaned_filtered_df = cleaned_df[(cleaned_df['h'] >= z_min) & (cleaned_df['h'] <= z_max)]

    # Check if there are enough points after filtering
    if cleaned_filtered_df.empty:
        logger.error(f"No points in cleaned data with z between {z_min} and {z_max}")
        return

    # Compute the convex hull of the filtered cleaned data points
    points = cleaned_filtered_df[['E', 'N']].values

    # If there are not enough points for a convex hull, handle accordingly
    if len(points) < 3:
        logger.error("Not enough points to compute a convex hull.")
        return

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Create a shapely polygon of the convex hull
    polygon = Polygon(hull_points)

    # Generate random points within the convex hull
    min_x, min_y, max_x, max_y = polygon.bounds

    num_random_points = 10000  # Adjust as needed
    random_points = []

    logger.info(f"Generating random points within the area where cleaned data z is between {z_min} and {z_max}")

    # Generate random points within the bounding rectangle and select those inside the polygon
    attempts = 0
    max_attempts = num_random_points * 10  # To prevent infinite loop
    while len(random_points) < num_random_points and attempts < max_attempts:
        rand_x = np.random.uniform(min_x, max_x)
        rand_y = np.random.uniform(min_y, max_y)
        random_point = Point(rand_x, rand_y)
        attempts += 1
        if polygon.contains(random_point):
            # Interpolate z value at this point from the cleaned data
            # We need an interpolator for the cleaned data
            cleaned_interp_point = NearestNDInterpolator(cleaned_filtered_df[['E', 'N']].values,
                                                         cleaned_filtered_df['h'].values)
            z_value = cleaned_interp_point([rand_x, rand_y])[0]
            if np.isfinite(z_value) and z_min <= z_value <= z_max:
                random_points.append((rand_x, rand_y))

    if len(random_points) < num_random_points:
        logger.warning(f"Only generated {len(random_points)} random points after {attempts} attempts.")

    random_points = np.array(random_points)

    logger.info(f"Generated {len(random_points)} random points within the specified z range.")

    # Prepare datasets for interpolation
    ref_points = reference_df[['E', 'N']].values
    ref_z = reference_df['h'].values
    rawD_points = raw_D_df[['E', 'N']].values
    rawD_z = raw_D_df['h'].values
    cleaned_points = cleaned_df[['E', 'N']].values
    cleaned_z = cleaned_df['h'].values

    # For large datasets, downsample the data to reduce computation time
    # Adjust the sampling step as necessary
    sample_step = 10  # Increase if still too large
    ref_points_sampled = ref_points[::sample_step]
    ref_z_sampled = ref_z[::sample_step]
    rawD_points_sampled = rawD_points[::sample_step]
    rawD_z_sampled = rawD_z[::sample_step]
    cleaned_points_sampled = cleaned_points[::sample_step]
    cleaned_z_sampled = cleaned_z[::sample_step]

    # Create interpolators using NearestNDInterpolator
    logger.info("Creating interpolators")

    try:
        ref_interp = NearestNDInterpolator(ref_points_sampled, ref_z_sampled)
        rawD_interp = NearestNDInterpolator(rawD_points_sampled, rawD_z_sampled)
        cleaned_interp = NearestNDInterpolator(cleaned_points_sampled, cleaned_z_sampled)
    except Exception as e:
        logger.error(f"Error creating interpolators: {e}")
        return

    # Interpolate z values at the random points
    logger.info("Interpolating z values at random points")

    z_ref = ref_interp(random_points)
    z_rawD = rawD_interp(random_points)
    z_cleaned = cleaned_interp(random_points)

    # Compute differences where interpolated values are valid (not NaN)
    valid_mask_ref_rawD = np.isfinite(z_ref) & np.isfinite(z_rawD)
    diff_rawD = z_ref[valid_mask_ref_rawD] - z_rawD[valid_mask_ref_rawD]
    valid_points_rawD = random_points[valid_mask_ref_rawD]

    valid_mask_ref_cleaned = np.isfinite(z_ref) & np.isfinite(z_cleaned)
    diff_cleaned = z_ref[valid_mask_ref_cleaned] - z_cleaned[valid_mask_ref_cleaned]
    valid_points_cleaned = random_points[valid_mask_ref_cleaned]

    # Compute RMSE and other statistics
    def compute_statistics(differences, label):
        rmse = np.sqrt(np.mean(differences ** 2))
        me = np.mean(differences)
        sd = np.std(differences)
        count = len(differences)
        logger.info(f"{label} - RMSE: {rmse:.3f}, ME: {me:.3f}, SD: {sd:.3f}, Count: {count}")
        print(f"{label} - RMSE: {rmse:.3f}, ME: {me:.3f}, SD: {sd:.3f}, Count: {count}")
        return rmse, me, sd

    rmse_rawD, me_rawD, sd_rawD = compute_statistics(diff_rawD, "Reference vs Raw_D")
    rmse_cleaned, me_cleaned, sd_cleaned = compute_statistics(diff_cleaned, "Reference vs Cleaned")

    # Visualize the differences
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(valid_points_rawD[:, 0], valid_points_rawD[:, 1],
                      c=diff_rawD, s=10, cmap='RdYlBu_r', marker='.')
    plt.colorbar(sc1, label='Height Difference (m)')
    plt.title(f'Difference between Reference and Raw_D Data\nRMSE: {rmse_rawD:.3f} m')
    plt.xlabel('Easting (E)')
    plt.ylabel('Northing (N)')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(valid_points_cleaned[:, 0], valid_points_cleaned[:, 1],
                      c=diff_cleaned, s=10, cmap='RdYlBu_r', marker='.')
    plt.colorbar(sc2, label='Height Difference (m)')
    plt.title(f'Difference between Reference and Cleaned Data\nRMSE: {rmse_cleaned:.3f} m')
    plt.xlabel('Easting (E)')
    plt.ylabel('Northing (N)')
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    # Histograms
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(diff_rawD, bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Height Differences\nReference vs Raw_D')
    plt.xlabel('Height Difference (m)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(diff_cleaned, bins=50, color='salmon', edgecolor='black')
    plt.title('Histogram of Height Differences\nReference vs Cleaned')
    plt.xlabel('Height Difference (m)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    logger.info("Visualization complete.")


if __name__ == "__main__":
    main()