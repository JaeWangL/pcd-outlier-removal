import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

from loader.las_loader import LasLoader


def create_xz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({'x': df['E'], 'z': df['h']}, index=df.index)


def create_yz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({'y': df['N'], 'z': df['h']}, index=df.index)


def statistical_outlier_removal_df(df: pd.DataFrame, k=20, z_max=2.0) -> pd.Series:
    """
    Applies Statistical Outlier Removal (SOR) to the input DataFrame.
    """
    # Convert DataFrame to numpy array
    points = df.values

    # Find the k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)

    # Exclude the first distance (distance to itself)
    mean_distances = np.mean(distances[:, 1:], axis=1)

    # Calculate the threshold
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    threshold = global_mean + z_max * global_std

    # Filter points
    mask = mean_distances < threshold

    return pd.Series(mask, index=df.index)


def remove_outliers_lof(pcd: pd.DataFrame, contamination: float = 0.01, n_neighbors: int = 20) -> pd.Series:
    """
    Applies Local Outlier Factor to detect outliers in the point cloud data.
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(pcd)
    mask = y_pred != -1  # Inliers are labeled as 1, outliers as -1
    return pd.Series(mask, index=pcd.index)


def visualize_points(original: pd.DataFrame, filtered: pd.DataFrame, title: str) -> None:
    """
    Visualizes the original and filtered point clouds with the same z-axis range.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Determine column names
    x_col = original.columns[0]
    z_col = original.columns[1]

    # Determine the overall z-axis range
    z_min = min(original[z_col].min(), filtered[z_col].min())
    z_max = max(original[z_col].max(), filtered[z_col].max())

    # Plot original points
    scatter1 = ax1.scatter(original[x_col], original[z_col], s=1, c=original[z_col], cmap='viridis')
    ax1.set_title(f'Original ({len(original)} points)')
    ax1.set_xlabel(x_col.upper())
    ax1.set_ylabel(z_col.upper())
    ax1.set_ylim(z_min, z_max)

    # Plot filtered points
    scatter2 = ax2.scatter(filtered[x_col], filtered[z_col], s=1, c=filtered[z_col], cmap='viridis')
    ax2.set_title(f'After Filtering ({len(filtered)} points)')
    ax2.set_xlabel(x_col.upper())
    ax2.set_ylabel(z_col.upper())
    ax2.set_ylim(z_min, z_max)

    # Add colorbars
    plt.colorbar(scatter1, ax=ax1, label=f'{z_col.upper()} value')
    plt.colorbar(scatter2, ax=ax2, label=f'{z_col.upper()} value')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    filename = "Seahawk_231015_230607_00_D.las"
    loader = LasLoader(f"./__rawdata__/{filename}")
    df = loader.load_to_dataframe()

    # Process X-Z plane
    xz_pcd = create_xz_pcd(df)
    xz_mask_sor = statistical_outlier_removal_df(xz_pcd, k=20, z_max=2.0)
    xz_pcd_sor = xz_pcd[xz_mask_sor]
    xz_mask_lof = remove_outliers_lof(xz_pcd_sor, contamination=0.01, n_neighbors=20)
    xz_mask = xz_mask_sor.copy()
    xz_mask[xz_mask_sor] = xz_mask_lof

    # Process Y-Z plane
    yz_pcd = create_yz_pcd(df)
    yz_mask_sor = statistical_outlier_removal_df(yz_pcd, k=20, z_max=2.0)
    yz_pcd_sor = yz_pcd[yz_mask_sor]
    yz_mask_lof = remove_outliers_lof(yz_pcd_sor, contamination=0.01, n_neighbors=20)
    yz_mask = yz_mask_sor.copy()
    yz_mask[yz_mask_sor] = yz_mask_lof

    # Combine masks
    inlier_mask = xz_mask & yz_mask

    df_filtered = df[inlier_mask]

    # Apply LOF on the 3D points
    pcd_3d = df_filtered[['E', 'N', 'h']]
    lof_mask_3d = remove_outliers_lof(pcd_3d, contamination=0.01, n_neighbors=20)
    df_final = df_filtered[lof_mask_3d]

    # Visualize the results
    # Visualize X-Z plane
    xz_original = xz_pcd
    xz_filtered_final = create_xz_pcd(df_final)
    visualize_points(xz_original, xz_filtered_final, 'X-Z Plane')

    # Visualize Y-Z plane
    yz_original = yz_pcd
    yz_filtered_final = create_yz_pcd(df_final)
    visualize_points(yz_original, yz_filtered_final, 'Y-Z Plane')

    # Write the filtered data to a LAS file
    output_file = f"__cleaneddata__/clean_{filename}"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="GCP_id", type=np.uint32))
    header.scales = [0.01, 0.01, 0.01]

    # Create LasData with the header
    las = laspy.LasData(header)

    # Set the coordinates
    las.x = df_final['E'].values
    las.y = df_final['N'].values
    las.z = df_final['h'].values

    # Write the LAS file
    las.write(output_file)
    print(f"Filtered point cloud saved to {output_file}")


if __name__ == "__main__":
    main()
