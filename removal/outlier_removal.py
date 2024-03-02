from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from sklearn.cluster import DBSCAN

from config import DEBUG
from visualizer.pcd_visualizer import PCDVisualizer

pd.set_option('display.max_rows', 100)


class OutlierRemoval:
    def __init__(self, df_original: pd.DataFrame, k: int = 20) -> None:
        """
        Initializes the OutlierRemoval class with the original LiDAR data and parameters for outlier detection.

        Parameters:
            df_original: pd.DataFrame - The original LiDAR data.
            k: int - The number of neighbors to consider for density estimates and outlier detection.
        """
        self.k = k
        self.df_original = df_original
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(df_original[['E', 'N', 'h']].values)

    def main(self, sor_only: bool = False) -> pd.DataFrame:
        """
        Main function to remove outliers from the LiDAR data. It first removes isolated outliers and then clustered outliers.

        Returns:
            pd.DataFrame: The DataFrame after removing both isolated and clustered outliers.
        """
        removed_isolated_outliers = self._remove_isolated_outliers()
        if sor_only:
            return removed_isolated_outliers

        removed_clusterd_outliers = self._remove_clustered_outliers(removed_isolated_outliers)

        if DEBUG:
            PCDVisualizer(removed_isolated_outliers, "SOR Only").visualize()
            PCDVisualizer(removed_clusterd_outliers, "With DBSCAN").visualize()

        return removed_clusterd_outliers

    def _estimate_std_ratio(self) -> float:
        """
        Estimates the standard deviation ratio (std_ratio) based on the distance to the k-th nearest neighbor,
        which reflects the relative variability in point density.

        Returns:
            float: The calculated standard deviation ratio used to adjust the threshold for statistical outlier removal.
        """
        # Estimate local density and set std_ratio based on that
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(self.df_original[['E', 'N', 'h']])
        distances, _ = nbrs.kneighbors(self.df_original[['E', 'N', 'h']])

        # Use the distance to the k-th nearest neighbor as a measure of local density
        kth_distances = distances[:, -1]  # Get the k-th distance for each point

        # Calculate a suitable std_ratio based on the spread of k-th distances
        std_dev = np.std(kth_distances)
        mean_dist = np.mean(kth_distances)

        # Set a threshold based on the standard deviation; you might adjust this formula based on your data
        std_ratio = std_dev / mean_dist
        print(f"Calculated SOR's std_ratio: {std_ratio}")
        return std_ratio

    def _estimate_dbscan_parameters(self, df: pd.DataFrame) -> Tuple[float, int]:
        """
        Estimates DBSCAN parameters 'eps' and 'min_samples' based on the local density derived from the LiDAR data.

        Parameters:
            df: pd.DataFrame - The DataFrame for which to estimate DBSCAN parameters.

        Returns:
            float: The estimated 'eps' value for DBSCAN, based on local point densities.
            int: The estimated 'min_samples' value for DBSCAN, based on variability in local densities.
        """
        # Fit NearestNeighbors and find distances to the k-th nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(df[['E', 'N', 'h']])
        distances, _ = nbrs.kneighbors(df[['E', 'N', 'h']])

        # Use the mean distance to the k-th nearest neighbor as base for eps
        kth_distances = distances[:, -1]  # Get the k-th distance for each point
        eps = np.mean(kth_distances) * 1.5  # Increase mean by 50% to cover local neighborhood

        # Calculate variability in local densities
        std_dev = np.std(kth_distances)
        mean_dist = np.mean(kth_distances)
        std_ratio = std_dev / mean_dist

        # Adjust min_samples based on std_ratio
        # Higher std_ratio indicates more variability, suggesting a more conservative min_samples
        if std_ratio < 0.1:  # Low variability
            min_samples = 5
        elif std_ratio < 0.2:  # Moderate variability
            min_samples = 8
        else:  # High variability
            min_samples = 12  # Increase min_samples to reduce sensitivity to noise
        print(f"Calculated DBSCAN's eps: {eps}, min_samples: {min_samples}")
        return eps, min_samples

    def _remove_isolated_outliers(self) -> pd.DataFrame:
        """
        Removes isolated outliers from the LiDAR data using the Statistical Outlier Removal (SOR) method.

        Returns:
            pd.DataFrame: The DataFrame after removing isolated outliers.
        """
        std_ratio = self._estimate_std_ratio()
        # nb_neighbors specifies how many nearest neighbors are used in the mean distance calculation.
        # You might want to adjust this based on your data's density.
        nb_neighbors = max(10, self.k)  # This is just an example value

        # Applying Statistical Outlier Removal
        cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                      std_ratio=std_ratio)

        # Filtering the original point cloud based on the indices returned by the SOR filter
        filtered_pcd = self.pcd.select_by_index(ind)

        # Convert the filtered point cloud back to DataFrame
        filtered_points = np.asarray(filtered_pcd.points)
        df_filtered = pd.DataFrame(filtered_points, columns=['E', 'N', 'h'])
        return df_filtered

    def _remove_clustered_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes normals and curvature for each point in the DataFrame using k-nearest neighbors.
        """
        points = df[['E', 'N', 'h']].values  # Convert DataFrame to numpy array
        eps, min_samples = self._estimate_dbscan_parameters(df)
        outlier_indices, inlier_indices = self._detect_clustered_outliers(points, eps=eps,
                                                                                         min_samples=min_samples)

        cleaned_df = df.iloc[inlier_indices].reset_index(drop=True)
        return cleaned_df

    def _detect_clustered_outliers(self, points: np.ndarray, eps=3.0, min_samples=8):
        """
        Detects outliers in the LiDAR data based on density and clustering size using DBSCAN.

        Parameters:
            points: np.ndarray - Array of point coordinates from the LiDAR data.
            eps: float - The maximum distance between two points for one to be considered as in the neighborhood of the other.
            min_samples: int - The number of samples in a neighborhood for a point to be considered a core point.

        Returns:
            list: Indices of outlier points.
            list: Indices of inlier points.
        """
        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        # Initialize lists to hold identified outliers and inliers
        outlier_indices = []
        inlier_indices = []

        # Get indices of core samples for density estimation
        core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True

        # Analyze each cluster (excluding noise)
        for label in set(labels):
            if label == -1:
                outlier_indices.extend(np.where(labels == label)[0])
                continue  # Skip noise points

            # Indices of points in the current cluster
            cluster_indices = np.where(labels == label)[0]

            # If the cluster is too small, consider it as outliers
            if len(cluster_indices) < min_samples:
                outlier_indices.extend(cluster_indices)
            else:
                # Calculate the density of the cluster
                core_indices = cluster_indices[core_samples_mask[cluster_indices]]
                if len(core_indices) < min_samples:  # Adjust this threshold as necessary
                    outlier_indices.extend(cluster_indices)
                else:
                    inlier_indices.extend(cluster_indices)

        return outlier_indices, inlier_indices
