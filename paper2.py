import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from loader.las_loader import LasLoader


def create_xz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({'x': df['E'], 'z': df['h']}, index=df.index)


def create_yz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({'y': df['N'], 'z': df['h']}, index=df.index)


def statistical_outlier_removal_df(df: pd.DataFrame, k=20, z_max=2.0) -> pd.Series:
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
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(pcd)
    mask = y_pred != -1  # Inliers are labeled as 1, outliers as -1
    return pd.Series(mask, index=pcd.index)


def visualize_points(original: pd.DataFrame, filtered: pd.DataFrame, title: str) -> None:
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


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=3, representation_dim=64):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z


def main():
    filename = "Seahawk_231015_223008_00_D.las"
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
    # Visualize the results
    # Visualize X-Z plane
    xz_original = xz_pcd
    yz_original = yz_pcd
    visualize_points(xz_original, xz_original, 'X-Z ')

    visualize_points(yz_original, yz_original, 'Y-Z')


if __name__ == "__main__":
    main()
