import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from loader.las_loader import LasLoader


def create_xz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a point cloud dataset for x-z relationship.

    Parameters:
        df: pd.DataFrame - The input DataFrame containing 'E' and 'h' columns.

    Returns:
        pd.DataFrame: A DataFrame containing 'x' and 'z' columns with index from df.
    """
    return pd.DataFrame({'x': df['E'], 'z': df['h']}, index=df.index)


def create_yz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a point cloud dataset for y-z relationship.

    Parameters:
        df: pd.DataFrame - The input DataFrame containing 'N' and 'h' columns.

    Returns:
        pd.DataFrame: A DataFrame containing 'y' and 'z' columns with index from df.
    """
    return pd.DataFrame({'y': df['N'], 'z': df['h']}, index=df.index)


def statistical_outlier_removal_df(df: pd.DataFrame, k=8, z_max=1.0) -> pd.Series:
    """
    Applies Statistical Outlier Removal (SOR) to the input DataFrame.

    Parameters:
        df: pd.DataFrame - Input DataFrame with two columns (any names)
        k: int - Number of nearest neighbors to use
        z_max: float - Maximum z-score for a point to be considered an inlier

    Returns:
        pd.Series: Boolean mask indicating inliers
    """
    if len(df.columns) != 2:
        raise ValueError("Input DataFrame must have exactly two columns")

    # Convert DataFrame to numpy array
    points = df.values

    # Find the k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)

    # Calculate the mean distance to k-nearest neighbors for each point
    mean_distances = np.mean(distances, axis=1)

    # Calculate the threshold
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    threshold = global_mean + z_max * global_std

    # Filter points
    mask = mean_distances < threshold

    return pd.Series(mask, index=df.index)


class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
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
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(8),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(8),
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def remove_outliers_autoencoder(pcd: pd.DataFrame, contamination: float = 0.01, epochs: int = 260,
                                model_save_path: str = None) -> pd.Series:
    import gc

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pcd)
    tensor_data = torch.tensor(scaled_data, dtype=torch.float32, device=device)

    model = DeepAutoencoder(input_dim=scaled_data.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        print(epoch)
        model.train()
        optimizer.zero_grad()
        outputs = model(tensor_data)
        loss = criterion(outputs, tensor_data)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    if model_save_path is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler': scaler
        }, model_save_path)
        print(f"Model saved to {model_save_path}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor_data)
        mse = torch.mean((tensor_data - reconstructed) ** 2, dim=1).cpu().numpy()

    threshold = np.percentile(mse, (1 - contamination) * 100)
    mask = mse <= threshold

    # Explicitly delete variables and collect garbage
    del model
    del optimizer
    del tensor_data
    del outputs
    del loss
    del reconstructed
    del mse
    gc.collect()

    return pd.Series(mask, index=pcd.index)


def visualize_points(original: pd.DataFrame, filtered: pd.DataFrame, title: str) -> None:
    """
    Visualizes the original and filtered point clouds with the same z-axis range.

    Parameters:
        original: pd.DataFrame - Original point cloud DataFrame
        filtered: pd.DataFrame - Filtered point cloud DataFrame
        title: str - Title for the plot
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
    loader = LasLoader("./__rawdata__/Seahawk_231015_223539_00_D.las")
    df = loader.load_to_dataframe()

    # Process Y-Z plane
    yz_pcd = create_yz_pcd(df)
    yz_mask_sor = statistical_outlier_removal_df(yz_pcd)
    yz_pcd_sor = yz_pcd[yz_mask_sor]
    yz_mask_ae = remove_outliers_autoencoder(yz_pcd_sor)
    yz_mask = yz_mask_sor.copy()
    yz_mask.loc[yz_pcd_sor.index] = yz_mask_ae

    # Process X-Z plane
    xz_pcd = create_xz_pcd(df)
    xz_mask_sor = statistical_outlier_removal_df(xz_pcd)
    xz_pcd_sor = xz_pcd[xz_mask_sor]
    xz_mask_ae = remove_outliers_autoencoder(xz_pcd_sor)
    xz_mask = xz_mask_sor.copy()
    xz_mask.loc[xz_pcd_sor.index] = xz_mask_ae

    # Combine masks
    inlier_mask = xz_mask & yz_mask

    df_filtered = df[inlier_mask]

    # Optionally, apply autoencoder on the 3D points
    pcd_3d = df_filtered[['E', 'N', 'h']]
    ae_mask_3d = remove_outliers_autoencoder(pcd_3d)
    df_final = df_filtered[ae_mask_3d]

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
    output_file = "__cleaneddata__/clean.las"
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
