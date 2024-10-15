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

    # Apply LOF on the 3D points
    # pcd_3d = df_filtered[['E', 'N', 'h']]
    # lof_mask_3d = remove_outliers_lof(pcd_3d, contamination=0.01, n_neighbors=20)
    # df_filtered = df_filtered[lof_mask_3d]

    # Now apply SRR after LOF
    print("Applying SRR after LOF...")

    # Prepare dataset for SRR
    class PointCloudDataset(Dataset):
        def __init__(self, dataframe):
            self.data = dataframe[['E', 'N', 'h']].values.astype(np.float32)
            self.indices = dataframe.index.values

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            return torch.from_numpy(sample)

    dataset = PointCloudDataset(df_filtered)

    # Device configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # SRR Parameters
    batch_size = 256
    representation_dim = 32  # Reduced dimensionality
    num_occ_estimators = 3  # Reduced number of estimators
    refinement_iterations = 3  # Reduced iterations
    convergence_threshold = 0.001
    num_epochs = 3  # Reduced epochs

    # Initialize AutoEncoder
    autoencoder = AutoEncoder(input_dim=3, representation_dim=representation_dim).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Initialize variables for SRR
    refined_indices = np.arange(len(dataset))
    prev_loss = float('inf')

    for iteration in range(refinement_iterations):
        print(f"SRR Iteration {iteration + 1}/{refinement_iterations}")

        # Precompute features for the full dataset
        print("Extracting features for the full dataset...")
        full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        full_features = []
        with torch.no_grad():
            for data in full_loader:
                data = data.float().to(device)
                _, z = autoencoder(data)
                feature = z.cpu().numpy()
                full_features.append(feature)
        full_features = np.concatenate(full_features, axis=0)

        # Initialize occ_predictions array
        occ_predictions = np.zeros((len(dataset), num_occ_estimators))

        # Data Refinement
        print("Data Refinement...")
        np.random.shuffle(refined_indices)
        subsets = np.array_split(refined_indices, num_occ_estimators)

        for i, subset_indices in enumerate(subsets):
            # Extract features from subset
            subset_features = full_features[subset_indices]

            # Normalize features
            scaler = StandardScaler()
            subset_features_scaled = scaler.fit_transform(subset_features)
            full_features_scaled = scaler.transform(full_features)

            # Train OCC - Use IsolationForest
            occ = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination='auto',
                random_state=42,
                n_jobs=-1
            ).fit(subset_features_scaled)

            # Predict on all data
            preds = occ.predict(full_features_scaled)
            occ_predictions[:, i] = preds

        # Consensus
        consensus = np.all(occ_predictions == 1, axis=1)
        refined_indices = np.where(consensus)[0]
        print(f"Refined dataset size: {len(refined_indices)}")

        # Update representation learner
        refined_dataset = torch.utils.data.Subset(dataset, refined_indices)
        refined_loader = DataLoader(refined_dataset, batch_size=batch_size, shuffle=True)

        # Self-supervised training (Autoencoder)
        epoch_losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for data in refined_loader:
                data = data.float().to(device)
                reconstructed, _ = autoencoder(data)
                loss = nn.MSELoss()(reconstructed, data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            epoch_loss /= len(refined_loader)
            epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        # Check for convergence
        if abs(prev_loss - epoch_loss) < convergence_threshold:
            print("Convergence achieved.")
            break
        prev_loss = epoch_loss

    # Training Final OCC
    print("Training Final OCC...")
    refined_dataset = torch.utils.data.Subset(dataset, refined_indices)
    refined_loader = DataLoader(refined_dataset, batch_size=batch_size, shuffle=False)

    features = []
    with torch.no_grad():
        for data in refined_loader:
            data = data.float().to(device)
            _, z = autoencoder(data)
            feature = z.cpu().numpy()
            features.append(feature)
    features = np.concatenate(features, axis=0)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Train final OCC (Isolation Forest)
    final_occ = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    final_occ.fit(features)

    # Compute anomaly scores on the data
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    with torch.no_grad():
        for data in full_loader:
            data = data.float().to(device)
            _, z = autoencoder(data)
            feature = z.cpu().numpy()
            features.append(feature)
    features = np.concatenate(features, axis=0)
    features = scaler.transform(features)

    anomaly_labels = final_occ.predict(features)
    # -1 indicates anomalies, 1 indicates normal
    anomaly_mask = anomaly_labels == -1  # anomalies
    normal_mask = anomaly_labels == 1  # normal data

    # Get indices in df_filtered
    df_filtered_indices = dataset.indices
    anomaly_indices_in_df_filtered = df_filtered_indices[anomaly_mask]
    normal_indices_in_df_filtered = df_filtered_indices[normal_mask]

    # Update df_final
    df_srr_filtered = df_filtered.loc[normal_indices_in_df_filtered]

    # Visualize the results
    # Visualize X-Z plane
    xz_original = xz_pcd
    xz_filtered_final = create_xz_pcd(df_srr_filtered)
    visualize_points(xz_original, xz_filtered_final, 'X-Z Plane after SRR')

    # Visualize Y-Z plane
    yz_original = yz_pcd
    yz_filtered_final = create_yz_pcd(df_srr_filtered)
    visualize_points(yz_original, yz_filtered_final, 'Y-Z Plane after SRR')

    # Write the filtered data to a LAS file
    output_file = f"__cleaneddata__/clean_{filename}"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="GCP_id", type=np.uint32))
    header.scales = [0.01, 0.01, 0.01]

    # Create LasData with the header
    las = laspy.LasData(header)

    # Set the coordinates
    las.x = df_srr_filtered['E'].values
    las.y = df_srr_filtered['N'].values
    las.z = df_srr_filtered['h'].values

    # Write the LAS file
    las.write(output_file)
    print(f"Filtered point cloud saved to {output_file}")


if __name__ == "__main__":
    main()
