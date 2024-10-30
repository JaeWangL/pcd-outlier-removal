import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from loader.las_loader import LasLoader  # Adjust import as needed


class DeepAutoEncoder(nn.Module):
    def __init__(self, input_dim=3):
        super(DeepAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x_reconstructed = self.decoder(x)
        return x_reconstructed


def visualize_3d_point_cloud(original: pd.DataFrame, filtered: pd.DataFrame, title: str) -> None:
    fig = plt.figure(figsize=(14, 6))

    # Original point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original['E'], original['N'], original['h'], c=original['h'], cmap='viridis', s=1)
    ax1.set_title(f'Original Point Cloud ({len(original)} points)')
    ax1.set_xlabel('E')
    ax1.set_ylabel('N')
    ax1.set_zlabel('h')

    # Filtered point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(filtered['E'], filtered['N'], filtered['h'], c=filtered['h'], cmap='viridis', s=1)
    ax2.set_title(f'Filtered Point Cloud ({len(filtered)} points)')
    ax2.set_xlabel('E')
    ax2.set_ylabel('N')
    ax2.set_zlabel('h')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    filename = "Seahawk_231015_225040_00_D.las"  # Adjust the filename as needed
    loader = LasLoader(f"./__rawdata__/{filename}")  # Adjust the path as needed
    df = loader.load_to_dataframe()

    # Print the total number of data points
    print(f"Total number of data points: {len(df)}")

    # Option to sample a subset of the data
    # Uncomment the following lines to sample N data points
    # N = 100000  # Adjust N as needed
    # df = df.sample(n=N, random_state=42)
    # print(f"Number of data points after sampling: {len(df)}")

    # Set df_filtered = df (no prior filtering)
    df_filtered = df.copy()  # Copy to avoid modifying the original DataFrame

    # Apply AutoEncoder directly on original data
    print("Applying Deep AutoEncoder on original data...")

    # Prepare dataset
    class PointCloudDataset(Dataset):
        def __init__(self, dataframe):
            self.data = dataframe[['E', 'N', 'h']].values.astype(np.float32)
            # Data normalization
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
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

    # Training Parameters
    batch_size = 1024  # Adjust batch_size considering your hardware capabilities
    num_epochs = 50  # You might reduce this for testing purposes
    learning_rate = 1e-3

    # Initialize AutoEncoder
    autoencoder = DeepAutoEncoder(input_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Data Loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training Loop
    print("Training AutoEncoder...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        autoencoder.train()
        for data in data_loader:
            data = data.to(device)
            reconstructed = autoencoder(data)
            loss = criterion(reconstructed, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")

    # Compute Reconstruction Error
    print("Computing reconstruction errors...")
    autoencoder.eval()
    errors = []
    reconstructed_data = []
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=batch_size):
            data = data.to(device)
            outputs = autoencoder(data)
            loss = torch.mean((outputs - data) ** 2, dim=1)
            errors.extend(loss.cpu().numpy())
            reconstructed_data.append(outputs.cpu().numpy())
    errors = np.array(errors)
    reconstructed_data = np.vstack(reconstructed_data)

    # Determine Threshold
    threshold_percentile = 95  # Adjust this value as needed
    threshold = np.percentile(errors, threshold_percentile)
    print(f"Anomaly detection threshold (at {threshold_percentile}th percentile): {threshold}")

    # Identify normal data
    normal_mask = errors <= threshold
    anomaly_mask = errors > threshold

    # Get indices in df_filtered
    df_filtered_indices = dataset.indices
    normal_indices_in_df_filtered = df_filtered_indices[normal_mask]
    anomaly_indices_in_df_filtered = df_filtered_indices[anomaly_mask]

    # Update df_final
    df_srr_filtered = df_filtered.loc[normal_indices_in_df_filtered]
    print(f"Number of points after filtering: {len(df_srr_filtered)}")

    # Visualize the results
    visualize_3d_point_cloud(df_filtered, df_srr_filtered, '3D Point Cloud Before and After AutoEncoder Filtering')

    # Optionally, visualize reconstruction error distribution
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=100)
    plt.axvline(threshold, color='r', linestyle='--')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.show()

    # Write the filtered data to a LAS file
    output_file = f"__cleaneddata__/clean_{filename}"
    header = laspy.LasHeader(point_format=3, version="1.2")
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