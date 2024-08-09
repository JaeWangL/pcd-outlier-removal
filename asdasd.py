import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR

from loader.las_loader import LasLoader


def create_xz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a point cloud dataset for x-z relationship.

    Parameters:
        df: pd.DataFrame - The input DataFrame containing 'E' and 'h' columns.

    Returns:
        pd.DataFrame: A DataFrame containing 'x' and 'z' columns.
    """
    return pd.DataFrame({'x': df['E'], 'z': df['h']})


def create_yz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a point cloud dataset for y-z relationship.

    Parameters:
        df: pd.DataFrame - The input DataFrame containing 'N' and 'h' columns.

    Returns:
        pd.DataFrame: A DataFrame containing 'y' and 'z' columns.
    """
    return pd.DataFrame({'y': df['N'], 'z': df['h']})


def create_xy_pcd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a point cloud dataset for x-y relationship.

    Parameters:
        df: pd.DataFrame - The input DataFrame containing 'E' and 'N' columns.

    Returns:
        pd.DataFrame: A DataFrame containing 'x' and 'y' columns.
    """
    return pd.DataFrame({'x': df['E'], 'y': df['N']})


def visualize_pcd(pcd: pd.DataFrame, title: str, xlabel: str, ylabel: str) -> None:
    """
    Visualizes a point cloud dataset as a scatter plot.

    Parameters:
        pcd: pd.DataFrame - The point cloud dataset to visualize.
        title: str - The title of the plot.
        xlabel: str - The label for the x-axis.
        ylabel: str - The label for the y-axis.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(pcd.iloc[:, 0], pcd.iloc[:, 1], alpha=0.5, s=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def remove_outliers(pcd: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """
    Removes outliers from the point cloud dataset using Isolation Forest.

    Parameters:
        pcd: pd.DataFrame - The input point cloud dataset.
        contamination: float - The proportion of outliers in the dataset.

    Returns:
        pd.DataFrame: The point cloud dataset with outliers removed.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(pcd)
    return pcd[outlier_labels == 1]  # Keep only inliers


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
                                model_save_path: str = None) -> pd.DataFrame:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pcd)
    tensor_data = torch.FloatTensor(scaled_data).to(device)

    model = DeepAutoencoder(input_dim=scaled_data.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(tensor_data)
        loss = criterion(outputs, tensor_data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    if model_save_path is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler': scaler
        }, model_save_path)
        print(f"Model saved to {model_save_path}")

    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor_data)
    mse = torch.mean((tensor_data - reconstructed) ** 2, dim=1).cpu().numpy()

    threshold = np.percentile(mse, (1 - contamination) * 100)
    return pcd[mse <= threshold]


def use_saved_model_for_outlier_removal(pcd: pd.DataFrame, model_path: str,
                                        contamination: float = 0.01) -> pd.DataFrame:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load the saved model
    checkpoint = torch.load(model_path)
    model = DeepAutoencoder(input_dim=pcd.shape[1]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']

    # Prepare data
    scaled_data = scaler.transform(pcd)
    tensor_data = torch.FloatTensor(scaled_data).to(device)

    # Use the model for prediction
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor_data)
    mse = torch.mean((tensor_data - reconstructed) ** 2, dim=1).cpu().numpy()

    threshold = np.percentile(mse, (1 - contamination) * 100)
    return pcd[mse <= threshold]


def fine_tune_model(pcd: pd.DataFrame, model_path: str, contamination: float = 0.01, epochs: int = 260) -> pd.DataFrame:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load the saved model
    checkpoint = torch.load(model_path)
    model = DeepAutoencoder(input_dim=pcd.shape[1]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']

    # Prepare data
    scaled_data = scaler.transform(pcd)
    tensor_data = torch.FloatTensor(scaled_data).to(device)

    # Set up for fine-tuning
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(tensor_data)
        loss = criterion(outputs, tensor_data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Fine-tuning Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Save the fine-tuned model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler
    }, model_path.replace('.pth', '_fine_tuned.pth'))
    print(f"Fine-tuned model saved to {model_path.replace('.pth', '_fine_tuned.pth')}")

    # Use the fine-tuned model for prediction
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor_data)
    mse = torch.mean((tensor_data - reconstructed) ** 2, dim=1).cpu().numpy()

    threshold = np.percentile(mse, (1 - contamination) * 100)
    return pcd[mse <= threshold]


def clean_point_cloud(pcd: pd.DataFrame, model_save_path: str = None) -> pd.DataFrame:
    return remove_outliers_autoencoder(pcd, 0.01, 260, model_save_path)


loader = LasLoader("./__rawdata__/Seahawk_231015_223539_00_D.las")
df = loader.load_to_dataframe()

# Create x-z and y-z point cloud datasets
xz_pcd = create_xz_pcd(df)
yz_pcd = create_yz_pcd(df)

"""
xz_pcd_clean = remove_outliers(xz_pcd)
yz_pcd_clean = remove_outliers(yz_pcd)
"""

xz_pcd_clean = clean_point_cloud(xz_pcd, './models/first_xz.pth')
yz_pcd_clean = clean_point_cloud(yz_pcd, './models/first_yz.pth')
"""
xz_pcd_clean = fine_tune_model(xz_pcd, './models/first_xz.pth')
yz_pcd_clean = fine_tune_model(yz_pcd, './models/first_yz.pth')
"""
# Visualize original and cleaned point cloud datasets
visualize_pcd(xz_pcd, "X-Z Point Cloud (Original)", "X (Easting)", "Z (Height)")
visualize_pcd(xz_pcd_clean, "X-Z Point Cloud (Cleaned)", "X (Easting)", "Z (Height)")

visualize_pcd(yz_pcd, "Y-Z Point Cloud (Original)", "Y (Northing)", "Z (Height)")
visualize_pcd(yz_pcd_clean, "Y-Z Point Cloud (Cleaned)", "Y (Northing)", "Z (Height)")

# Print the number of points removed
print(f"X-Z points removed: {len(xz_pcd) - len(xz_pcd_clean)}")
print(f"Y-Z points removed: {len(yz_pcd) - len(yz_pcd_clean)}")

x_clean = xz_pcd_clean['x'].values
y_clean = yz_pcd_clean['y'].values
z_clean = np.mean([xz_pcd_clean['z'].values, yz_pcd_clean['z'].values], axis=0)

output_file = "__cleaneddata__/clean.las"
header = laspy.LasHeader(point_format=3, version="1.2")
header.add_extra_dim(laspy.ExtraBytesParams(name="GCP_id", type=np.uint32))
header.scales = [0.01, 0.01, 0.01]

# Create LasData with the header
las = laspy.LasData(header)

# Set the coordinates
las.x = x_clean
las.y = y_clean
las.z = z_clean

# Write the LAS file
las.write(output_file)
