import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Added for LR scheduling

# Assuming LasLoader exists and works as before
from loader.las_loader import LasLoader


def create_xz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns exist before accessing
    if 'E' not in df.columns or 'h' not in df.columns:
        raise ValueError("DataFrame must contain 'E' and 'h' columns")
    return pd.DataFrame({'x': df['E'], 'z': df['h']}, index=df.index)


def create_yz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns exist before accessing
    if 'N' not in df.columns or 'h' not in df.columns:
        raise ValueError("DataFrame must contain 'N' and 'h' columns")
    return pd.DataFrame({'y': df['N'], 'z': df['h']}, index=df.index)


def statistical_outlier_removal_df(df: pd.DataFrame, k=20, z_max=2.0) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    points = df.values
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(points)  # Include point itself
    distances, _ = nbrs.kneighbors(points)
    # Exclude the first distance (distance to itself) if k+1 neighbors were found
    if distances.shape[1] > 1:
        mean_distances = np.mean(distances[:, 1:], axis=1)
    else:  # Handle cases with very few points
        mean_distances = np.zeros(len(df))  # Or handle as desired

    # Check for zero standard deviation
    if len(mean_distances) > 1:
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        if global_std == 0:  # Avoid division by zero if all distances are the same
            threshold = global_mean
        else:
            threshold = global_mean + z_max * global_std
        mask = mean_distances < threshold
    else:  # Handle single point case
        mask = np.ones(len(df), dtype=bool)

    return pd.Series(mask, index=df.index)


def remove_outliers_lof(pcd: pd.DataFrame, contamination: float = 0.01, n_neighbors: int = 20) -> pd.Series:
    if pcd.empty or len(pcd) <= n_neighbors:  # LOF requires more points than neighbors
        return pd.Series(dtype=bool)
    # Consider scaling *before* LOF, although LOF is less sensitive than KNN to scale
    # scaler = StandardScaler()
    # pcd_scaled = scaler.fit_transform(pcd.values)
    # lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
    # y_pred = lof.fit_predict(pcd_scaled)

    # Using unscaled data as in original code
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
    y_pred = lof.fit_predict(pcd.values)
    mask = y_pred != -1
    return pd.Series(mask, index=pcd.index)


def visualize_points(original: pd.DataFrame, filtered: pd.DataFrame, title: str) -> None:
    if original.empty and filtered.empty:
        print(f"Skipping visualization for '{title}': Both DataFrames are empty.")
        return
    if original.empty:
        print(f"Warning for '{title}': Original DataFrame is empty.")
        # Optionally plot only filtered if desired
    if filtered.empty:
        print(f"Warning for '{title}': Filtered DataFrame is empty.")
        # Optionally plot only original if desired

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # Share Y axis
    x_col = original.columns[0] if not original.empty else filtered.columns[0]
    z_col = original.columns[1] if not original.empty else filtered.columns[1]

    # Determine the overall z-axis range robustly
    z_min_orig = original[z_col].min() if not original.empty else np.inf
    z_max_orig = original[z_col].max() if not original.empty else -np.inf
    z_min_filt = filtered[z_col].min() if not filtered.empty else np.inf
    z_max_filt = filtered[z_col].max() if not filtered.empty else -np.inf

    z_min = min(z_min_orig, z_min_filt)
    z_max = max(z_max_orig, z_max_filt)

    # Plot original points
    if not original.empty:
        scatter1 = ax1.scatter(original[x_col], original[z_col], s=1, c=original[z_col], cmap='viridis', vmin=z_min,
                               vmax=z_max)
        ax1.set_title(f'Original ({len(original)} points)')
        ax1.set_xlabel(x_col.upper())
        ax1.set_ylabel(z_col.upper())
        # ax1.set_ylim(z_min, z_max) # Set by sharey
        plt.colorbar(scatter1, ax=ax1, label=f'{z_col.upper()} value')
    else:
        ax1.set_title('Original (0 points)')
        ax1.set_xlabel(x_col.upper())
        ax1.set_ylabel(z_col.upper())

    # Plot filtered points
    if not filtered.empty:
        scatter2 = ax2.scatter(filtered[x_col], filtered[z_col], s=1, c=filtered[z_col], cmap='viridis', vmin=z_min,
                               vmax=z_max)
        ax2.set_title(f'After Filtering ({len(filtered)} points)')
        ax2.set_xlabel(x_col.upper())
        # ax2.set_ylabel(z_col.upper()) # Set by sharey
        # ax2.set_ylim(z_min, z_max) # Set by sharey
        plt.colorbar(scatter2, ax=ax2, label=f'{z_col.upper()} value')
    else:
        ax2.set_title('After Filtering (0 points)')
        ax2.set_xlabel(x_col.upper())

    # Ensure common y-lim if sharey=True didn't catch edge cases (like one empty df)
    if not np.isinf(z_min) and not np.isinf(z_max):
        ax1.set_ylim(z_min, z_max)
        ax2.set_ylim(z_min, z_max)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.show()


# --- Improved AutoEncoder ---
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=3, representation_dim=32):  # Make rep_dim tunable
        super(AutoEncoder, self).__init__()
        # Simpler Architecture with BatchNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # Added BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # Added BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, representation_dim)  # To representation space
        )
        self.decoder = nn.Sequential(
            nn.Linear(representation_dim, 64),
            nn.BatchNorm1d(64),  # Added BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),  # Added BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, input_dim)  # To original dimension
            # No activation/norm after last layer for reconstruction
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z


# --- Point Cloud Dataset (No changes needed here) ---
class PointCloudDataset(Dataset):
    def __init__(self, dataframe):
        # Ensure data is float32 for PyTorch
        dataframe_values = dataframe.values.astype(np.float32)
        self.scaler = StandardScaler()
        # Fit scaler only on the data passed to this dataset instance
        self.data = self.scaler.fit_transform(dataframe_values)
        # Store original indices to map back results
        self.original_indices = dataframe.index.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.from_numpy(sample)

    def get_original_indices(self, subset_indices):
        # Helper to get original DataFrame indices from dataset indices
        return self.original_indices[subset_indices]


# --- Updated SRR Function ---
def run_srr_on_dataset(dataset: PointCloudDataset, input_dim=3):
    """
    Runs the Self-Representation Refinement algorithm.

    Args:
        dataset (PointCloudDataset): The dataset containing potentially filtered points.
                                      Assumes dataset.original_indices holds the
                                      indices corresponding to the *original* DataFrame.
        input_dim (int): Number of features per point (e.g., 3 for E, N, h).

    Returns:
        tuple: (normal_indices_in_df, anomaly_indices_in_df)
               Indices are relative to the *original* DataFrame.
    """
    # --- SRR Hyperparameters ---
    batch_size = 1024  # Adjust based on memory
    representation_dim = 32  # Tunable AE parameter
    num_occ_estimators = 5  # Ensemble size for refinement
    refinement_iterations = 5  # Number of SRR loops
    convergence_threshold = 0.0001  # Loss-based convergence
    num_epochs_per_iteration = 50  # Increased epochs for AE training
    ae_learning_rate = 0.0005  # Adjusted LR
    occ_contamination = 0.01  # Contamination for IF (Crucial hyperparameter)
    # --- Device Configuration ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Use MPS for M1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Original
    print(f"Using device: {device}")

    # --- Initialize AutoEncoder ---
    autoencoder = AutoEncoder(input_dim=input_dim, representation_dim=representation_dim).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=ae_learning_rate)
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # --- Initialize variables for SRR ---
    # Indices *within the current dataset* being considered inliers
    current_refined_dataset_indices = np.arange(len(dataset))
    prev_loss = float('inf')
    dataset_size = len(dataset)

    # --- SRR Iteration Loop ---
    for iteration in range(refinement_iterations):
        print(f"\n--- SRR Iteration {iteration + 1}/{refinement_iterations} ---")
        print(f"Current refined set size: {len(current_refined_dataset_indices)}")

        if len(current_refined_dataset_indices) < num_occ_estimators * 2:  # Need enough points
            print("Warning: Refined dataset too small to continue refinement. Stopping SRR.")
            break

        # --- Feature Extraction (on the *entire* current dataset) ---
        print("Extracting features for the current dataset...")
        autoencoder.eval()  # Set AE to evaluation mode (important for BatchNorm)
        full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_features_list = []
        with torch.no_grad():
            for data_batch in full_loader:
                data_batch = data_batch.float().to(device)
                _, z = autoencoder(data_batch)
                all_features_list.append(z.cpu().numpy())
        all_features = np.concatenate(all_features_list, axis=0)
        print(f"Extracted features shape: {all_features.shape}")

        # --- Data Refinement using Ensemble OCC ---
        print("Data Refinement using OCC ensemble...")
        # We predict on *all* data points currently in the dataset,
        # but train OCCs only on subsets of the *currently refined* points.
        occ_predictions = np.zeros((dataset_size, num_occ_estimators))

        # Indices to sample from for training OCCs (subset of current refined set)
        indices_to_sample_from = current_refined_dataset_indices
        np.random.shuffle(indices_to_sample_from)
        # Ensure subsets are reasonably sized, prevent tiny subsets
        min_subset_size = max(10, len(indices_to_sample_from) // num_occ_estimators)  # Heuristic min size
        actual_num_estimators = 0

        # Create subsets ensuring minimum size
        subsets_indices = []
        start_idx = 0
        for i in range(num_occ_estimators):
            end_idx = start_idx + len(indices_to_sample_from) // num_occ_estimators
            # Adjust last subset to include remaining points
            if i == num_occ_estimators - 1:
                end_idx = len(indices_to_sample_from)
            subset = indices_to_sample_from[start_idx:end_idx]
            if len(subset) >= min_subset_size:
                subsets_indices.append(subset)
                actual_num_estimators += 1
            start_idx = end_idx

        if actual_num_estimators == 0:
            print("Warning: Could not form any valid OCC subsets. Stopping refinement.")
            break  # Exit refinement if no valid subsets

        print(f"Training {actual_num_estimators} OCC estimators...")
        for i, subset_idx_in_dataset in enumerate(subsets_indices):
            print(f"  Training OCC {i + 1}/{actual_num_estimators} on {len(subset_idx_in_dataset)} points...")
            # Extract features for this subset
            subset_features = all_features[subset_idx_in_dataset]

            # Normalize features *based on the subset*
            scaler_occ = StandardScaler()
            subset_features_scaled = scaler_occ.fit_transform(subset_features)
            # Scale *all* features using the *subset's* scaler
            all_features_scaled = scaler_occ.transform(all_features)

            # Train OCC (Isolation Forest) on the scaled subset
            occ = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=occ_contamination,  # Use defined contamination
                random_state=42 + i,
                n_jobs=-1  # Use multiple cores if available
            ).fit(subset_features_scaled)

            # Predict on *all* scaled features
            preds = occ.predict(all_features_scaled)  # Predicts 1 for inliers, -1 for outliers
            occ_predictions[:, i] = preds  # Store predictions

        # --- Consensus (Majority Voting) ---
        # Count inlier (1) votes; need more than half
        consensus_mask = np.sum(occ_predictions[:, :actual_num_estimators] == 1, axis=1) >= (
                    actual_num_estimators // 2 + 1)
        # Update the indices *within the dataset* that are considered inliers
        current_refined_dataset_indices = np.where(consensus_mask)[0]
        print(f"Refined dataset size after consensus: {len(current_refined_dataset_indices)}")

        if len(current_refined_dataset_indices) == 0:
            print("Error: No points survived the refinement process. Stopping SRR.")
            # Return empty sets, indicating failure
            return np.array([], dtype=int), dataset.original_indices  # All points considered anomalous

        # --- Update Representation Learner (Autoencoder) ---
        print("Updating Representation Learner (Autoencoder)...")
        # Create a Subset of the original dataset based on refined indices
        refined_subset = Subset(dataset, current_refined_dataset_indices)
        refined_loader = DataLoader(refined_subset, batch_size=batch_size, shuffle=True)

        autoencoder.train()  # Set AE back to training mode
        iteration_total_loss = 0.0
        for epoch in range(num_epochs_per_iteration):
            epoch_loss = 0.0
            for data_batch in refined_loader:
                data_batch = data_batch.float().to(device)
                reconstructed, _ = autoencoder(data_batch)
                loss = nn.MSELoss()(reconstructed, data_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(refined_loader)
            iteration_total_loss += epoch_loss
            if (epoch + 1) % 10 == 0:  # Print loss every 10 epochs
                print(f"  Epoch {epoch + 1}/{num_epochs_per_iteration}, AE Loss: {epoch_loss:.6f}")

        average_iteration_loss = iteration_total_loss / num_epochs_per_iteration
        print(f"Iteration Average AE Loss: {average_iteration_loss:.6f}")

        # Step the scheduler based on the average loss of the iteration
        scheduler.step(average_iteration_loss)

        # --- Check for Convergence ---
        if abs(prev_loss - average_iteration_loss) < convergence_threshold:
            print("Convergence threshold reached.")
            break
        prev_loss = average_iteration_loss

    # --- Final Anomaly Detection ---
    print("\n--- Training Final OCC and Performing Final Prediction ---")
    autoencoder.eval()  # Set AE to evaluation mode

    # Extract features from the *final refined set* for training the final OCC
    print(f"Extracting features from final refined set ({len(current_refined_dataset_indices)} points)...")
    final_refined_subset = Subset(dataset, current_refined_dataset_indices)
    final_refined_loader = DataLoader(final_refined_subset, batch_size=batch_size, shuffle=False)

    final_features_list = []
    with torch.no_grad():
        for data_batch in final_refined_loader:
            data_batch = data_batch.float().to(device)
            _, z = autoencoder(data_batch)
            final_features_list.append(z.cpu().numpy())
    final_refined_features = np.concatenate(final_features_list, axis=0)

    # Scale features for final OCC training
    final_scaler = StandardScaler()
    final_refined_features_scaled = final_scaler.fit_transform(final_refined_features)

    # Train final OCC (using the same contamination setting)
    print("Training final Isolation Forest...")
    final_occ = IsolationForest(contamination=occ_contamination, random_state=42, n_jobs=-1)
    final_occ.fit(final_refined_features_scaled)

    # --- Predict on the *entire original dataset* passed to SRR ---
    print("Extracting features for the entire dataset for final prediction...")
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features_list = []
    with torch.no_grad():
        for data_batch in full_loader:
            data_batch = data_batch.float().to(device)
            _, z = autoencoder(data_batch)
            all_features_list.append(z.cpu().numpy())
    all_features_final = np.concatenate(all_features_list, axis=0)

    # Scale *all* features using the *final* scaler
    all_features_final_scaled = final_scaler.transform(all_features_final)

    # Predict anomalies on the entire dataset
    print("Predicting anomalies on the entire dataset...")
    final_anomaly_labels = final_occ.predict(all_features_final_scaled)  # 1=inlier, -1=outlier

    # --- Map results back to original DataFrame indices ---
    original_df_indices = dataset.original_indices

    final_normal_mask = (final_anomaly_labels == 1)
    final_anomaly_mask = (final_anomaly_labels == -1)

    normal_indices_in_df = original_df_indices[final_normal_mask]
    anomaly_indices_in_df = original_df_indices[final_anomaly_mask]

    print(f"Final result: {len(normal_indices_in_df)} normal points, {len(anomaly_indices_in_df)} anomaly points.")

    return normal_indices_in_df, anomaly_indices_in_df


# --- New Visualization Function for SRR Results ---
def visualize_srr_results(original_df: pd.DataFrame,
                          initial_filter_mask: pd.Series,
                          srr_normal_indices: np.ndarray,
                          srr_anomaly_indices: np.ndarray,
                          plane: str = 'xz'):
    """
    Visualizes the results of the SRR pipeline.

    Args:
        original_df: The very original DataFrame loaded from LAS.
        initial_filter_mask: Boolean Series indicating points kept by initial SOR/LOF.
        srr_normal_indices: Indices (in original_df) kept by SRR.
        srr_anomaly_indices: Indices (in original_df) flagged as anomalies by SRR.
        plane: 'xz' or 'yz' for projection.
    """
    print(f"\nVisualizing SRR Results for {plane.upper()} Plane...")

    if plane == 'xz':
        pcd = create_xz_pcd(original_df)
        x_col, z_col = 'x', 'z'
    elif plane == 'yz':
        pcd = create_yz_pcd(original_df)
        x_col, z_col = 'y', 'z'
    else:
        raise ValueError("Plane must be 'xz' or 'yz'")

    # Identify different groups based on indices
    initially_removed_mask = ~initial_filter_mask
    srr_kept_mask = pd.Series(False, index=original_df.index)
    srr_kept_mask.loc[srr_normal_indices] = True

    srr_removed_mask = pd.Series(False, index=original_df.index)
    srr_removed_mask.loc[srr_anomaly_indices] = True

    # Define point groups for plotting
    points_initially_removed = pcd[initially_removed_mask]
    points_kept_by_srr = pcd[srr_kept_mask]  # Final normal points
    points_removed_by_srr = pcd[srr_removed_mask]  # Final anomaly points (were kept by initial filter)

    print(f"  Total Original: {len(pcd)}")
    print(f"  Initially Removed (SOR/LOF): {len(points_initially_removed)}")
    print(f"  Input to SRR: {initial_filter_mask.sum()}")
    print(f"  Kept by SRR (Final Inliers): {len(points_kept_by_srr)}")
    print(f"  Removed by SRR (Final Outliers): {len(points_removed_by_srr)}")

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Determine plot limits
    z_min = pcd[z_col].min()
    z_max = pcd[z_col].max()
    x_min = pcd[x_col].min()
    x_max = pcd[x_col].max()

    # Plot points removed by SRR first (red 'x') - these are the ones to inspect
    if not points_removed_by_srr.empty:
        ax.scatter(points_removed_by_srr[x_col], points_removed_by_srr[z_col], s=10, c='red', marker='x',
                   label=f'Removed by SRR ({len(points_removed_by_srr)})')

    # Plot points kept by SRR (green '.')
    if not points_kept_by_srr.empty:
        ax.scatter(points_kept_by_srr[x_col], points_kept_by_srr[z_col], s=1, c='green', marker='.',
                   label=f'Kept by SRR ({len(points_kept_by_srr)})')

    # Plot points initially removed (gray '.') - optional, can be noisy
    # if not points_initially_removed.empty:
    #     ax.scatter(points_initially_removed[x_col], points_initially_removed[z_col], s=1, c='gray', alpha=0.3, marker='.', label=f'Initially Removed ({len(points_initially_removed)})')

    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(z_col.upper())
    ax.set_title(f'SRR Results - {plane.upper()} Plane')
    ax.legend(markerscale=4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --- Main Execution Logic ---
def main():
    filename = "Seahawk_231015_223539_00_D.las"  # Replace with your actual filename
    # filename = "your_large_las_file.las" # Use a representative file
    loader = LasLoader(f"./__rawdata__/{filename}")  # Adjust path as needed
    try:
        df = loader.load_to_dataframe()
        print(f"Loaded {len(df)} points.")
    except FileNotFoundError:
        print(f"Error: File not found at ./__rawdata__/{filename}")
        return
    except Exception as e:
        print(f"Error loading LAS file: {e}")
        return

    if df.empty:
        print("Loaded DataFrame is empty. Exiting.")
        return

    # --- Initial 2D Filtering ---
    print("\n--- Stage 1: Initial 2D Filtering (SOR + LOF) ---")
    # Process X-Z plane
    print("Processing X-Z plane...")
    xz_pcd = create_xz_pcd(df)
    xz_mask_sor = statistical_outlier_removal_df(xz_pcd, k=20, z_max=2.0)
    xz_pcd_sor = xz_pcd[xz_mask_sor]
    print(f"  X-Z SOR removed {len(xz_pcd) - len(xz_pcd_sor)} points.")
    xz_mask_lof = remove_outliers_lof(xz_pcd_sor, contamination=0.01, n_neighbors=20)
    # Combine SOR and LOF masks for XZ
    xz_combined_mask = pd.Series(False, index=df.index)
    xz_combined_mask.loc[xz_mask_sor.index[xz_mask_sor]] = xz_mask_lof  # Only apply LOF mask where SOR was True
    print(f"  X-Z LOF removed {len(xz_pcd_sor) - xz_mask_lof.sum()} points from SOR survivors.")

    # Process Y-Z plane
    print("Processing Y-Z plane...")
    yz_pcd = create_yz_pcd(df)
    yz_mask_sor = statistical_outlier_removal_df(yz_pcd, k=20, z_max=2.0)
    yz_pcd_sor = yz_pcd[yz_mask_sor]
    print(f"  Y-Z SOR removed {len(yz_pcd) - len(yz_pcd_sor)} points.")
    yz_mask_lof = remove_outliers_lof(yz_pcd_sor, contamination=0.01, n_neighbors=20)
    # Combine SOR and LOF masks for YZ
    yz_combined_mask = pd.Series(False, index=df.index)
    yz_combined_mask.loc[yz_mask_sor.index[yz_mask_sor]] = yz_mask_lof  # Only apply LOF mask where SOR was True
    print(f"  Y-Z LOF removed {len(yz_pcd_sor) - yz_mask_lof.sum()} points from SOR survivors.")

    # Combine masks: Point must survive filtering in *both* projections
    initial_inlier_mask = xz_combined_mask & yz_combined_mask
    df_initial_filtered = df[initial_inlier_mask]
    print(f"Total points after initial 2D filtering: {len(df_initial_filtered)}")

    # Visualize initial filtering stages (optional, can be verbose)
    # visualize_points(create_xz_pcd(df), create_xz_pcd(df_initial_filtered), 'X-Z Plane Initial Filtering (SOR+LOF)')
    # visualize_points(create_yz_pcd(df), create_yz_pcd(df_initial_filtered), 'Y-Z Plane Initial Filtering (SOR+LOF)')

    if df_initial_filtered.empty:
        print("No points survived initial filtering. Cannot proceed with SRR.")
        return

    # --- Stage 2: SRR on Filtered 3D Point Cloud ---
    print("\n--- Stage 2: Applying SRR on initially filtered 3D point cloud ---")
    # Select relevant columns for SRR
    pcd_for_srr = df_initial_filtered[['E', 'N', 'h']]
    # Create dataset (this performs scaling)
    pcd_dataset = PointCloudDataset(pcd_for_srr)

    # Run SRR - Get indices relative to the *original* DataFrame
    srr_normal_indices, srr_anomaly_indices = run_srr_on_dataset(pcd_dataset, input_dim=3)

    # Get the final filtered DataFrame using SRR normal indices
    df_srr_final_filtered = df.loc[srr_normal_indices]

    # --- Stage 3: Visualization and Validation ---
    print("\n--- Stage 3: Visualizing Final Results ---")
    # Use the dedicated SRR visualization
    visualize_srr_results(df, initial_inlier_mask, srr_normal_indices, srr_anomaly_indices, plane='xz')
    visualize_srr_results(df, initial_inlier_mask, srr_normal_indices, srr_anomaly_indices, plane='yz')

    # You can now save df_srr_final_filtered or use it further
    print(f"\nPipeline complete. Final point count: {len(df_srr_final_filtered)}")
    # Example: Save final points
    # final_output_filename = filename.replace(".las", "_filtered.las")
    # Note: Saving back to LAS requires laspy or similar and mapping columns back.
    # df_srr_final_filtered.to_csv(filename.replace(".las", "_filtered.csv"), index=False)
    # print(f"Filtered data saved to {filename.replace('.las', '_filtered.csv')}")


if __name__ == "__main__":
    # Add basic checks or setup if needed
    # For example, check PyTorch MPS availability
    if torch.backends.mps.is_available():
        print("MPS backend is available. PyTorch will use the M1 GPU.")
    elif torch.cuda.is_available():
        print("CUDA backend is available. PyTorch will use the NVIDIA GPU.")
    else:
        print("No GPU accelerator available. PyTorch will use the CPU.")

    main()