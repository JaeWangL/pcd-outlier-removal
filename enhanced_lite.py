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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time # To measure execution time
import laspy # Added for LAS I/O

# Assuming LasLoader exists and works as before
# Make sure it stores the original header!
from loader.las_loader import LasLoader # Needs header access, e.g., self.loader.header

# --- Utility Functions (create_xz_pcd, create_yz_pcd - unchanged) ---
def create_xz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    if 'E' not in df.columns or 'h' not in df.columns:
        raise ValueError("DataFrame must contain 'E' and 'h' columns")
    return pd.DataFrame({'x': df['E'], 'z': df['h']}, index=df.index)

def create_yz_pcd(df: pd.DataFrame) -> pd.DataFrame:
    if 'N' not in df.columns or 'h' not in df.columns:
        raise ValueError("DataFrame must contain 'N' and 'h' columns")
    return pd.DataFrame({'y': df['N'], 'z': df['h']}, index=df.index)

# --- Outlier Removal Functions ---
def statistical_outlier_removal_df(df: pd.DataFrame, k=20, z_max=2.0) -> pd.Series:
    # (Code mostly unchanged, ensure n_jobs=-1 is present for potential speedup)
    if df.empty:
        return pd.Series(dtype=bool)
    points = df.values
    # Use n_jobs=-1 to leverage multiple cores
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    if distances.shape[1] > 1:
        mean_distances = np.mean(distances[:, 1:], axis=1)
    else:
         mean_distances = np.zeros(len(df))

    if len(mean_distances) > 1:
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        if global_std == 0:
             threshold = global_mean
        else:
             threshold = global_mean + z_max * global_std
        mask = mean_distances < threshold
    else:
        mask = np.ones(len(df), dtype=bool)

    return pd.Series(mask, index=df.index)

def remove_outliers_lof(pcd: pd.DataFrame, contamination: float = 0.01, n_neighbors: int = 20) -> pd.Series:
    """Applies LOF after scaling the data."""
    if pcd.empty or len(pcd) <= n_neighbors:
        print(f"Warning: LOF skipped. Not enough points ({len(pcd)}) for n_neighbors={n_neighbors}.")
        # Return a mask indicating all existing points are kept
        return pd.Series(True, index=pcd.index)

    # **Apply StandardScaler before LOF**
    scaler = StandardScaler()
    pcd_scaled = scaler.fit_transform(pcd.values)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
    y_pred = lof.fit_predict(pcd_scaled) # Predict on scaled data
    mask = y_pred != -1
    return pd.Series(mask, index=pcd.index)

# --- Visualization (visualize_points, visualize_srr_results - unchanged) ---
# (Keep visualize_points and visualize_srr_results as they were in the previous version)
def visualize_points(original: pd.DataFrame, filtered: pd.DataFrame, title: str) -> None:
    if original.empty and filtered.empty:
        print(f"Skipping visualization for '{title}': Both DataFrames are empty.")
        return
    if original.empty:
        print(f"Warning for '{title}': Original DataFrame is empty.")
    if filtered.empty:
        print(f"Warning for '{title}': Filtered DataFrame is empty.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x_col = original.columns[0] if not original.empty else filtered.columns[0]
    z_col = original.columns[1] if not original.empty else filtered.columns[1]

    z_min_orig = original[z_col].min() if not original.empty else np.inf
    z_max_orig = original[z_col].max() if not original.empty else -np.inf
    z_min_filt = filtered[z_col].min() if not filtered.empty else np.inf
    z_max_filt = filtered[z_col].max() if not filtered.empty else -np.inf

    z_min = min(z_min_orig, z_min_filt)
    z_max = max(z_max_orig, z_max_filt)
    if np.isinf(z_min) or np.isinf(z_max): # Handle cases where one or both are empty
        z_min_valid = original[z_col].min() if not original.empty else (filtered[z_col].min() if not filtered.empty else 0)
        z_max_valid = original[z_col].max() if not original.empty else (filtered[z_col].max() if not filtered.empty else 1)
        z_min = z_min_valid
        z_max = z_max_valid


    if not original.empty:
        scatter1 = ax1.scatter(original[x_col], original[z_col], s=1, c=original[z_col], cmap='viridis', vmin=z_min, vmax=z_max)
        ax1.set_title(f'Original ({len(original)} points)')
        ax1.set_xlabel(x_col.upper())
        ax1.set_ylabel(z_col.upper())
        if not np.isinf(z_min) and not np.isinf(z_max):
            plt.colorbar(scatter1, ax=ax1, label=f'{z_col.upper()} value')
    else:
        ax1.set_title('Original (0 points)')
        ax1.set_xlabel(x_col.upper())
        ax1.set_ylabel(z_col.upper())


    if not filtered.empty:
        scatter2 = ax2.scatter(filtered[x_col], filtered[z_col], s=1, c=filtered[z_col], cmap='viridis', vmin=z_min, vmax=z_max)
        ax2.set_title(f'After Filtering ({len(filtered)} points)')
        ax2.set_xlabel(x_col.upper())
        if not np.isinf(z_min) and not np.isinf(z_max):
             plt.colorbar(scatter2, ax=ax2, label=f'{z_col.upper()} value')
    else:
        ax2.set_title('After Filtering (0 points)')
        ax2.set_xlabel(x_col.upper())


    if not np.isinf(z_min) and not np.isinf(z_max):
         ax1.set_ylim(z_min, z_max)
         ax2.set_ylim(z_min, z_max)


    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def visualize_srr_results(original_df: pd.DataFrame,
                           initial_filter_mask: pd.Series,
                           srr_normal_indices: np.ndarray,
                           srr_anomaly_indices: np.ndarray,
                           plane: str = 'xz'):
    """Visualizes the results of the SRR pipeline."""
    print(f"\nVisualizing SRR Results for {plane.upper()} Plane...")

    if plane == 'xz':
        pcd = create_xz_pcd(original_df)
        x_col, z_col = 'x', 'z'
    elif plane == 'yz':
        pcd = create_yz_pcd(original_df)
        x_col, z_col = 'y', 'z'
    else:
        raise ValueError("Plane must be 'xz' or 'yz'")

    initially_removed_mask = ~initial_filter_mask
    srr_kept_mask = pd.Series(False, index=original_df.index)
    if len(srr_normal_indices) > 0: # Check if array is not empty
        srr_kept_mask.loc[srr_normal_indices] = True

    srr_removed_mask = pd.Series(False, index=original_df.index)
    if len(srr_anomaly_indices) > 0: # Check if array is not empty
        srr_removed_mask.loc[srr_anomaly_indices] = True

    points_initially_removed = pcd[initially_removed_mask]
    points_kept_by_srr = pcd[srr_kept_mask]
    points_removed_by_srr = pcd[srr_removed_mask & initial_filter_mask] # Show only points removed by SRR *that passed initial filter*

    print(f"  Total Original: {len(pcd)}")
    print(f"  Initially Removed (SOR/LOF): {len(points_initially_removed)}")
    print(f"  Input to SRR: {initial_filter_mask.sum()}")
    print(f"  Kept by SRR (Final Inliers): {len(points_kept_by_srr)}")
    print(f"  Removed by SRR (Final Outliers): {len(points_removed_by_srr)}")


    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    z_min = pcd[z_col].min() if not pcd.empty else 0
    z_max = pcd[z_col].max() if not pcd.empty else 1
    x_min = pcd[x_col].min() if not pcd.empty else 0
    x_max = pcd[x_col].max() if not pcd.empty else 1


    if not points_removed_by_srr.empty:
        ax.scatter(points_removed_by_srr[x_col], points_removed_by_srr[z_col], s=15, c='red', marker='x', label=f'Removed by SRR ({len(points_removed_by_srr)})', zorder=3)


    if not points_kept_by_srr.empty:
        ax.scatter(points_kept_by_srr[x_col], points_kept_by_srr[z_col], s=1, c='green', marker='.', label=f'Kept by SRR ({len(points_kept_by_srr)})', zorder=2)

    # Optional: Plot initially removed points
    # if not points_initially_removed.empty:
    #     ax.scatter(points_initially_removed[x_col], points_initially_removed[z_col], s=1, c='gray', alpha=0.2, marker='.', label=f'Initially Removed ({len(points_initially_removed)})', zorder=1)


    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(z_col.upper())
    ax.set_title(f'SRR Results - {plane.upper()} Plane (Points removed by SRR marked "x")')
    ax.legend(markerscale=3)
    if not pcd.empty:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- Deep Learning Components (AutoEncoder, PointCloudDataset - unchanged) ---
class AutoEncoder(nn.Module):
    # (Unchanged from previous version with BatchNorm)
    def __init__(self, input_dim=3, representation_dim=32):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, representation_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(representation_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

class PointCloudDataset(Dataset):
    # (Unchanged from previous version)
    def __init__(self, dataframe):
        dataframe_values = dataframe.values.astype(np.float32)
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(dataframe_values)
        self.original_indices = dataframe.index.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.from_numpy(sample)

    def get_original_indices(self, subset_indices):
        return self.original_indices[subset_indices]

# --- SRR Function ---
def run_srr_on_dataset(dataset: PointCloudDataset, input_dim=3):
    """Runs SRR, optimized for speed."""
    # --- SRR Hyperparameters ---
    batch_size = 1024 # Keep batch size reasonable for memory
    representation_dim = 32
    num_occ_estimators = 5
    refinement_iterations = 5
    convergence_threshold = 0.0001
    # **Optimization: Reduce epochs per SRR iteration**
    num_epochs_per_iteration = 15 # Reduced from 50 - significant speedup expected
    ae_learning_rate = 0.0005
    occ_contamination = 0.01
    # **Optimization: DataLoader workers (use 0 for MPS/macOS stability initially)**
    num_workers = 0 # Set > 0 (e.g., 2 or 4) if I/O is slow and you test stability

    # --- Device Configuration ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Initialize AutoEncoder ---
    autoencoder = AutoEncoder(input_dim=input_dim, representation_dim=representation_dim).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=ae_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True) # Reduced patience slightly

    # --- SRR Initialization ---
    current_refined_dataset_indices = np.arange(len(dataset))
    prev_loss = float('inf')
    dataset_size = len(dataset)

    # --- SRR Iteration Loop ---
    for iteration in range(refinement_iterations):
        iteration_start_time = time.time()
        print(f"\n--- SRR Iteration {iteration + 1}/{refinement_iterations} ---")
        print(f"Current refined set size: {len(current_refined_dataset_indices)}")

        if len(current_refined_dataset_indices) < num_occ_estimators * 2:
             print("Warning: Refined dataset too small. Stopping SRR.")
             break

        # --- Feature Extraction ---
        feature_start_time = time.time()
        print("Extracting features...")
        autoencoder.eval()
        # Use num_workers in DataLoader
        full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        all_features_list = []
        with torch.no_grad():
            for data_batch in full_loader:
                data_batch = data_batch.float().to(device)
                _, z = autoencoder(data_batch)
                all_features_list.append(z.cpu().numpy())
        all_features = np.concatenate(all_features_list, axis=0)
        print(f"Feature extraction time: {time.time() - feature_start_time:.2f}s")

        # --- Data Refinement (OCC Ensemble) ---
        refinement_start_time = time.time()
        print("Data Refinement (OCC)...")
        occ_predictions = np.zeros((dataset_size, num_occ_estimators))
        indices_to_sample_from = current_refined_dataset_indices
        np.random.shuffle(indices_to_sample_from)
        min_subset_size = max(10, len(indices_to_sample_from) // num_occ_estimators)
        subsets_indices = []
        start_idx = 0
        actual_num_estimators = 0
        for i in range(num_occ_estimators):
            end_idx = start_idx + len(indices_to_sample_from) // num_occ_estimators
            if i == num_occ_estimators - 1: end_idx = len(indices_to_sample_from)
            subset = indices_to_sample_from[start_idx:end_idx]
            if len(subset) >= min_subset_size :
                 subsets_indices.append(subset)
                 actual_num_estimators += 1
            start_idx = end_idx

        if actual_num_estimators == 0:
            print("Warning: No valid OCC subsets. Stopping refinement.")
            break

        print(f"Training {actual_num_estimators} OCC estimators...")
        for i, subset_idx_in_dataset in enumerate(subsets_indices):
            subset_features = all_features[subset_idx_in_dataset]
            scaler_occ = StandardScaler()
            subset_features_scaled = scaler_occ.fit_transform(subset_features)
            all_features_scaled = scaler_occ.transform(all_features)
            occ = IsolationForest(
                n_estimators=100, max_samples='auto', contamination=occ_contamination,
                random_state=42 + i, n_jobs=-1 # Use n_jobs=-1 for IF
            ).fit(subset_features_scaled)
            preds = occ.predict(all_features_scaled)
            occ_predictions[:, i] = preds

        consensus_mask = np.sum(occ_predictions[:, :actual_num_estimators] == 1, axis=1) >= (actual_num_estimators // 2 + 1)
        current_refined_dataset_indices = np.where(consensus_mask)[0]
        print(f"Refinement time: {time.time() - refinement_start_time:.2f}s")
        print(f"Refined dataset size after consensus: {len(current_refined_dataset_indices)}")

        if len(current_refined_dataset_indices) == 0:
            print("Error: No points survived refinement. Stopping SRR.")
            return np.array([], dtype=int), dataset.original_indices

        # --- Update Representation Learner (AE) ---
        ae_update_start_time = time.time()
        print("Updating AE...")
        refined_subset = Subset(dataset, current_refined_dataset_indices)
        # Use num_workers in DataLoader
        refined_loader = DataLoader(refined_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        autoencoder.train()
        iteration_total_loss = 0.0
        for epoch in range(num_epochs_per_iteration): # Reduced number of epochs
            epoch_loss = 0.0
            for data_batch in refined_loader:
                data_batch = data_batch.float().to(device)
                # --- Optional: Mixed Precision (CUDA only) ---
                # with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                reconstructed, _ = autoencoder(data_batch)
                loss = nn.MSELoss()(reconstructed, data_batch)
                # --- Optional: Mixed Precision (CUDA only) ---
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                # --- Standard Precision ---
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # --- End Standard Precision ---

                epoch_loss += loss.item()

            epoch_loss /= len(refined_loader)
            iteration_total_loss += epoch_loss
            # Print less frequently due to fewer epochs
            if (epoch + 1) % 5 == 0 or epoch == num_epochs_per_iteration - 1:
                 print(f"  Epoch {epoch + 1}/{num_epochs_per_iteration}, AE Loss: {epoch_loss:.6f}")

        average_iteration_loss = iteration_total_loss / num_epochs_per_iteration
        scheduler.step(average_iteration_loss)
        print(f"AE update time: {time.time() - ae_update_start_time:.2f}s")
        print(f"Iteration Average AE Loss: {average_iteration_loss:.6f}")


        # --- Check Convergence ---
        if abs(prev_loss - average_iteration_loss) < convergence_threshold and iteration > 0: # Don't check on first iteration
            print("Convergence threshold reached.")
            break
        prev_loss = average_iteration_loss
        print(f"Total time for SRR Iteration {iteration + 1}: {time.time() - iteration_start_time:.2f}s")


    # --- Final Anomaly Detection ---
    print("\n--- Final Anomaly Detection ---")
    final_pred_start_time = time.time()
    autoencoder.eval()

    # Extract features from final refined set for final OCC training
    print(f"Extracting features from final refined set ({len(current_refined_dataset_indices)} points)...")
    final_refined_subset = Subset(dataset, current_refined_dataset_indices)
    final_refined_loader = DataLoader(final_refined_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    final_features_list = []
    with torch.no_grad():
        for data_batch in final_refined_loader:
            data_batch = data_batch.float().to(device)
            _, z = autoencoder(data_batch)
            final_features_list.append(z.cpu().numpy())

    # Handle case where final refined set might be empty after loop finishes early
    if not final_features_list:
         print("Warning: Final refined feature set is empty. No final OCC trained. Returning based on last refinement.")
         # Decide how to handle this: return current refined set as normal? Or all as anomalies?
         # Let's return the current refined set as normal for now.
         normal_indices_in_df = dataset.get_original_indices(current_refined_dataset_indices)
         all_original_indices = dataset.original_indices
         anomaly_indices_in_df = np.setdiff1d(all_original_indices, normal_indices_in_df, assume_unique=True)
         return normal_indices_in_df, anomaly_indices_in_df


    final_refined_features = np.concatenate(final_features_list, axis=0)
    final_scaler = StandardScaler()
    final_refined_features_scaled = final_scaler.fit_transform(final_refined_features)

    print("Training final Isolation Forest...")
    final_occ = IsolationForest(contamination=occ_contamination, random_state=42, n_jobs=-1)
    final_occ.fit(final_refined_features_scaled)

    # Predict on the *entire original dataset* passed to SRR
    print("Extracting features for the entire dataset for final prediction...")
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_features_list = []
    with torch.no_grad():
        for data_batch in full_loader:
            data_batch = data_batch.float().to(device)
            _, z = autoencoder(data_batch)
            all_features_list.append(z.cpu().numpy())
    all_features_final = np.concatenate(all_features_list, axis=0)
    all_features_final_scaled = final_scaler.transform(all_features_final)

    print("Predicting anomalies on the entire dataset...")
    final_anomaly_labels = final_occ.predict(all_features_final_scaled)

    # Map results back to original DataFrame indices
    original_df_indices = dataset.original_indices
    final_normal_mask = (final_anomaly_labels == 1)
    final_anomaly_mask = (final_anomaly_labels == -1)
    normal_indices_in_df = original_df_indices[final_normal_mask]
    anomaly_indices_in_df = original_df_indices[final_anomaly_mask]

    print(f"Final prediction time: {time.time() - final_pred_start_time:.2f}s")
    print(f"Final result: {len(normal_indices_in_df)} normal points, {len(anomaly_indices_in_df)} anomaly points.")

    return normal_indices_in_df, anomaly_indices_in_df

# --- Function to Save DataFrame to LAS ---
def save_dataframe_to_las(original_las_path: str, filtered_df: pd.DataFrame, output_las_path: str):
    """
    Saves a Pandas DataFrame with point cloud data back to a LAS file,
    preserving the header information from the original file.

    Args:
        original_las_path: Path to the original LAS file to copy header from.
        filtered_df: DataFrame containing the filtered point data.
                     Must contain columns corresponding to LAS dimensions (e.g., 'E', 'N', 'h', 'intensity').
        output_las_path: Path where the new filtered LAS file will be saved.
    """
    if filtered_df.empty:
        print("Warning: Filtered DataFrame is empty. No LAS file will be saved.")
        return

    try:
        # Read the original LAS file to get the header
        with laspy.open(original_las_path) as original_las:
            original_header = original_las.header

            # Create a new LAS data object
            new_las = laspy.LasData(header=original_header)

            # Map DataFrame columns to standard LAS dimensions
            # Adjust this mapping based on your DataFrame columns!
            point_data = np.zeros(len(filtered_df), dtype=original_header.point_format.dtype)

            # Essential dimensions (assuming E, N, h from DataFrame)
            if 'E' in filtered_df.columns: point_data['X'] = filtered_df['E'].values
            if 'N' in filtered_df.columns: point_data['Y'] = filtered_df['N'].values
            if 'h' in filtered_df.columns: point_data['Z'] = filtered_df['h'].values

            # Optional common dimensions (add more if needed and present in your df)
            if 'intensity' in filtered_df.columns and 'intensity' in original_header.point_format.dimension_names:
                 point_data['intensity'] = filtered_df['intensity'].values
            if 'return_number' in filtered_df.columns and 'return_number' in original_header.point_format.dimension_names:
                 point_data['return_number'] = filtered_df['return_number'].values
            if 'number_of_returns' in filtered_df.columns and 'number_of_returns' in original_header.point_format.dimension_names:
                 point_data['number_of_returns'] = filtered_df['number_of_returns'].values
            if 'classification' in filtered_df.columns and 'classification' in original_header.point_format.dimension_names:
                 point_data['classification'] = filtered_df['classification'].values
            if 'gps_time' in filtered_df.columns and 'gps_time' in original_header.point_format.dimension_names:
                 point_data['gps_time'] = filtered_df['gps_time'].values
            # Add other dimensions (scan_angle_rank, user_data, etc.) if necessary

            new_las.points = point_data

            print(f"Saving {len(new_las.points)} points to {output_las_path}...")
            new_las.write(output_las_path)
            print("Filtered LAS file saved successfully.")

    except FileNotFoundError:
        print(f"Error: Original LAS file not found at {original_las_path}")
    except Exception as e:
        print(f"Error saving DataFrame to LAS file: {e}")


# --- Main Execution Logic ---
def main():
    start_time = time.time()
    filename = "Seahawk_231015_223539_00_D.las" # Replace with your actual filename
    input_filepath = f"./__rawdata__/{filename}" # Adjust path as needed
    output_filename = filename.replace(".las", "_filtered.las")
    output_filepath = f"./__output__/{output_filename}" # Save to an output folder

    # Ensure output directory exists (optional)
    import os
    os.makedirs("./__output__", exist_ok=True)

    # --- Load Data ---
    print(f"Loading data from {input_filepath}...")
    # **Crucial:** Ensure LasLoader loads the header and makes it accessible
    # e.g., loader = LasLoader(input_filepath); loader.load_to_dataframe(); header = loader.header
    # For now, we load header again in save function, which is less efficient but works.
    loader = LasLoader(input_filepath) # Assuming loader exists
    try:
        df = loader.load_to_dataframe()
        print(f"Loaded {len(df)} points.")
    except FileNotFoundError:
        print(f"Error: File not found at {input_filepath}")
        return
    except Exception as e:
        print(f"Error loading LAS file: {e}")
        return

    if df.empty:
        print("Loaded DataFrame is empty. Exiting.")
        return

    # --- Stage 1: Initial 2D Filtering ---
    print("\n--- Stage 1: Initial 2D Filtering (SOR + LOF) ---")
    stage1_start = time.time()
    # X-Z
    print("Processing X-Z plane...")
    xz_pcd = create_xz_pcd(df)
    xz_mask_sor = statistical_outlier_removal_df(xz_pcd, k=20, z_max=2.0)
    xz_pcd_sor = xz_pcd[xz_mask_sor]
    xz_mask_lof = remove_outliers_lof(xz_pcd_sor, contamination=0.01, n_neighbors=20) # Uses scaling now
    xz_combined_mask = pd.Series(False, index=df.index)
    if xz_mask_sor.any(): # Check if SOR mask is not all False
      xz_combined_mask.loc[xz_mask_sor.index[xz_mask_sor]] = xz_mask_lof
    print(f"  X-Z kept {xz_combined_mask.sum()} points.")

    # Y-Z
    print("Processing Y-Z plane...")
    yz_pcd = create_yz_pcd(df)
    yz_mask_sor = statistical_outlier_removal_df(yz_pcd, k=20, z_max=2.0)
    yz_pcd_sor = yz_pcd[yz_mask_sor]
    yz_mask_lof = remove_outliers_lof(yz_pcd_sor, contamination=0.01, n_neighbors=20) # Uses scaling now
    yz_combined_mask = pd.Series(False, index=df.index)
    if yz_mask_sor.any(): # Check if SOR mask is not all False
        yz_combined_mask.loc[yz_mask_sor.index[yz_mask_sor]] = yz_mask_lof
    print(f"  Y-Z kept {yz_combined_mask.sum()} points.")

    initial_inlier_mask = xz_combined_mask & yz_combined_mask
    df_initial_filtered = df[initial_inlier_mask]
    print(f"Total points after initial 2D filtering: {len(df_initial_filtered)}")
    print(f"Stage 1 time: {time.time() - stage1_start:.2f}s")

    if df_initial_filtered.empty:
        print("No points survived initial filtering. Cannot proceed with SRR.")
        return

    # --- Stage 2: SRR ---
    print("\n--- Stage 2: SRR ---")
    stage2_start = time.time()
    pcd_for_srr = df_initial_filtered[['E', 'N', 'h']] # Ensure correct columns
    pcd_dataset = PointCloudDataset(pcd_for_srr)
    srr_normal_indices, srr_anomaly_indices = run_srr_on_dataset(pcd_dataset, input_dim=3)
    df_srr_final_filtered = df.loc[srr_normal_indices]
    print(f"Stage 2 (SRR) time: {time.time() - stage2_start:.2f}s")


    # --- Stage 3: Visualization & Saving ---
    print("\n--- Stage 3: Visualization & Saving ---")
    # Visualize SRR results
    visualize_srr_results(df, initial_inlier_mask, srr_normal_indices, srr_anomaly_indices, plane='xz')
    visualize_srr_results(df, initial_inlier_mask, srr_normal_indices, srr_anomaly_indices, plane='yz')

    # Save the final filtered DataFrame to a new LAS file
    save_dataframe_to_las(input_filepath, df_srr_final_filtered, output_filepath)

    print(f"\nPipeline complete. Final point count: {len(df_srr_final_filtered)}")
    print(f"Total execution time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        print("MPS backend is available.")
    elif torch.cuda.is_available():
         print("CUDA backend is available.")
    else:
        print("Using CPU.")
    main()