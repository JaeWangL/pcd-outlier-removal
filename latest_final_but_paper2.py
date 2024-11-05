import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
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
    def __init__(self, input_dim=3, representation_dim=32):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(32),
            nn.Linear(32, representation_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(representation_dim, 32),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

class PointCloudDataset(Dataset):
    def __init__(self, dataframe):
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(dataframe.values.astype(np.float32))
        self.indices = dataframe.index.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.from_numpy(sample)

def run_srr_on_dataset(dataset, input_dim=3):
    # 디바이스 설정
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # SRR 파라미터 설정
    batch_size = 1024
    representation_dim = 32
    num_occ_estimators = 5
    refinement_iterations = 3
    convergence_threshold = 0.0001
    initial_num_epochs = 30  # 초기 에폭 수
    initial_learning_rate = 0.001  # 초기 학습률
    initial_contamination = 0.01  # 초기 contamination 값
    contamination_decay = 0.5  # contamination 감소율 (반복마다 반으로 감소)
    min_contamination = 0.001  # contamination의 최소값

    # AutoEncoder 초기화
    autoencoder = AutoEncoder(input_dim=input_dim, representation_dim=representation_dim).to(device)

    # SRR을 위한 변수 초기화
    refined_indices = np.arange(len(dataset))
    prev_loss = float('inf')

    # 전체 데이터의 범위 계산 (시각화를 위한 축 범위 고정)
    data_array = dataset.data
    data_array_original = dataset.scaler.inverse_transform(data_array)
    x_min, x_max = np.min(data_array_original[:, 0]), np.max(data_array_original[:, 0])
    y_min, y_max = np.min(data_array_original[:, 1]), np.max(data_array_original[:, 1])
    z_min, z_max = np.min(data_array_original[:, 2]), np.max(data_array_original[:, 2])

    # 초기 데이터 시각화
    visualize_initial_data(dataset, x_min, x_max, y_min, y_max, z_min, z_max)

    # 이전 단계의 정상 데이터 인덱스 저장을 위한 변수
    previous_normal_indices = refined_indices.copy()

    # 이상치 비율 변화를 저장할 리스트
    refinement_stats = []

    for iteration in range(refinement_iterations):
        print(f"SRR Iteration {iteration + 1}/{refinement_iterations}")

        # 현재 contamination 값 계산
        contamination = max(initial_contamination * (contamination_decay ** iteration), min_contamination)
        print(f"Current contamination: {contamination}")

        # AutoEncoder의 학습률 및 에폭 수 조정
        learning_rate = initial_learning_rate * (0.9 ** iteration)
        num_epochs = max(5, initial_num_epochs - iteration * 2)  # 에폭 수를 점진적으로 감소시키되 최소 5 에폭 유지
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

        # 전체 데이터셋에 대한 특징 추출
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

        # OCC 예측 배열 초기화
        occ_predictions = np.zeros((len(dataset), num_occ_estimators))

        # 모든 모델의 이상치 점수를 누적하기 위한 배열 초기화
        all_anomaly_scores = np.zeros(len(dataset))

        # 데이터 정제
        print("Data Refinement...")
        np.random.shuffle(refined_indices)
        subsets = np.array_split(refined_indices, num_occ_estimators)

        for i, subset_indices in enumerate(subsets):
            # 부분집합 시각화
            visualize_subset(
                dataset,
                subset_indices,
                iteration,
                i + 1,  # 부분집합 번호 (1부터 시작)
                x_min, x_max, y_min, y_max, z_min, z_max
            )

            # 부분집합에서 특징 추출
            subset_features = full_features[subset_indices]

            # 특징 정규화
            scaler = StandardScaler()
            subset_features_scaled = scaler.fit_transform(subset_features)
            full_features_scaled = scaler.transform(full_features)

            # Isolation Forest 학습
            occ = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=contamination,
                random_state=42 + i,
                n_jobs=-1
            ).fit(subset_features_scaled)

            # 전체 데이터에 대한 예측
            preds = occ.predict(full_features_scaled)
            occ_predictions[:, i] = preds

            # 전체 데이터에 대한 이상치 점수 계산
            scores = -occ.decision_function(full_features_scaled)  # 이상치일수록 높은 값
            all_anomaly_scores += scores

        # 각 데이터 포인트에 대한 평균 이상치 점수 계산
        avg_anomaly_scores = all_anomaly_scores / num_occ_estimators

        # 현재 iteration에서의 이상치 점수 분포 시각화
        visualize_anomaly_scores(avg_anomaly_scores, f'Anomaly Scores Distribution at Iteration {iteration + 1}')

        # 다수결 합의를 통한 인라이어 결정
        consensus = np.sum(occ_predictions == 1, axis=1) >= (num_occ_estimators // 2 + 1)
        refined_indices = np.where(consensus)[0]
        print(f"Refined dataset size: {len(refined_indices)}")

        # 데이터 다양성 유지를 위해 이전 단계의 정상 데이터 중 일부를 추가
        if iteration > 0:
            sample_size = int(len(previous_normal_indices) * 0.1)  # 이전 단계의 정상 데이터 중 10%를 샘플링
            sampled_indices = np.random.choice(previous_normal_indices, size=sample_size, replace=False)
            refined_indices = np.unique(np.concatenate([refined_indices, sampled_indices]))
        previous_normal_indices = refined_indices.copy()

        # 정상치와 이상치 마스크 생성
        normal_mask = np.zeros(len(dataset), dtype=bool)
        normal_mask[refined_indices] = True
        anomaly_mask = ~normal_mask

        # 현재 iteration의 데이터 시각화
        visualize_iteration(dataset, iteration, normal_mask, anomaly_mask, x_min, x_max, y_min, y_max, z_min, z_max)

        # 정상치와 이상치 개수 기록
        num_inliers = np.sum(normal_mask)
        num_outliers = np.sum(anomaly_mask)
        refinement_stats.append({
            'iteration': iteration + 1,
            'num_inliers': num_inliers,
            'num_outliers': num_outliers
        })

        # 표현 학습기 업데이트
        refined_dataset = torch.utils.data.Subset(dataset, refined_indices)
        refined_loader = DataLoader(refined_dataset, batch_size=batch_size, shuffle=True)

        # 자기 지도 학습 (AutoEncoder)
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

        # 손실 함수의 변화 추이 시각화
        plt.figure()
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        plt.title(f'Iteration {iteration + 1} - AutoEncoder Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

        prev_loss = epoch_loss

    # 이상치 비율 변화 시각화
    visualize_outlier_ratio(refinement_stats)

    # 최종 OCC 학습
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

    # 최종 Isolation Forest 학습
    final_occ = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    final_occ.fit(features)

    # 전체 데이터셋에 대한 이상치 점수 계산
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features_full = []
    with torch.no_grad():
        for data in full_loader:
            data = data.float().to(device)
            _, z = autoencoder(data)
            feature = z.cpu().numpy()
            features_full.append(feature)
    features_full = np.concatenate(features_full, axis=0)
    features_full = scaler.transform(features_full)
    anomaly_labels = final_occ.predict(features_full)
    anomaly_scores = -final_occ.decision_function(features_full)  # 이상치일수록 높은 값

    # -1은 이상치, 1은 정상
    anomaly_mask = anomaly_labels == -1  # 이상치 마스크
    normal_mask = anomaly_labels == 1    # 정상 데이터 마스크

    # 최종 결과 시각화
    visualize_final_results(dataset, normal_mask, anomaly_mask, x_min, x_max, y_min, y_max, z_min, z_max)

    # 최종 이상치 점수 분포 시각화
    visualize_anomaly_scores(anomaly_scores, 'Anomaly Scores Distribution at Final Iteration')

    # 잠재 공간 시각화
    visualize_latent_space(features_full, anomaly_labels, 'Latent Space Visualization')

    # 이상치 비율 변화 시각화
    visualize_outlier_ratio(refinement_stats)

    # 원본 데이터프레임에서의 인덱스 얻기
    indices = dataset.indices
    normal_indices_in_df = indices[normal_mask]

    return normal_indices_in_df

def visualize_subset(dataset, subset_indices, iteration, subset_number, x_min, x_max, y_min, y_max, z_min, z_max):
    data_array = dataset.data
    data_array_original = dataset.scaler.inverse_transform(data_array)

    # 부분집합이 너무 큰 경우 샘플링
    if len(subset_indices) > 100000:
        sample_indices = np.random.choice(subset_indices, size=int(len(subset_indices) * 0.1), replace=False)
        subset_data = data_array_original[sample_indices]
    else:
        subset_data = data_array_original[subset_indices]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(subset_data[:, 0], subset_data[:, 1], subset_data[:, 2], c='green', label=f'Subset {subset_number}', s=1)
    ax.set_title(f'Iteration {iteration + 1} - Subset {subset_number} Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.view_init(elev=20, azim=-60)  # 시각화 각도 고정
    plt.tight_layout()
    plt.show()

def visualize_anomaly_scores(scores, title):
    plt.figure()
    plt.hist(scores, bins=50, color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.show()

def visualize_outlier_ratio(refinement_stats):
    iterations = [stat['iteration'] for stat in refinement_stats]
    num_inliers = [stat['num_inliers'] for stat in refinement_stats]
    num_outliers = [stat['num_outliers'] for stat in refinement_stats]

    plt.figure()
    plt.plot(iterations, num_inliers, marker='o', label='Number of Inliers')
    plt.plot(iterations, num_outliers, marker='o', label='Number of Outliers')
    plt.title('Inliers and Outliers over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Points')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_final_results(dataset, normal_mask, anomaly_mask, x_min, x_max, y_min, y_max, z_min, z_max):
    data_array = dataset.data
    data_array_original = dataset.scaler.inverse_transform(data_array)
    # 정상치와 이상치 분리
    normal_data = data_array_original[normal_mask]
    anomaly_data = data_array_original[anomaly_mask]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(normal_data[:, 0], normal_data[:, 1], normal_data[:, 2], c='b', label='Normal', s=1)
    ax.scatter(anomaly_data[:, 0], anomaly_data[:, 1], anomaly_data[:, 2], c='r', label='Outlier', s=1)
    ax.set_title('Final Results')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.view_init(elev=20, azim=-60)  # 시각화 각도 고정
    plt.tight_layout()
    plt.show()

def visualize_latent_space(features, anomaly_labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    normal_idx = anomaly_labels == 1
    anomaly_idx = anomaly_labels == -1

    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[normal_idx, 0], features_2d[normal_idx, 1], c='b', label='Normal', s=1)
    plt.scatter(features_2d[anomaly_idx, 0], features_2d[anomaly_idx, 1], c='r', label='Outlier', s=1)
    plt.legend()
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

def visualize_initial_data(dataset, x_min, x_max, y_min, y_max, z_min, z_max):
    data_array = dataset.data
    data_array_original = dataset.scaler.inverse_transform(data_array)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_array_original[:, 0], data_array_original[:, 1], data_array_original[:, 2], c='gray', s=1)

    ax.set_title('Initial Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    plt.tight_layout()
    plt.show()

def visualize_iteration(dataset, iteration, normal_mask, anomaly_mask, x_min, x_max, y_min, y_max, z_min, z_max):
    data_array = dataset.data
    data_array_original = dataset.scaler.inverse_transform(data_array)

    # 정상치와 이상치 분리
    normal_data = data_array_original[normal_mask]
    anomaly_data = data_array_original[anomaly_mask]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(normal_data[:, 0], normal_data[:, 1], normal_data[:, 2], c='b', label='Normal', s=1)
    ax.scatter(anomaly_data[:, 0], anomaly_data[:, 1], anomaly_data[:, 2], c='r', label='Outlier', s=1)

    ax.set_title(f'Iteration {iteration + 1}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.view_init(elev=20, azim=-60)  # 시각화 각도 고정
    plt.tight_layout()
    plt.show()

def visualize_final_results(dataset, normal_mask, anomaly_mask, x_min, x_max, y_min, y_max, z_min, z_max):
    data_array = dataset.data
    data_array_original = dataset.scaler.inverse_transform(data_array)

    # 정상치와 이상치 분리
    normal_data = data_array_original[normal_mask]
    anomaly_data = data_array_original[anomaly_mask]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(normal_data[:, 0], normal_data[:, 1], normal_data[:, 2], c='b', label='Normal', s=1)
    ax.scatter(anomaly_data[:, 0], anomaly_data[:, 1], anomaly_data[:, 2], c='r', label='Outlier', s=1)

    ax.set_title('Final Results')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.view_init(elev=20, azim=-60)  # 시각화 각도 고정
    plt.tight_layout()
    plt.show()

def main():
    filename = "Seahawk_231015_223539_00_D.las"
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

    # Now apply SRR on the full 3D point cloud
    print("Applying SRR on full 3D point cloud...")
    pcd = df[['E', 'N', 'h']]
    pcd_filtered = pcd[inlier_mask]
    pcd_dataset = PointCloudDataset(pcd_filtered)
    pcd_normal_indices = run_srr_on_dataset(pcd_dataset, input_dim=3)

    # Get the filtered DataFrame from the indices
    df_srr_filtered = df.loc[pcd_normal_indices]

    # Visualize the results
    # Visualize X-Z plane
    xz_original = xz_pcd
    xz_filtered_final = create_xz_pcd(df_srr_filtered)
    visualize_points(xz_original, xz_filtered_final, 'X-Z Plane after SRR')

    # Visualize Y-Z plane
    yz_original = yz_pcd
    yz_filtered_final = create_yz_pcd(df_srr_filtered)
    visualize_points(yz_original, yz_filtered_final, 'Y-Z Plane after SRR')

if __name__ == "__main__":
    main()