import open3d as o3d
import numpy as np
import laspy

def load_las_as_point_cloud(las_file_path):
    # Load LAS file
    with laspy.open(las_file_path) as file:
        las = file.read()

    # Extract points from the LAS file
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Create an Open3D point cloud from the LAS file points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud

def apply_sparse_outlier_removal(point_cloud, nb_neighbors=20, std_ratio=2.0):
    # Apply Sparse Outlier Removal
    filtered_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors, std_ratio)
    return filtered_cloud

def display_point_cloud(point_cloud):
    # Display the point cloud
    o3d.visualization.draw_geometries([point_cloud])

# Main function
if __name__ == "__main__":
    las_file_path = '__rawdata__2/Seahawk_231015_225544_00_D.las'  # Change this to the path of your LAS file
    original_point_cloud = load_las_as_point_cloud(las_file_path)
    filtered_point_cloud = original_point_cloud
    filtered_point_cloud = apply_sparse_outlier_removal(original_point_cloud)
    display_point_cloud(filtered_point_cloud)
