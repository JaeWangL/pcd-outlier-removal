import numpy as np
from scipy.spatial import cKDTree
import laspy
import csv
import os

# Read GCP coordinates from the CSV file
gcp_coordinates = []
gcp_names = []
with open("../__reference__/reference.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        gcp_name, e, n, h = row
        gcp_coordinates.append([float(e), float(n), float(h)])
        gcp_names.append(gcp_name)
gcp_coordinates = np.array(gcp_coordinates)

# Directory containing the LAS files
las_directory = "../__rawdata__2/filtered"

# Create a dictionary to store the differences for each GCP
gcp_differences = {gcp_name: [] for gcp_name in gcp_names}

# Process each LAS file
for filename in os.listdir(las_directory):
    if filename.endswith(".las"):
        las_path = os.path.join(las_directory, filename)

        # Read the LAS file
        las = laspy.read(las_path)
        point_cloud_data = np.vstack((las.x, las.y, las.z)).transpose()

        # Build a KD-tree from the point cloud data
        kdtree = cKDTree(point_cloud_data[:, :2])  # Use only x and y coordinates

        # Find the nearest point in the point cloud for each GCP coordinate
        distances, indices = kdtree.query(gcp_coordinates[:, :2])

        # Calculate the differences in z-coordinate and store them for each GCP
        for gcp_name, dist, idx in zip(gcp_names, distances, indices):
            if dist <= 1.0:  # Threshold distance for nearby points (adjust as needed)
                diff = point_cloud_data[idx, 2] - gcp_coordinates[gcp_names.index(gcp_name), 2]
                gcp_differences[gcp_name].append(diff)

# Analyze the differences for each GCP
for gcp_name, diffs in gcp_differences.items():
    if len(diffs) > 0:
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        print(f"GCP Name: {gcp_name}")
        print(f"Number of nearby points: {len(diffs)}")
        print(f"Mean difference: {mean_diff:.3f}")
        print(f"Standard deviation of differences: {std_diff:.3f}")
        print()
    else:
        print(f"GCP Name: {gcp_name}")
        print("No nearby points found.")