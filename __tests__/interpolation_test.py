import os
import numpy as np
import laspy
from scipy.spatial import Delaunay
from rasterio import open as rio_open
from rasterio.transform import from_bounds


def merge_las_files(directory):
    merged_points = []

    for file in os.listdir(directory):
        if file.endswith(".las"):
            las = laspy.read(os.path.join(directory, file))
            points = np.vstack((las.x, las.y, las.z)).transpose()
            merged_points.append(points)

    return np.concatenate(merged_points)


def tin_interpolation(points, resolution):
    # Create a Delaunay triangulation from the points
    tri = Delaunay(points[:, :2])

    # Find the minimum and maximum coordinates
    min_x, min_y = np.min(points[:, :2], axis=0)
    max_x, max_y = np.max(points[:, :2], axis=0)

    # Create a regular grid of points
    x = np.arange(min_x, max_x, resolution)
    y = np.arange(min_y, max_y, resolution)
    x_grid, y_grid = np.meshgrid(x, y)

    # Interpolate the z-values at the grid points
    z_grid = np.zeros_like(x_grid, dtype=np.float32)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            simplex = tri.find_simplex((x_grid[i, j], y_grid[i, j]))
            if simplex != -1:
                vertices = tri.simplices[simplex]
                weights = np.array([
                    ((y_grid[i, j] - points[vertices[1], 1]) * (points[vertices[2], 0] - points[vertices[1], 0]) -
                     (x_grid[i, j] - points[vertices[1], 0]) * (points[vertices[2], 1] - points[vertices[1], 1])) /
                    ((points[vertices[0], 1] - points[vertices[1], 1]) * (
                                points[vertices[2], 0] - points[vertices[1], 0]) -
                     (points[vertices[0], 0] - points[vertices[1], 0]) * (
                                 points[vertices[2], 1] - points[vertices[1], 1])),

                    ((y_grid[i, j] - points[vertices[2], 1]) * (points[vertices[0], 0] - points[vertices[2], 0]) -
                     (x_grid[i, j] - points[vertices[2], 0]) * (points[vertices[0], 1] - points[vertices[2], 1])) /
                    ((points[vertices[1], 1] - points[vertices[2], 1]) * (
                                points[vertices[0], 0] - points[vertices[2], 0]) -
                     (points[vertices[1], 0] - points[vertices[2], 0]) * (
                                 points[vertices[0], 1] - points[vertices[2], 1])),

                    1 - ((y_grid[i, j] - points[vertices[0], 1]) * (points[vertices[1], 0] - points[vertices[0], 0]) -
                         (x_grid[i, j] - points[vertices[0], 0]) * (points[vertices[1], 1] - points[vertices[0], 1])) /
                    ((points[vertices[2], 1] - points[vertices[0], 1]) * (
                                points[vertices[1], 0] - points[vertices[0], 0]) -
                     (points[vertices[2], 0] - points[vertices[0], 0]) * (
                                 points[vertices[1], 1] - points[vertices[0], 1]))
                ])
                z_grid[i, j] = np.sum(weights * points[vertices, 2])

    return x_grid, y_grid, z_grid


def save_as_tif(x_grid, y_grid, z_grid, output_file):
    transform = from_bounds(
        x_grid.min(), y_grid.min(), x_grid.max(), y_grid.max(), x_grid.shape[1], x_grid.shape[0]
    )

    with rio_open(
            output_file,
            'w',
            driver='GTiff',
            height=z_grid.shape[0],
            width=z_grid.shape[1],
            count=1,
            dtype=z_grid.dtype,
            crs='+proj=latlong',
            transform=transform,
    ) as dst:
        dst.write(z_grid, 1)


# Directory containing the .las files
directory = "../__rawdata__"

# Merge the .las files
merged_points = merge_las_files(directory)

# Perform TIN interpolation
resolution = 1.0  # Specify the desired resolution
x_grid, y_grid, z_grid = tin_interpolation(merged_points, resolution)

# Save the result as a tif file
output_file = "output.tif"
save_as_tif(x_grid, y_grid, z_grid, output_file)