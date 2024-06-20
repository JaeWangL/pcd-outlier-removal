import pandas as pd
import laspy

import pandas as pd
import laspy
import numpy as np


def csv_to_las(csv_filepath: str, las_filepath: str) -> None:
    """
    Converts a CSV file with columns ['GCP', 'E', 'N', 'h'] to a LAS file, mapping 'GCP' strings to integer IDs.

    Parameters:
        csv_filepath: str - The file path of the source CSV file.
        las_filepath: str - The file path where the LAS file will be saved.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_filepath)

    # Prepare data
    data = df[['E', 'N', 'h']].values

    # Map GCP strings to unique integers
    unique_gcp, gcp_indices = np.unique(df['GCP'], return_inverse=True)

    # Create a new header and specify the scales and offsets
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="GCP_id", type=np.uint32))
    header.offsets = np.min(data, axis=0)
    header.scales = [0.01, 0.01, 0.01]

    # Create LasData with the header
    las = laspy.LasData(header)

    # Set the coordinates
    las.x = df['E'].values
    las.y = df['N'].values
    las.z = df['h'].values

    # Set the GCP ID
    las['GCP_id'] = gcp_indices.astype(np.uint32)

    # Write the LAS file
    las.write(las_filepath)

    print(f"LAS file has been saved to {las_filepath}")

csv_to_las("./reference.csv", "./reference.las")