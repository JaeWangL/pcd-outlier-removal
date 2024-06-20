import laspy
import numpy as np
import pandas as pd


class LasLoader:
    def __init__(self, filepath: str) -> None:
        """
        Initializes the loader with the path to the LAS file.

        Parameters:
            filepath: str - The path to the LAS file to be loaded.
        """
        self.filepath = filepath

    def load_to_dataframe(self) -> pd.DataFrame:
        """
        Loads the LAS file and converts it to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the 'E', 'N', and 'h' columns from the LAS file.
        """
        with laspy.open(self.filepath) as file:
            las = file.read()

            # Extract X, Y, Z coordinates and convert them to NumPy arrays
        E = np.array(las.x)
        N = np.array(las.y)
        h = np.array(las.z)

        # Combine arrays into a dictionary
        data = {'E': E, 'N': N, 'h': h}

        df = pd.DataFrame(data)

        return df

    def save_to_las(self, output_filepath: str) -> None:
        """
        Saves the DataFrame to a LAS file.

        The DataFrame must contain columns 'E', 'N', and 'h', corresponding to the easting, northing, and height.
        """
        header = laspy.LasHeader(version="1.2", point_format=3)
        header.x_scale = 0.01
        header.y_scale = 0.01
        header.z_scale = 0.01

        # Create a new LAS file with the given header
        with laspy.create(header, output_filepath) as las:
            # Ensure the DataFrame columns are in the correct format
            E = self.dataframe['E'].to_numpy()
            N = self.dataframe['N'].to_numpy()
            h = self.dataframe['h'].to_numpy()

            # Assigning coordinates to points
            las.x = E
            las.y = N
            las.z = h