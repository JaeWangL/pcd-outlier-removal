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
