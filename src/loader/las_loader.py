import logging
import laspy
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LasLoader:
    """A class for loading and saving LAS files."""

    def __init__(self, filepath: str | None = None) -> None:
        """Initializes the LasLoader with the path to the LAS file.

        Args:
            filepath (str | None): The path to the LAS file to be loaded.
        """
        self.filepath = filepath
        self.dataframe = None

    def load_to_dataframe(self) -> pd.DataFrame:
        """Loads the LAS file and converts it to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing 'E', 'N', and 'h' columns.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        try:
            # Open and read the LAS file
            with laspy.open(self.filepath) as file:
                las = file.read()

            # Extract X, Y, Z coordinates
            E = np.array(las.x)
            N = np.array(las.y)
            h = np.array(las.z)

            # Create a DataFrame from the coordinate data
            data = {'E': E, 'N': N, 'h': h}
            self.dataframe = pd.DataFrame(data)

            logger.info(f"Loaded LAS file into DataFrame: {self.filepath}")
            return self.dataframe

        except FileNotFoundError as e:
            logger.error(f"File not found: {self.filepath}")
            raise e

    def save_to_las(self, output_filepath: str, df: pd.DataFrame = None) -> None:
        """Saves the DataFrame to a LAS file.

        Args:
            output_filepath (str): The path to the output LAS file.
            df (pd.DataFrame, optional): The DataFrame to save. If not provided, uses self.dataframe.

        Raises:
            ValueError: If no DataFrame is provided or if required columns are missing.
        """
        # Use the provided DataFrame or the one stored in the instance
        if df is not None:
            dataframe = df
            logger.info("Using provided DataFrame.")
        elif self.dataframe is not None:
            dataframe = self.dataframe
            logger.info("Using instance's DataFrame.")
        else:
            logger.error("No DataFrame provided.")
            raise ValueError("No DataFrame provided.")

        # Check if required columns are present
        required_columns = {'E', 'N', 'h'}
        if not required_columns.issubset(dataframe.columns):
            logger.error(f"The DataFrame must contain the following columns: {required_columns}")
            raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")

        # Create a LAS header with appropriate scaling
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scale = [0.01, 0.01, 0.01]
        logger.info("Created LAS file header.")

        # Create a new LAS data object
        las = laspy.LasData(header)

        # Assign coordinates from the DataFrame to the LAS data object
        las.x = dataframe['E'].to_numpy()
        las.y = dataframe['N'].to_numpy()
        las.z = dataframe['h'].to_numpy()
        logger.info("Assigned coordinate data to LAS object.")

        # Write the LAS data to a file
        las.write(output_filepath)
        logger.info(f"Saved DataFrame to LAS file: {output_filepath}")

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Sets the instance's DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to set.
        """
        self.dataframe = dataframe
        logger.info("DataFrame has been set.")