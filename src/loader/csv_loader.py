import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CsvLoader:
    """A class for loading CSV files into pandas DataFrames."""

    def __init__(self, filepath: str, is_reference: bool = True) -> None:
        """Initializes the CsvLoader with the path to the CSV file.

        Args:
            filepath (str): The path to the CSV file to be loaded.
            is_reference (bool, optional): Indicates if the CSV data is reference data with a 'GCP' column.
                Defaults to True.
        """
        self.filepath = filepath
        self.is_reference = is_reference

    def load_to_dataframe(self) -> pd.DataFrame:
        """Loads the CSV file into a pandas DataFrame and checks for required columns.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the CSV file.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            ValueError: If the required columns are missing from the CSV file.
        """
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(self.filepath)
            logger.info(f"Loaded CSV file: {self.filepath}")
        except FileNotFoundError as e:
            logger.error(f"CSV file not found: {self.filepath}")
            raise e

        # Determine required columns based on whether it's reference data
        if self.is_reference:
            required_columns = {'GCP', 'E', 'N', 'h'}
        else:
            required_columns = {'E', 'N', 'h'}

        # Check if all required columns are present in the DataFrame
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            error_msg = f"The following required columns are missing from the CSV file: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"All required columns are present: {required_columns}")
        return df