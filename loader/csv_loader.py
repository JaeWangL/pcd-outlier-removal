import pandas as pd


class CsvLoader:
    def __init__(self, filepath: str, is_reference: bool = True) -> None:
        """
        Initializes the loader with the path to the CSV file and a flag indicating whether it's reference data.

        Parameters:
            filepath: str - The path to the CSV file to be loaded.
            is_reference: bool - Flag indicating whether the CSV data is reference data.
        """
        self.filepath = filepath
        self.is_reference = is_reference

    def load_to_dataframe(self) -> pd.DataFrame:
        """
        Loads the CSV file into a pandas DataFrame and asserts the presence of required columns.

        Returns:
            pd.DataFrame: A DataFrame containing the necessary columns from the CSV file.

        Raises:
            AssertionError: If the DataFrame does not contain the required columns.
        """
        df = pd.read_csv(self.filepath)

        # Check if the data is reference data and assert the required columns
        if self.is_reference:
            required_columns = ['GCP', 'E', 'N', 'h']
        else:
            required_columns = ['E', 'N', 'h']

        assert all(column in df.columns for column in
                   required_columns), f"DataFrame missing required columns: {required_columns}"

        return df
