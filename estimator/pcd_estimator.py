import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from typing import Literal, Tuple
from visualizer.pcd_visualizer import PCDVisualizer

from config import DEBUG


class PCDEstimator:
    def __init__(self, df_reference: pd.DataFrame, df_raw: pd.DataFrame, df_target1: pd.DataFrame, df_target2: pd.DataFrame) -> None:
        """
        Initializes the estimator with reference and target dataframes.

        Parameters:
            df_reference: pd.DataFrame - The reference dataframe containing 'GCP', 'E', 'N', and 'h' columns.
            df_target1: pd.DataFrame - The first target dataframe containing 'E', 'N', and 'h' columns.
            df_target2: pd.DataFrame - The second target dataframe containing 'E', 'N', and 'h' columns.
        """
        self.df_reference = df_reference
        self.df_raw = df_raw
        self.df_target1 = df_target1
        self.df_target2 = df_target2

    def main(self) -> Tuple[float, float, float, Literal['target1', 'target2']]:
        """
        Analyzes and compares the height differences between the reference and each target dataframe.

        Returns:
            Tuple[float, float, float, Literal['target1', 'target2']]: A dictionary containing the mean differences for each target and the identifier of the better target.
        """
        # Compare heights for each target and get the resulting dataframes
        compare_raw = self._compare_heights(self.df_raw, 'Raw')
        compare_df1 = self._compare_heights(self.df_target1, 'Target - 1')
        compare_df2 = self._compare_heights(self.df_target2, 'Target - 2')

        # Calculate the mean height difference for each target to assess overall deviation
        mean_raw = compare_raw['h_diff'].mean()
        mean_diff1 = compare_df1['h_diff'].mean()
        mean_diff2 = compare_df2['h_diff'].mean()

        # Determine which target has the smaller mean difference (closer to the reference)
        min_value, better_target = min((mean_diff1, 'target1'), (mean_diff2, 'target2'), (mean_raw, 'raw'))

        return mean_raw, mean_diff1, mean_diff2, better_target

    def _validate_dataframes(self):
        """
        Ensure all required columns are present in the dataframes.
        reference dataframe have to have the columns ["GCP", "E", "N", "h"]
        target dataframe have to have the columns ["E", "N", "h"]

        Raises:
            AssertionError: If any of the dataframes do not contain the required columns.
        """
        assert 'GCP' in self.df_reference and 'E' in self.df_reference and 'N' in self.df_reference and 'h' in self.df_reference, "Reference dataframe missing required columns"
        assert 'E' in self.df_raw and 'N' in self.df_raw and 'h' in self.df_raw, "Raw dataframe missing required columns"
        assert 'E' in self.df_target1 and 'N' in self.df_target1 and 'h' in self.df_target1, "Target1 dataframe missing required columns"
        assert 'E' in self.df_target2 and 'N' in self.df_target2 and 'h' in self.df_target2, "Target2 dataframe missing required columns"

    def _interpolate_data(self, df: pd.DataFrame, method: Literal['linear', 'nearest', 'cubic'] = 'nearest',
                          max_points: int = 100000) -> pd.DataFrame:
        """
        Interpolates any given dataset within the extents of the reference data, optimized to avoid memory issues.

        Parameters:
            df: pd.DataFrame - The dataframe to be interpolated.
            method: str - The interpolation method; defaults to 'nearest'.
            max_points: int - The maximum number of points to interpolate to avoid memory issues.

        Returns:
            pd.DataFrame: A new dataframe with interpolated 'h' values.
        """
        # Extents based on reference data for consistency
        min_E, max_E = self.df_reference['E'].min(), self.df_reference['E'].max()
        min_N, max_N = self.df_reference['N'].min(), self.df_reference['N'].max()

        # Define the grid resolution based on max_points while maintaining aspect ratio
        total_range_E = max_E - min_E
        total_range_N = max_N - min_N
        aspect_ratio = total_range_E / total_range_N
        num_intervals_E = int((max_points * aspect_ratio) ** 0.5)
        num_intervals_N = int(max_points / num_intervals_E)

        grid_x, grid_y = np.mgrid[min_E:max_E:num_intervals_E * 1j, min_N:max_N:num_intervals_N * 1j]

        # Perform the interpolation
        grid_h = griddata((df['E'], df['N']), df['h'], (grid_x, grid_y), method=method)

        # Create dataframe for interpolated points
        interpolated_df = pd.DataFrame({
            'E': grid_x.flatten(),
            'N': grid_y.flatten(),
            'h': grid_h.flatten()
        }).dropna()  # Remove any NaN values which are out of interpolation bounds

        return interpolated_df

    def _compare_heights(self, df_target: pd.DataFrame, target_label: str) -> pd.DataFrame:
        """
        Compares the heights between the reference and a target dataframe.

        Parameters:
            df_target: pd.DataFrame - The target dataframe to be compared with the reference.

        Returns:
            pd.DataFrame: A dataframe containing the original 'E', 'N' coordinates and the height differences 'h_diff'.
        """
        df_reference_interpolated = self._interpolate_data(self.df_reference)
        df_target_interpolated = self._interpolate_data(df_target)

        # Merge the interpolated datasets on E and N coordinates
        merged_df = pd.merge(df_reference_interpolated, df_target_interpolated, on=['E', 'N'], how='inner', suffixes=('_ref', '_target'))
        merged_df['h_diff'] = np.abs(merged_df['h_target'] - merged_df['h_ref'])

        if DEBUG:
            PCDVisualizer(df_reference_interpolated, 'Reference', df_target_interpolated, target_label).visualize()

        return merged_df
