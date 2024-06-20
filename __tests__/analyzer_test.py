from estimator.pcd_estimator import PCDEstimator
from loader.csv_loader import CsvLoader
from loader.las_loader import LasLoader
from removal.outlier_removal import OutlierRemoval

df_reference = CsvLoader('../__reference__/reference.csv').load_to_dataframe()
df_raw = LasLoader('../__rawdata__2/Seahawk_231015_231341_00_D.las').load_to_dataframe()
df_target = LasLoader('../__testdata__2/Seahawk_231015_231341_00_D.las').load_to_dataframe()

estimator = PCDEstimator(df_reference, df_raw, OutlierRemoval(df_raw).main(), df_target)
mean_diff_raw, mean_diff_mine, mean_diff_target, better_target = estimator.main()
print(mean_diff_raw, mean_diff_mine, mean_diff_target, better_target)
