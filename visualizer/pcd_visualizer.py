import pandas as pd
import open3d as o3d


class PCDVisualizer:
    def __init__(self, df_1: pd.DataFrame, df_1_label: str, df_2: pd.DataFrame | None = None, df_2_label: str | None = None) -> None:
        self.df_1 = df_1
        self.df_1_label = df_1_label
        self.df_2 = df_2
        self.df_2_label = df_2_label

    def visualize(self) -> None:
        # Convert dataframes to Open3D point clouds for visualization
        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(self.df_1[['E', 'N', 'h']].values)
        if self.exist_df2():
            pcd_2 = o3d.geometry.PointCloud()
            pcd_2.points = o3d.utility.Vector3dVector(self.df_2[['E', 'N', 'h']].values)

        # Set colors for differentiation
        if self.exist_df2():
            pcd_1.paint_uniform_color([0, 1, 0])  # Green for data1
            pcd_2.paint_uniform_color([0, 0, 1])  # Blue for data2

        # Prepare visualization
        vis = o3d.visualization.Visualizer()
        df_2_part = f' vs {self.df_2_label}' if self.df_2_label is not None else ''
        vis.create_window(window_name=f'Comparison: {self.df_1_label}{df_2_part}', width=1200, height=800)
        vis.add_geometry(pcd_1)
        if self.exist_df2():
            vis.add_geometry(pcd_2)
        vis.run()
        vis.destroy_window()

    def exist_df2(self) -> bool:
        return self.df_2 is not None and self.df_2_label is not None
