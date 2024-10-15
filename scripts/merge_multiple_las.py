import os
import glob
import logging
import pandas as pd

from src.loader.las_loader import LasLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    las_dir = '../__rawdata__'
    las_files = glob.glob(os.path.join(las_dir, '*_D.las'))

    if not las_files:
        logger.error("No LAS files found in the specified directory.")
        return

    logger.info(f"Found {len(las_files)} LAS files.")

    # List to hold dataframes
    dataframes = []

    # Load each LAS file into a dataframe
    for las_file in las_files:
        logger.info(f"Loading LAS file: {las_file}")
        loader = LasLoader(las_file)
        df = loader.load_to_dataframe()
        dataframes.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info("All LAS files have been loaded and concatenated.")

    # Create a new LasLoader instance for saving
    output_loader = LasLoader(filepath=None)
    output_loader.set_dataframe(combined_df)

    # Specify the output file path
    output_filepath = os.path.join(las_dir, 'merged_output.las')  # Adjust as needed

    # Save the combined dataframe to a new LAS file
    output_loader.save_to_las(output_filepath)
    logger.info(f"Merged LAS file has been saved to {output_filepath}")


if __name__ == "__main__":
    main()
